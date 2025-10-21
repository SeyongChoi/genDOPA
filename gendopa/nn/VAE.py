import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

class VAE(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, pad_idx,
                 hidden_dim_encoder, num_layers_encoder, dropout_encoder, latent_dim, 
                 weight_init:str ='xavier_uniform',
                 lr: float = 1e-3,
                 beta: float= 4.0,
                 optimizer: str = 'adam',
                 weight_decay: float = 0.0
                 ):
        
        super(VAE, self).__init__()
        self.save_hyperparameters()  # 하이퍼파라미터 저장
        self.weight_init = weight_init

        #### Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=pad_idx)
        
        #### Encoder
        # extract the features from input sequence using a bidirectional GRU (past and future)
        # the final hidden states from both directions are concatenated
        # to form the encoded representation --> the reason for hidden_dim//2
        self.encoder = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_dim_encoder//2,
                              num_layers=num_layers_encoder,
                              batch_first=True,
                              dropout=dropout_encoder if num_layers_encoder > 1 else 0,
                              bidirectional=True)
        
        #### For latent vector
        self.to_mu = nn.Linear(hidden_dim_encoder, latent_dim)      # nn.Sequential OK
        self.to_logvar = nn.Linear(hidden_dim_encoder, latent_dim)  # nn.Sequential OK

        #### To map latent vector to initial hidden state of decoder
        # - The GRU decoder expects an initial hidden state of shape [num_layers, batch_size, hidden_dim].
        # - Since the latent vector z has shape [batch_size, latent_dim], we need to project it
        #   into a vector large enough to fill all GRU layers' hidden states.
        # - Therefore, we use a Linear layer that outputs num_layers * hidden_dim values per sample,
        #   which will later be reshaped into [num_layers, batch_size, hidden_dim].
        self.latent_to_hidden = nn.Linear(latent_dim, num_layers_encoder*hidden_dim_encoder)  # nn.Sequential OK
        
        #### Decoder
        # predict the next token given the previous tokens and latent vector in unidirectional GRU
        self.decoder = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_dim_encoder,
                              num_layers=num_layers_encoder,
                              batch_first=True,
                              dropout=dropout_encoder if num_layers_encoder > 1 else 0)
        
        #### Output layer to vocab        
        self.outputs_to_vocab = nn.Linear(hidden_dim_encoder, vocab_size) # nn.Sequential OK (do not deep)

        # 가중치 초기화 적용
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        """
        레이어별 가중치 초기화 함수.
        self.weight_init 값에 따라 초기화 방식을 분기 처리함.
        
        Parameters
        ----------
        m : nn.Module
            초기화 대상 레이어 (주로 nn.Linear).
        """
        if isinstance(m, nn.Linear):
            if self.weight_init == 'xavier_uniform':
                init.xavier_uniform_(m.weight)
            elif self.weight_init == 'xavier_normal':
                init.xavier_normal_(m.weight)
            elif self.weight_init in ['he_uniform', 'kaiming_uniform']:
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif self.weight_init in ['he_normal', 'kaiming_normal']:
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif self.weight_init == 'orthogonal':
                init.orthogonal_(m.weight)
            elif self.weight_init == 'default':
                # PyTorch 기본 초기화 (변경하지 않음)
                pass
            else:
                raise ValueError(f"Unknown weight_init method: {self.weight_init}")

            if m.bias is not None:
                init.zeros_(m.bias)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from N(mu, var) from N(0,I).
        # mu: [B, L], logvar: [B, L]
        # std: standard deviation
        std = torch.exp(0.5 * logvar)
        # eps: random normal tensor
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, mask=None):
        ### B: batch size, T: token length, D: embedding dim,
        ### H: hidden dim, L: latent dim, V: vocab size
        # embedding
        if mask is None: mask = (x != self.hparams.pad_idx).long()
        x_emb = self.embedding(x)                     # [B, T, D]
        # encoder
        enc_out, _ = self.encoder(x_emb)              # [B, T, H]
        # masked mean pooling
        # - mask.sum(1): compute the number of valid (non-PAD) tokens for each sequence in the batch → [B]
        # - .clamp(min=1): prevent division by zero when a sequence has only PAD tokens (set minimum length = 1)
        # - .unsqueeze(1): expand to [B, 1] for broadcasting with the hidden dimension later
        lens = mask.sum(1).clamp(min=1).unsqueeze(1)  # [B,1]
        # - mask.unsqueeze(2): reshape mask to [B, T, 1] so it can broadcast over enc_out [B, T, H]
        # - enc_out * mask.unsqueeze(2): zero out hidden states corresponding to PAD tokens
        # - .sum(1): sum over the token dimension (T) → gives one vector per sequence [B, H]
        # - / lens: divide by the number of valid tokens to get the mean (exclude PADs)
        enc_repr = (enc_out * mask.unsqueeze(2)).sum(1) / lens  # [B, H]
        # latent vector
        mu, logvar = self.to_mu(enc_repr), self.to_logvar(enc_repr) 
        z = self.reparameterize(mu, logvar)          # [B, L]

        # prepare decoder inputs and targets
        # - x_in: input sequence to the decoder (all tokens except the last one)
        # - tgt:  target sequence for prediction (all tokens except the first one)
        #   → this implements "teacher forcing", where the decoder learns to predict
        #     the next token given all previous ones.
        x_in = x[:, :-1]
        tgt = x[:, 1:]
        dec_in = self.embedding(x_in)               # [B, T-1, D]
        # initialize decoder hidden state from latent vector
        hidden_0 = self.latent_to_hidden(z).view(self.hparams.num_layers_encoder, x.size(0), self.hparams.hidden_dim_encoder).contiguous()
        # decoder
        dec_out, _ = self.decoder(dec_in, hidden_0)   # [B, T-1, H]
        # output layer
        logits = self.outputs_to_vocab(dec_out)      # [B, T-1, V]
        return logits, tgt, mu, logvar
    
    def loss_fn(self, logits, targets, mu, logvar):
        # reconstruction loss
        ce = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.hparams.pad_idx)
        
        # KL divergence loss
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # KL annealing
        # step = max(1, self.global_step)
        beta = self.hparams.beta
        
        # total loss
        loss = ce + beta * kl
        return loss, ce.item(), kl.item(), beta

    def training_step(self, batch, _):
        x, mask = batch
        logits, tgt, mu, logvar = self.forward(x, mask)
        loss, ce, kl, beta = self.loss_fn(logits, tgt, mu, logvar)
        self.log_dict({"train/loss":loss, "train/ce":ce, "train/kl":kl, "train/beta":beta}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, mask = batch
        logits, tgt, mu, logvar = self(x, mask)
        loss, ce, kl, beta = self.loss_fn(logits, tgt, mu, logvar)
        self.log_dict({"val/loss":loss, "val/ce":ce, "val/kl":kl, "val/beta":beta}, prog_bar=True)

    def configure_optimizers(self):
        """
        최적화 알고리즘을 설정하는 함수.
        Returns
        -------
        torch.optim.Optimizer
            설정된 최적화 알고리즘.
        """
        optimizer_type = self.hparams.optimizer.lower()
        lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay  # <-- 여기서 가져옴

        if optimizer_type == 'adam':
            return optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    @torch.no_grad()
    def sample(self, max_len=128, batch_size=4, temperature=1.0, top_k=0, z=None, bos_idx=1, eos_idx=2):
        device = self.device
        if z is None:
            z = torch.randn(batch_size, self.hparams.latent_dim, device=device)
        h = self.latent_to_hidden(z).view(self.hparams.num_layers_encoder, batch_size, self.hparams.hidden_dim_encoder).contiguous()

        inp = torch.full((batch_size,1), bos_idx, dtype=torch.long, device=device)
        seq = [inp]
        for _ in range(max_len):
            emb = self.embedding(inp)           # [B,1,D]
            out, h = self.decoder(emb, h)   # [B,1,H]
            logits = self.outputs_to_vocab(out[:,-1,:])  # [B,V]
            logits = logits / max(1e-6, temperature)

            if top_k>0:
                v, ix = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter_(1, ix, F.softmax(v, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)

            nxt = torch.multinomial(probs, 1)         # [B,1]
            seq.append(nxt)
            inp = nxt
            if (nxt == eos_idx).all(): break
        return torch.cat(seq, dim=1)  # [B,T']
    
if __name__=="__main__":
    import os
    import selfies as sf
    from gendopa.reader import MolDataReader
    from gendopa.encoder import Encoder
    from gendopa.loader import split_n_load

    data_root_dir = 'D:\\genDOPA\\data\\'
    data_fpath = os.path.join(data_root_dir, "adsorption_property.csv")
    vocab_fpath = os.path.join(data_root_dir, "vocab.csv")
    reader = MolDataReader(data_fpath, vocab_fpath=vocab_fpath)

    print(reader.dataset.head())

    MolDataset = reader.read(save_result=False)
    encoder = Encoder(dataset=MolDataset, represent='gsf')
    
    vocab = encoder.alphabet
    vocab_stoi = encoder.vocab_stoi
    vocab_itos = encoder.vocab_itos

    train_loader, val_loader, test_loader = split_n_load(
        dataset=MolDataset,
        batch_size=1)
    
    import pytorch_lightning as pl

    model = VAE(vocab_size=len(vocab),
                embedding_dim=128,
                pad_idx=vocab_stoi['[nop]'],
                hidden_dim_encoder=256,
                num_layers_encoder=3,
                dropout_encoder=0.1,
                latent_dim=64,
                weight_init='xavier_uniform',
                lr=1e-3,
                beta=0.05)
    trainer = pl.Trainer(max_epochs=50, gradient_clip_val=1.0)
    trainer.fit(model, train_loader, val_loader)
    sampled = model.sample()[0].detach().cpu().tolist()
    print(sampled)
    print(sf.encoding_to_selfies(sampled, 
                                 vocab_itos=vocab_itos, enc_type='label'))
    
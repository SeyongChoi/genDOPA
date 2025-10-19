import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class VAE(pl.LightningModule):
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 128,
                 pad_idx: int = 0,
                 hidden_dim: int = 256,
                 latent_dim: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-3,
                 kl_anneal_steps: int = 2500,
                 tie_weights: bool = True, 
                 ):
        
        super(VAE, self).__init__()
        self.save_hyperparameters()  # 하이퍼파라미터 저장

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=pad_idx)
        
        self.encoder = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_dim//2,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
        
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, num_layers*hidden_dim)
        self.decoder = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
        
        self.outputs_to_vocab = nn.Linear(hidden_dim, vocab_size)
        if tie_weights:
            self.outputs_to_vocab.weight = self.embedding.weight

        self.kl_anneal_steps = kl_anneal_steps
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, mask=None):
        if mask is None: mask = (x != self.hparams.pad_idx).long()
        x_emb = self.embedding(x)                     # [B, T, D]
        enc_out, _ = self.encoder(x_emb)              # [B, T, H]
        lens = mask.sum(1).clamp(min=1).unsqueeze(1)  # [B,1]
        enc_repr = (enc_out * mask.unsqueeze(2)).sum(1) / lens  # [B, H]

        mu, logvar = self.to_mu(enc_repr), self.to_logvar(enc_repr) 
        z = self.reparameterize(mu, logvar)          # [B, L]

        x_in = x[:, :-1]
        tgt = x[:, 1:]
        dec_in = self.embedding(x_in)               # [B, T-1, D]
        hidden0 = self.latent_to_hidden(z).view(self.num_layers, x.size(0), self.hidden_dim).contiguous()
        dec_out, _ = self.decoder(dec_in, hidden0)   # [B, T-1, H]
        logits = self.outputs_to_vocab(dec_out)      # [B, T-1, V]
        return logits, tgt, mu, logvar
    
    def loss_fn(self, logits, targets, mu, logvar):
        ce = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.hparams.pad_idx)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        step = max(1, self.global_step)
        beta = min(1.0, step/float(self.kl_anneal_steps))
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
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    @torch.no_grad()
    def sample(self, max_len=128, batch_size=4, temperature=1.0, top_k=0, z=None, bos_idx=1, eos_idx=2):
        device = self.device
        if z is None:
            z = torch.randn(batch_size, self.hparams.latent_dim, device=device)
        h = self.latent_to_hidden(z).view(self.num_layers, batch_size, self.hidden_dim).contiguous()

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
                hidden_dim=256,
                latent_dim=64,
                num_layers=3,
                dropout=0.1,
                learning_rate=1e-3,
                kl_anneal_steps=2500,
                tie_weights=False)
    trainer = pl.Trainer(max_epochs=10, gradient_clip_val=1.0)
    trainer.fit(model, train_loader, val_loader)
    print(model.sample()[0].detach().cpu().tolist())
    print(sf.encoding_to_selfies(model.sample()[0].detach().cpu().tolist(), 
                                 vocab_itos=vocab_itos, enc_type='label'))
import torch
from torch.utils.data import Dataset

import pandas as pd
from typing import List, Optional, Literal
from gendopa.molinfo import AdsMolData
from gendopa.encoder import Encoder

class AdsMolDataset(Dataset):
    def __init__(self, 
                 dataset: Optional[List[AdsMolData]] = None,
                 dataframe: Optional[pd.DataFrame] = None,
                 represent: Literal['gsf', 'selfies', 'smiles'] = 'gsf',
                 enc_type: Literal['label', 'one_hot'] = 'label'):
        
        self.encoder = Encoder(dataset=dataset,
                               dataframe=dataframe,
                               represent=represent)
        self.repr_set = self.encoder.repr_set
        self.vocab_stoi = self.encoder.vocab_stoi
        self.max_len = self.encoder.max_len
        self.enc_type = enc_type

    def __len__(self):
        return len(self.repr_set)
    
    def __getitem__(self, idx):

        repr = self.repr_set[idx]
        
        encoding = self.encoder.repr_to_encoding(repr=repr,
                                                 vocab_stoi=self.vocab_stoi,
                                                 pad_to_len=self.max_len,
                                                 enc_type=self.enc_type)
        encoding = torch.tensor(encoding, dtype=torch.long)
        return encoding
        
        
if __name__=="__main__":
    import os
    from gendopa.reader import MolDataReader
    from torch import nn

    data_root_dir = 'D:\\genDOPA\\data\\'
    data_fpath = os.path.join(data_root_dir, "adsorption_property.csv")
    vocab_fpath = os.path.join(data_root_dir, "vocab.csv")
    reader = MolDataReader(data_fpath, vocab_fpath=vocab_fpath)

    print(reader.dataset.head())

    MolDataset = reader.read(save_result=False)

    encoder = Encoder(dataset=MolDataset, represent='gsf')
    print(encoder.vocab_stoi)
    print(encoder.max_len)

    dataset = AdsMolDataset(dataset=MolDataset,
                           represent='gsf',
                           enc_type='label')
    print(len(dataset))
    print(dataset[0])

    embedding_layer=nn.Embedding(num_embeddings=len(encoder.alphabet),
                 embedding_dim=4,
                 padding_idx=encoder.vocab_stoi["[nop]"])

    print(embedding_layer(dataset[0]))
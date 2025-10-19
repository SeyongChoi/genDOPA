import pandas as pd
from typing import List, Optional, Literal

import selfies as sf
from gendopa.molinfo import AdsMolData


class Encoder:
    def __init__(self,
                 dataset: Optional[List[AdsMolData]] = None,
                 dataframe: Optional[pd.DataFrame] = None,
                 represent: Literal['gsf', 'selfies', 'smiles'] = 'gsf'):
        self.dataset = dataset
        self.dataframe = dataframe
        self.represent = represent
        self.repr_set = None
        self.alphabet = None

        self.repr_set = self._generate_repr_set()
        if self.repr_set is not None:
            self.max_len = max(sf.len_selfies(s) for s in self.repr_set)
            self.alphabet = self._generate_alphabet()
            self.vocab_stoi = {s:i for i,s in enumerate(self.alphabet)}
            self.vocab_itos = {i:s for i,s in enumerate(self.alphabet)}

    def _generate_repr_set(self) -> List[str]:
        # mapping attribute/columns
        rep2attr = {
            'gsf':     ('GroupSELFIES', 'GroupSELFIES'),
            'selfies': ('SELFIES',      'SELFIES'),
            'smiles':  ('SMILES',       'SMILES'),
        }
        if self.represent not in rep2attr:
            raise ValueError(f"represent must be one of {list(rep2attr.keys())}, got: {self.represent}")
        
        ds_attr, df_col = rep2attr[self.represent]

        # only Dataset
        if self.dataset is not None and self.dataframe is None:
            missing = [i for i,m in enumerate(self.dataset) if not hasattr(m, ds_attr)]
            if missing:
                raise AttributeError(f"Some AdsMolData items are missing attribute '{ds_attr}': indices {missing[:5]}...")
            repr_set = [getattr(ads_mol, ds_attr) for ads_mol in self.dataset]
        # only DataFrame
        elif self.dataset is None and self.dataframe is not None:
            if df_col not in self.dataframe.columns:
                raise KeyError(f"DataFrame must contain column '{df_col}'. Have: {list(self.dataframe.columns)}")
            repr_set = self.dataframe[df_col].astype(str).tolist()

        # both Dataset/DataFrame
        elif self.dataset is not None and self.dataframe is not None:
            missing = [i for i,m in enumerate(self.dataset) if not hasattr(m, ds_attr)]
            if missing:
                raise AttributeError(f"Some AdsMolData items are missing attribute '{ds_attr}': indices {missing[:5]}...")
            ds_list = [getattr(ads_mol, ds_attr) for ads_mol in self.dataset]

            if df_col not in self.dataframe.columns:
                raise KeyError(f"DataFrame must contain column '{df_col}'. Have: {list(self.dataframe.columns)}")
            df_list = self.dataframe[df_col].astype(str).tolist()
            
            if len(ds_list) != len(df_list):
                raise ValueError(f"Size mismatch between dataset({len(ds_list)}) and dataframe({len(df_list)}).")

            if ds_list != df_list:
                raise ValueError("Content mismatch between dataset and dataframe.")

            repr_set = ds_list
        # neither Dataset/DataFrame
        else:
            raise ValueError("Either 'dataset' or 'dataframe' must be provided.")
        
        return repr_set

    def _generate_alphabet(self) -> List[str]:    
        alphabet = sf.get_alphabet_from_selfies(self.repr_set)
        alphabet.discard("[nop]")
        others = sorted(alphabet)


        return ["[nop]"] + others

    def _repr_to_encoding(self,
                          repr: str,
                          vocab_stoi: dict,
                          pad_to_len: int,
                          enc_type: Literal['one_hot','label','both']='both'):
        
        if enc_type not in ['one_hot','label','both']:
            raise ValueError(f"enc_type must be one of ['one_hot','label','both'], got: {enc_type}")
        
        if enc_type == 'label':
            label = sf.selfies_to_encoding(selfies=repr,
                                           vocab_stoi=vocab_stoi,
                                           pad_to_len=pad_to_len,
                                           enc_type=enc_type)
            return label
        elif enc_type == 'one_hot':
            one_hot = sf.selfies_to_encoding(selfies=repr,
                                            vocab_stoi=vocab_stoi,
                                            pad_to_len=pad_to_len,
                                            enc_type=enc_type)
            return one_hot
        else:  # both
            label, one_hot = sf.selfies_to_encoding(selfies=repr,
                                                   vocab_stoi=vocab_stoi,
                                                   pad_to_len=pad_to_len,
                                                   enc_type=enc_type)
            return label, one_hot
        
    def encoding(self,
                 enc_type: Literal['one_hot','label','both']='both') -> List:
        if enc_type not in ['one_hot','label','both']:
            raise ValueError(f"enc_type must be one of ['one_hot','label','both'], got: {enc_type}")
        
        encoding_labels = []
        encoding_one_hots = []
        if enc_type == 'label':
            for repr in self.repr_set:
                encoding = self._repr_to_encoding(repr=repr,
                                                 vocab_stoi=self.vocab_stoi,
                                                 pad_to_len=self.max_len,
                                                 enc_type=enc_type)
                encoding_labels.append(encoding)
            return encoding_labels
        elif enc_type == 'one_hot':
            for repr in self.repr_set:
                encoding = self._repr_to_encoding(repr=repr,
                                                 vocab_stoi=self.vocab_stoi,
                                                 pad_to_len=self.max_len,
                                                 enc_type=enc_type)
                encoding_one_hots.append(encoding)
            return encoding_one_hots
        else:  # both
            for repr in self.repr_set:
                label, one_hot = self._repr_to_encoding(repr=repr,
                                                       vocab_stoi=self.vocab_stoi,
                                                       pad_to_len=self.max_len,
                                                       enc_type=enc_type)
                encoding_labels.append(label)
                encoding_one_hots.append(one_hot)
            return encoding_labels, encoding_one_hots
    
if __name__ == "__main__":
    import os
    from gendopa.reader import MolDataReader

    data_root_dir = 'D:\\genDOPA\\data\\'
    data_fpath = os.path.join(data_root_dir, "adsorption_property.csv")
    vocab_fpath = os.path.join(data_root_dir, "vocab.csv")
    reader = MolDataReader(data_fpath, vocab_fpath=vocab_fpath)

    print(reader.dataset.head())

    AdsMolDataset = reader.read(save_result=False)

    encoder = Encoder(dataset=AdsMolDataset, represent='gsf')
    print(encoder.alphabet)
    labels = encoder.encoding(enc_type='label')
    print(labels[:2])
    one_hots = encoder.encoding(enc_type='one_hot')
    print(one_hots[:2])

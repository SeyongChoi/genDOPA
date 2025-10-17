import os
from tqdm import tqdm
import pandas as pd
from typing import List, Optional, Literal

import selfies as sf
from group_selfies import Group, GroupGrammar
from gendopa.molinfo import AdsMolData
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*') 

class MolDataReader:
    def __init__(self,
                 data_fpath: str,
                 vocab_fpath: Optional[str],
                 normalize: bool = True):
        
        # initialize the variables
        self.data_fpath = data_fpath
        self.vocab_fpath = vocab_fpath
        self.normalize = normalize
        self.vocab = None
        self.grammar = None

        # load Ads.mol. property
        self.dataset = self._load_prop()
        # load vocab for building group grammar(for GroupSELFIES)
        if self.vocab_fpath is not None:
            self.vocab = self._load_vocab()
        # build the group grammar(for GroupSELFIES)
        if self.vocab is not None:
            self._build_grammar()

    def _load_prop(self) -> pd.DataFrame:
        
        if not os.path.exists(self.data_fpath):
            raise FileNotFoundError(f"File {self.data_fpath} does not exist.")
        
        dataset = pd.read_csv(self.data_fpath)
        return dataset
    
    def _load_vocab(self) -> pd.DataFrame:

        if not os.path.exists(self.vocab_fpath):
            raise FileNotFoundError(f"File {self.vocab_fpath} does not exist.")
    
        vocab = pd.read_csv(self.vocab_fpath, encoding='cp949')

        return vocab

    def _build_grammar(self):
        
        self.Groups = []
        for i, row in self.vocab.iterrows():
            g = Group(row['Fragment'], row['group_SMILES'], all_attachment=bool(row['all_attachment']))
            self.Groups.append(g)

        self.grammar = GroupGrammar(self.Groups)
    
    
    def read(self,
             properties: Optional[List[str]] = None,
             show_progress: bool = True,
             save_result: Optional[bool] = False) -> List[AdsMolData]:       

        if properties == None:
            properties = ['Ads.E']

        # validate the CSV columns name
        for prop in properties:
            if prop not in self.dataset.columns:
                raise KeyError(f"Property '{prop}' not found in dataset columns: {list(self.dataset.columns)}")

        if 'SMILES' not in self.dataset.columns:
            raise KeyError("Dataset must contain a 'SMILES' column.")
        
        # set the tqdm iterator
        iterator = tqdm(self.dataset.iterrows(),
                        total=len(self.dataset),
                        desc="Read Molecules",
                        disable=not show_progress)
        
        AdsMolDataset: List[AdsMolData] = []

        for i, row in iterator:
            # load the info.
            smiles = row['SMILES']
            prop_dict = {prop: row[prop] for prop in properties}
            
            # build the AdsMolData
            ads_mol = AdsMolData(SMILES=smiles, name=f'mol{i+1}', properties=prop_dict)
            self.dataset.loc[i, 'SELFIES'] = ads_mol.SELFIES
            if self.grammar is not None:
                m = ads_mol.rdkit_mol
                extracted = self.grammar.extract_groups(m)
                encoded = self.grammar.encoder(m, extracted)
                ads_mol.GroupSELFIES = encoded
                self.dataset.loc[i, 'GroupSELFIES'] = encoded
            
            AdsMolDataset.append(ads_mol)
        
        if save_result:
            self.dataset.to_csv('./dataset.csv', index=False)
        return AdsMolDataset
    
def encoding(dataset: Optional[List[AdsMolData]] = None,
             dataframe: Optional[pd.DataFrame] = None,
             represent: Literal['gsf', 'selfies', 'smiles'] = 'gsf',
) -> List[str]:
    # mapping attribute/columns
    rep2attr = {
        'gsf':     ('GroupSELFIES', 'GroupSELFIES'),
        'selfies': ('SELFIES',      'SELFIES'),
        'smiles':  ('SMILES',       'SMILES'),
    }
    if represent not in rep2attr:
        raise ValueError(f"represent must be one of {list(rep2attr.keys())}, got: {represent}")
    
    ds_attr, df_col = rep2attr[represent]

    # only Dataset
    if dataset is not None and dataframe is None:
        missing = [i for i,m in enumerate(dataset) if not hasattr(m, ds_attr)]
        if missing:
            raise AttributeError(f"Some AdsMolData items are missing attribute '{ds_attr}': indices {missing[:5]}...")
        repr_set = [getattr(ads_mol, ds_attr) for ads_mol in dataset]
    # only DataFrame
    elif dataset is None and dataframe is not None:
        if df_col not in dataframe.columns:
            raise KeyError(f"DataFrame must contain column '{df_col}'. Have: {list(dataframe.columns)}")
        repr_set = dataframe[df_col].astype(str).tolist()

    # both Dataset/DataFrame
    elif dataset is not None and dataframe is not None:
        missing = [i for i,m in enumerate(dataset) if not hasattr(m, ds_attr)]
        if missing:
            raise AttributeError(f"Some AdsMolData items are missing attribute '{ds_attr}': indices {missing[:5]}...")
        ds_list = [getattr(ads_mol, ds_attr) for ads_mol in dataset]

        if df_col not in dataframe.columns:
            raise KeyError(f"DataFrame must contain column '{df_col}'. Have: {list(dataframe.columns)}")
        df_list = dataframe[df_col].astype(str).tolist()
        
        if len(ds_list) != len(df_list):
            raise ValueError(f"Size mismatch between dataset({len(ds_list)}) and dataframe({len(df_list)}).")

        if ds_list != df_list:
            raise ValueError("Content mismatch between dataset and dataframe.")

        repr_set = ds_list
    # neither Dataset/DataFrame
    else:
        raise ValueError("Either 'dataset' or 'dataframe' must be provided.")

    return repr_set
            
    

if __name__=="__main__":


    data_root_dir = 'D:\\genDOPA\\data\\'
    data_fpath = os.path.join(data_root_dir, "adsorption_property.csv")
    vocab_fpath = os.path.join(data_root_dir, "vocab.csv")
    reader = MolDataReader(data_fpath, vocab_fpath=vocab_fpath)

    print(reader.dataset.head())

    AdsMolDataset = reader.read(save_result=False)
    
    print(reader.dataset.head())
    repr_set = encoding(dataset=AdsMolDataset, represent='gsf')
    gsf_set = [ads_mol.GroupSELFIES for ads_mol in AdsMolDataset]
    alphabet = sf.get_alphabet_from_selfies(repr_set)
    alphabet.add("[nop]")
    alphabet = list(sorted(alphabet))
    print(alphabet)
        


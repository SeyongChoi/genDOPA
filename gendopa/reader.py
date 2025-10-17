import os
from tqdm import tqdm
import pandas as pd
from typing import List, Optional

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
             show_progress: bool = True) -> List[AdsMolData]:       

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
        
        
        return AdsMolDataset

if __name__=="__main__":

    data_root_dir = 'D:\\genDOPA\\data\\'
    data_fpath = os.path.join(data_root_dir, "adsorption_property.csv")
    vocab_fpath = os.path.join(data_root_dir, "vocab.csv")
    reader = MolDataReader(data_fpath, vocab_fpath=vocab_fpath)

    print(reader.dataset.head())

    AdsMolDataset = reader.read()
    
    print(reader.dataset.head())
        


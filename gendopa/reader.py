import os
from tqdm import tqdm
import pandas as pd
from typing import List, Optional

from gendopa.molinfo import AdsMolData

class MolDataReader:
    def __init__(self,
                 data_fpath: str,
                 normalize: bool = True):
        
        self.data_fpath = data_fpath
        self.normalize = normalize
        self.dataset = self._load_csv()

    def _load_csv(self) -> pd.DataFrame:

        
        if not os.path.exists(self.data_fpath):
            raise FileNotFoundError(f"File {self.data_fpath} does not exist.")
        
        dataset = pd.read_csv(self.data_fpath)
        return dataset
    
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
            smiles = row['SMILES']
            prop_dict = {prop: row[prop] for prop in properties}
            ads_mol = AdsMolData(SMILES=smiles, name=f'mol{i+1}', properties=prop_dict)
            AdsMolDataset.append(ads_mol)
        
        
        return AdsMolDataset

if __name__=="__main__":

    data_root_dir = 'D:\\genDOPA\\data\\'
    data_fpath = os.path.join(data_root_dir, "adsorption_property.csv")

    reader = MolDataReader(data_fpath)

    print(reader.dataset.head())

    AdsMolDataset = reader.read()
    
    print(AdsMolDataset[:4])
        


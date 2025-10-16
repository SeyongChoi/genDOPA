import os
import pandas as pd
from typing import List

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
    
    def read(self) -> List[AdsMolData]:       

        AdsMolDataset = list()
        for i, smiles in enumerate(self.dataset['SMILES']):
            ads_mol = AdsMolData(SMILES=smiles, name=f'mol{i+1}')
            AdsMolDataset.append(ads_mol)
        
        return AdsMolDataset

if __name__=="__main__":

    data_root_dir = 'D:\\genDOPA\\data\\'
    data_fpath = os.path.join(data_root_dir, "adsorption_property.csv")

    reader = MolDataReader(data_fpath)

    print(reader.dataset.head())

    AdsMolDataset = reader.read()
    
    print(AdsMolDataset[:4])
        


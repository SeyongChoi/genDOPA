"""
This module provides classes and functions to handle molecular data.
"""

from rdkit import Chem
from rdkit.Chem import Draw

import selfies as sf
from group_selfies import Group, GroupGrammar
from dataclasses import dataclass, field
from typing import Optional, Dict, List

@dataclass
class AdsMolData:
    pdb_path: Optional[str] = None
    name: Optional[str] = None
    SMILES: Optional[str] = None
    SELFIES: Optional[str] = None
    GroupSELFIES: Optional[str] = None
    properties: Dict[str, float] = field(default_factory=dict)
    rdkit_mol: Optional[Chem.Mol] = None
    gsf_groups: Optional[List[Group]] = None

    def __post_init__(self):
        # Load RDKit Mol from PDB
        if self.pdb_path and self.rdkit_mol is None:
            self.rdkit_mol = Chem.MolFromPDBFile(self.pdb_path, sanitize=False, removeHs=False)
            if self.rdkit_mol is None:
                raise ValueError(f"Could not read PDB file at {self.pdb_path}")
            try:
                Chem.SanitizeMol(self.rdkit_mol)
            except Exception:
                print(f"[Warning] Molecule sanitization failed for {self.name}")

        # Load RDKit Mol from SMILES
        if self.SMILES and self.rdkit_mol is None:
            self.rdkit_mol = Chem.MolFromSmiles(self.SMILES, sanitize=False)
            if self.rdkit_mol is None:
                raise ValueError(f"[SMILES] Could not parse: {self.SMILES}")
            try:
                Chem.SanitizeMol(self.rdkit_mol)
            except Exception:
                print(f"[Warning] Molecule sanitization failed for {self.name} (SMILES)")

        # Generate SMILES
        if self.SMILES is None and self.rdkit_mol is not None:
            self.SMILES = Chem.MolToSmiles(self.rdkit_mol)

        # Generate SELFIES
        if self.SMILES and self.SELFIES is None:
            try:
                self.SELFIES = sf.encoder(self.SMILES)
            except Exception as e:
                print(f"[Warning] SELFIES encoding failed for {self.name}: {e}")

        # Generate GroupSELFIES
        if self.gsf_groups:
            if self.GroupSELFIES is None:
                grammar = GroupGrammar(self.gsf_groups)
                self.GroupSELFIES = grammar.full_encode(self.rdkit_mol)
        

    def __repr__(self) -> str:
        return (f"AdsMolData(name={self.name}, SMILES={self.SMILES}, "
                f"SELFIES={self.SELFIES}, "
                f"GroupSELFIES={self.GroupSELFIES}, "
                f"properties={self.properties})")
    
    def vis(self,
            save_path: Optional[str] = None):
        if self.rdkit_mol:
            img = Draw.MolToImage(self.rdkit_mol, size=(300, 300))
            if save_path:
                img.save(save_path)
            return Draw.MolToImage(self.rdkit_mol, size=(300, 300))
        else:
            raise ValueError("RDKit molecule is not available for visualization.")
        
if __name__ == "__main__":
    import os
    
    data_dir = "D:\\genDOPA\\data\\"
    pdb_path = os.path.join(data_dir,"mol1.pdb")
    AdsMolData_test = AdsMolData(SMILES='NC(CC1=C(CO)C(O)=C(O)C=C1)C(O)=O', name="mol2")
    print(AdsMolData_test)
    print(AdsMolData_test.vis(save_path=os.path.join(data_dir,"mol6.png")))
    print(sf.decoder(AdsMolData_test.SELFIES))
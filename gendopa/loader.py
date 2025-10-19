import numpy as np
from typing import List, Optional, Literal, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split

from gendopa.molinfo import AdsMolData
from gendopa.dataset import AdsMolDataset

class Collator:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        x = torch.stack(batch, dim=0)       # [B, T]
        mask = (x != self.pad_idx).long()   # [B, T]
        return x, mask

def split_n_load(
    dataset: List[AdsMolData],
    data_split: List[float] = [0.7, 0.1, 0.2],
    batch_size: int = 32,
    represent: Literal['gsf', 'selfies', 'smiles'] = 'gsf',
    enc_type: Literal['label', 'one_hot'] = 'label',
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # 1) 파라미터 검증
    if not np.isclose(sum(data_split), 1.0):
        raise ValueError(f"data_split must sum to 1.0, got {data_split} (sum={sum(data_split):.4f})")
    if len(dataset) < 3:
        raise ValueError("Need at least 3 samples to split into train/val/test.")

    # 2) 원본 AdsMolData → 인덱싱 가능한 PyTorch Dataset
    mol_dataset = AdsMolDataset(
        dataset=dataset,
        represent=represent,
        enc_type=enc_type
    )

    # 3) 길이 계산(올림/내림 섞여 합이 정확히 N이 되도록 마지막에 보정)
    N = len(mol_dataset)
    n_train = int(N * data_split[0])
    n_val   = int(N * data_split[1])
    n_test  = N - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        # 각 split이 최소 1개는 갖도록 간단 보정
        n_train = max(1, n_train)
        n_val   = max(1, n_val)
        n_test  = max(1, N - n_train - n_val)
        if n_train + n_val + n_test != N:
            n_test = N - n_train - n_val

    # 4) 재현 가능한 split (torch random_split)
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(mol_dataset, [n_train, n_val, n_test], generator=g)

    # 5) Collator (PAD 인덱스)
    pad_idx = mol_dataset.vocab_stoi['[nop]']
    collate_fn = Collator(pad_idx=pad_idx)

    # 6) DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader

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
    train_loader, val_loader, test_loader = split_n_load(
        dataset=MolDataset,
        batch_size=1)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    for batch in train_loader:
        x, mask = batch
        print(f"x: {x.shape}, mask: {mask.shape}")
        break
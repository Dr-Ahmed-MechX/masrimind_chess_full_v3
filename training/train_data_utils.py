from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Tuple, List, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from training.datasets import DistillDataset, RLDataset, default_collate
from training.encode import fen_to_tensor  # موجود عندك

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_jsonl(input_path: str, out_train: str, out_val: str, val_ratio: float = 0.1, seed: int = 42):
    set_seed(seed)
    inp = Path(input_path)
    assert inp.exists(), f"Not found: {inp}"
    with inp.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    random.shuffle(lines)
    n = len(lines)
    n_val = max(1, int(n * val_ratio))
    val = lines[:n_val]
    train = lines[n_val:]
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(out_val).parent.mkdir(parents=True, exist_ok=True)
    Path(out_train).write_text("".join(train), encoding="utf-8")
    Path(out_val).write_text("".join(val), encoding="utf-8")
    print(f"Split done → train={len(train)}, val={len(val)}")

def build_distill_loaders(train_jsonl: str, val_jsonl: str,
                          batch_size: int = 256,
                          num_workers: int = 0,
                          device: str = "cpu") -> Tuple[DataLoader, DataLoader]:
    ds_train = DistillDataset(train_jsonl, encode_fn=fen_to_tensor, device=device)
    ds_val = DistillDataset(val_jsonl, encode_fn=fen_to_tensor, device=device)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=(device=="cuda"),
                          collate_fn=default_collate, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device=="cuda"),
                        collate_fn=default_collate, drop_last=False)
    return dl_train, dl_val

def build_rl_loaders(train_jsonl: str, val_jsonl: str,
                     batch_size: int = 256,
                     num_workers: int = 0,
                     device: str = "cpu") -> Tuple[DataLoader, DataLoader]:
    ds_train = RLDataset(train_jsonl, encode_fn=fen_to_tensor, device=device)
    ds_val = RLDataset(val_jsonl, encode_fn=fen_to_tensor, device=device)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=(device=="cuda"),
                          collate_fn=default_collate, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device=="cuda"),
                        collate_fn=default_collate, drop_last=False)
    return dl_train, dl_val

def cp_to_value_tanh(cp: float, scale: float = 600.0) -> float:
    """
    Convert Stockfish centipawn eval (side-to-move) to [-1,1] via tanh.
    Mate scores should be handled upstream (e.g., clamp to +/- inf then sign).
    """
    import math
    return math.tanh(cp / scale)

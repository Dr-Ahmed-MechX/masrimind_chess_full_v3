import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
import chess

# نعتمد نفس ترميز السياسة (policy) المستخدم في رأس الشبكة: index = from*64 + to
def uci_to_index(uci_move: str) -> int:
    """
    Map a UCI move like 'e2e4' (or with promotion e.g. 'e7e8q') to [0..4095] via from*64+to.
    Promotions are ignored at the head level (you can encode promo as part of target distribution
    by mapping all promos of same from-to to the same index).
    """
    move = chess.Move.from_uci(uci_move)
    return move.from_square * 64 + move.to_square

def index_to_uci(idx: int, board: chess.Board) -> str:
    frm = idx // 64
    to = idx % 64
    # Promotion ambiguity: default to 'q' if that move requires promotion
    move = chess.Move(frm, to)
    if move not in board.legal_moves:
        # try promotions if needed
        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            pm = chess.Move(frm, to, promotion=promo)
            if pm in board.legal_moves:
                move = pm
                break
    return move.uci()

def legal_mask_4096(board: chess.Board) -> torch.Tensor:
    """Mask (4096,) with 1.0 for legal from-to indices, else 0.0."""
    mask = torch.zeros(4096, dtype=torch.float32)
    for mv in board.legal_moves:
        idx = mv.from_square * 64 + mv.to_square
        mask[idx] = 1.0
    return mask

def normalize_policy(vec: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vector over (masked) entries to sum=1 (if there is any mass)."""
    if mask is not None:
        vec = vec * mask
    s = vec.sum()
    if s <= eps:
        # fallback: uniform over legal if provided
        if mask is not None and mask.sum() > 0:
            return mask / mask.sum()
        return vec  # zeros
    return vec / s

class JsonlDatasetBase(Dataset):
    """
    Base class to read JSONL files where each line is a dict.

    - We require 'fen' (string).
    - Policy targets can be one of:
        * 'policy_4096': list[float] length 4096
        * 'policy_dict': dict[str move_uci -> float prob]
    - Value targets can be one of:
        * 'value': float in [-1, 1]
        * 'result': float in {-1, 0, 1}   (win/loss/draw from side-to-move perspective)
    """
    def __init__(self, jsonl_path: str, encode_fn=None, device: str = "cpu"):
        self.path = Path(jsonl_path)
        assert self.path.exists(), f"Dataset file not found: {self.path}"
        self.encode_fn = encode_fn  # training.encode.fen_to_tensor
        self.device = device
        self._index_to_offset: List[int] = []
        # Build index of byte offsets for fast random access
        with self.path.open("rb") as f:
            off = 0
            while True:
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._index_to_offset.append(off)
                off = f.tell()

    def __len__(self) -> int:
        return len(self._index_to_offset)

    def _read_line(self, idx: int) -> Dict[str, Any]:
        off = self._index_to_offset[idx]
        with self.path.open("rb") as f:
            f.seek(off)
            line = f.readline().decode("utf-8").strip()
        return json.loads(line)

    @staticmethod
    def _policy_from_entry(entry: Dict[str, Any], board: chess.Board) -> torch.Tensor:
        # Prefer full vector if available
        if "policy_4096" in entry:
            arr = entry["policy_4096"]
            assert isinstance(arr, list) and len(arr) == 4096, "policy_4096 must be length 4096"
            t = torch.tensor(arr, dtype=torch.float32)
            # mask illegal just in case then renorm
            mask = legal_mask_4096(board)
            return normalize_policy(t, mask=mask)
        # Else accept sparse dict
        if "policy_dict" in entry:
            dct = entry["policy_dict"]
            vec = torch.zeros(4096, dtype=torch.float32)
            for uci, p in dct.items():
                try:
                    idx = uci_to_index(uci)
                    vec[idx] += float(p)
                except Exception:
                    # skip malformed move
                    pass
            mask = legal_mask_4096(board)
            return normalize_policy(vec, mask=mask)
        # If none given, fallback to uniform over legal
        mask = legal_mask_4096(board)
        return normalize_policy(mask.clone(), mask=mask)

    @staticmethod
    def _value_from_entry(entry: Dict[str, Any]) -> float:
        if "value" in entry:
            return float(entry["value"])
        if "result" in entry:
            # result is from side-to-move perspective
            return float(entry["result"])
        # default zero (neutral)
        return 0.0

class DistillDataset(JsonlDatasetBase):
    """
    Supervised distillation dataset:
     - 'fen' : str
     - 'policy_4096' or 'policy_dict' (optional but recommended)
     - 'value' in [-1,1]  (e.g., Stockfish eval scaled)
    """
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self._read_line(idx)
        fen = entry["fen"]
        board = chess.Board(fen)
        # X
        x = self.encode_fn(fen) if self.encode_fn else None  # torch.FloatTensor [C,H,W]
        # Targets
        pi = self._policy_from_entry(entry, board)  # (4096,)
        v = torch.tensor([self._value_from_entry(entry)], dtype=torch.float32)
        return {"x": x, "pi": pi, "v": v, "fen": fen}

class RLDataset(JsonlDatasetBase):
    """
    RL/self-play dataset:
     - 'fen' : str
     - 'policy_4096' or 'policy_dict'  (MCTS visit distribution)
     - 'result' in {-1,0,1}
    """
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self._read_line(idx)
        fen = entry["fen"]
        board = chess.Board(fen)
        x = self.encode_fn(fen) if self.encode_fn else None
        pi = self._policy_from_entry(entry, board)
        z = torch.tensor([self._value_from_entry(entry)], dtype=torch.float32)  # 'result'
        return {"x": x, "pi": pi, "z": z, "fen": fen}

def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Stack X if present
    xs = [b["x"] for b in batch if b["x"] is not None]
    X = torch.stack(xs, dim=0) if len(xs) == len(batch) else None
    PIs = torch.stack([b["pi"] for b in batch], dim=0)
    # v or z depending on dataset
    if "v" in batch[0]:
        tgt = torch.cat([b["v"] for b in batch], dim=0)  # (B,1)
        return {"x": X, "pi": PIs, "v": tgt, "fen": [b["fen"] for b in batch]}
    else:
        tgt = torch.cat([b["z"] for b in batch], dim=0)  # (B,1)
        return {"x": X, "pi": PIs, "z": tgt, "fen": [b["fen"] for b in batch]}

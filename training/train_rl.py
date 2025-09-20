from __future__ import annotations
import os, json, math, time, random, argparse, glob, subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from training.move_index import uci_to_index, POLICY_SIZE

# =========================
# Utilities
# =========================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def log(msg: str):
    print(f"[train_rl] {msg}", flush=True)

def save_checkpoint(state: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path.as_posix())

# =========================
# Model (fallback same as distill)
# =========================
class SimplePVNet(nn.Module):
    POLICY_SIZE = 4672
    def __init__(self, in_channels: int = 13, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(True),
        )
        self.head_policy = nn.Sequential(
            nn.Conv2d(hidden, 32, 1), nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32*8*8, self.POLICY_SIZE),
        )
        self.head_value = nn.Sequential(
            nn.Conv2d(hidden, 16, 1), nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(16*8*8, 64), nn.ReLU(True),
            nn.Linear(64, 1), nn.Tanh(),
        )
    def forward(self, x):
        z = self.enc(x)
        return self.head_policy(z), self.head_value(z)

def load_user_model_or_fallback(model_name: Optional[str] = None, in_channels: int = 13) -> nn.Module:
    if model_name and model_name.lower() != "auto":
        pkg, cls = model_name.rsplit(":", 1) if ":" in model_name else (model_name, "PVNetPolicyValue")
        try:
            mod = __import__(pkg, fromlist=[cls])
            Model = getattr(mod, cls)
            return Model()
        except Exception as e:
            log(f"⚠️ Failed to import custom model '{model_name}': {e}. Falling back.")
    for pkg, cls in [("models.pvnet","PVNetPolicyValue"), ("training.models.pvnet","PVNetPolicyValue"), ("models","PVNetPolicyValue")]:
        try:
            mod = __import__(pkg, fromlist=[cls]); Model = getattr(mod, cls)
            log(f"Loaded user model: {pkg}.{cls}"); return Model()
        except Exception: pass
    log("Using SimplePVNet fallback.")
    return SimplePVNet(in_channels=in_channels)

# =========================
# Chess helpers
# =========================
def fen_to_planes(fen: str) -> np.ndarray:
    import chess
    board = chess.Board(fen)
    planes = np.zeros((13, 8, 8), dtype=np.float32)
    piece_map = {chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2, chess.ROOK:3, chess.QUEEN:4, chess.KING:5}
    for sq, piece in board.piece_map().items():
        r = 7 - (sq // 8); c = sq % 8
        base = 0 if piece.color == chess.WHITE else 6
        planes[base + piece_map[piece.piece_type], r, c] = 1.0
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return planes

def policy_ce_from_vector(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    target = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-8)
    logp = F.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()

# Optional: KL to Stockfish over legal moves
def stockfish_eval_moves(fen: str, stockfish_path: str, movetime_ms: int = 100) -> Dict[str, float]:
    """
    بترجع dict uci->score(cp). بنستخدم softmax على cp/temperature لبناء توزيع مرجعي.
    تنفيذ UCI مبسط عبر subprocess (لكل موضع) — كافي للتدريب الخفيف.
    """
    import subprocess, tempfile, sys
    try:
        proc = subprocess.Popen([stockfish_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        raise RuntimeError(f"Failed to launch Stockfish: {e}")

    def send(cmd):
        assert proc.stdin is not None
        proc.stdin.write(cmd + "\n"); proc.stdin.flush()
    def read_until(token: str, timeout=2.0):
        proc.stdout.flush()
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line.strip())
            if token in line:
                break
        return lines

    send("uci"); read_until("uciok")
    send("isready"); read_until("readyok")
    send(f"position fen {fen}")
    send(f"go movetime {movetime_ms}")
    lines = read_until("bestmove")

    # parse 'info ... score cp X ... pv <move> ...'
    scores: Dict[str, float] = {}
    current_move = None
    for ln in lines:
        if " score " in ln and " pv " in ln:
            # try to parse cp
            parts = ln.split()
            if "cp" in parts:
                try:
                    cp_idx = parts.index("cp") + 1
                    cp = float(parts[cp_idx])
                except Exception:
                    continue
                # pv move is last token usually after 'pv'
                if "pv" in parts:
                    pv_idx = parts.index("pv") + 1
                    if pv_idx < len(parts):
                        mv = parts[pv_idx]
                        scores[mv] = cp
    try:
        send("quit")
    except Exception:
        pass
    return scores

def kl_divergence(p_logits: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
    """
    KL(P || Q) حيث P=softmax(p_logits), Q = q_probs (normalized).
    """
    p = F.log_softmax(p_logits, dim=1)
    q = (q_probs / (q_probs.sum(dim=1, keepdim=True) + 1e-8)).clamp_min(1e-8)
    return (torch.exp(p) * (p - torch.log(q))).sum(dim=1).mean()

# =========================
# Dataset
# =========================
class RLDataset(Dataset):
    """
    يتوقع ملفات JSONL داخل مجلد الـ selfplay تحتوي سجلات مثل:
      { "fen": "...", "policy": [4672-vector OR None], "result": -1/0/1 }
    أو زيارة MCTS كـ "policy_dist": [...4672...]
    لو مفيش policy: ندرّب قيمة فقط، ومع KL لستوكفيش (لو متاح).
    """
    def __init__(self, data_dir: Path):
        self.items: List[Dict[str, Any]] = []
        files = sorted(glob.glob(str(data_dir / "*.jsonl")))
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    j = json.loads(line)
                    if "fen" not in j:  # fallback: لو عندك planes مباشرة
                        if "planes" in j:
                            j["planes"] = np.array(j["planes"], dtype=np.float32)
                        else:
                            continue
                    self.items.append(j)
        if not self.items:
            # fallback: ملف episodes.jsonl
            epi = data_dir / "episodes.jsonl"
            if epi.exists():
                with open(epi, "r", encoding="utf-8") as f:
                    for line in f:
                        j = json.loads(line)
                        # لو مش موجود fen، هنعدّي بقيم فقط
                        self.items.append(j)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        if "planes" in it:
            planes = np.array(it["planes"], dtype=np.float32)
        else:
            fen = it.get("fen", None)
            if fen is None:
                # dummy empty board planes (لن يُستفاد منها إلا للقيمة)
                planes = np.zeros((13,8,8), dtype=np.float32)
            else:
                planes = fen_to_planes(fen)
        value = float(it.get("result", it.get("value", 0.0)))
        pol = it.get("policy", it.get("policy_dist", None))
        pol = None if pol is None else np.array(pol, dtype=np.float32)
        fen = it.get("fen", None)
        return planes, np.array([value], dtype=np.float32), pol, fen

# =========================
# Train
# =========================
def train_rl(
    data_dir: Path,
    init_net_path: Path,
    out_model_path: Path,
    epochs: int = 3,
    batch_size: int = 1024,
    lr: float = 5e-4,
    kl_to_stockfish: bool = False,
    kl_weight: float = 0.1,
    stockfish_path: Optional[str] = None,
    model_name: Optional[str] = "auto",
    grad_clip: float = 1.0,
    num_workers: int = 0,
    seed: int = 42,
):
    set_seed(seed)
    dev = device_auto()
    ds = RLDataset(data_dir)
    if len(ds) == 0:
        raise RuntimeError("RL dataset is empty. Check self-play output.")

    # infer channels
    planes0, _, _, _ = ds[0]
    in_ch = planes0.shape[0]

    model = load_user_model_or_fallback(model_name, in_channels=in_ch).to(dev)

    # لو في موديل ابتدائي - نحاول نحمّله (state_dict)
    if Path(init_net_path).exists():
        try:
            state = torch.load(init_net_path.as_posix(), map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
            log(f"Loaded init weights from {init_net_path}")
        except Exception as e:
            log(f"⚠️ Failed to load init net: {e}")

    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type=="cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    v_loss_fn = nn.MSELoss()  # RL value - ممكن Huber برضه
    p_loss_weight = 1.0
    v_loss_weight = 1.0

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(dev.type=="cuda"))

    best = float("inf")
    out_model_path = Path(out_model_path)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out_model_path.with_suffix(".ckpt.pt")

    for ep in range(1, epochs+1):
        model.train()
        running = 0.0; nb = 0
        for planes, value, policy, fens in dl:
            planes = torch.tensor(planes, dtype=torch.float32, device=dev)
            value  = torch.tensor(value, dtype=torch.float32, device=dev)
            policy = None if policy is None else torch.tensor(policy, dtype=torch.float32, device=dev)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev.type=="cuda")):
                plogits, vpred = model(planes)
                v_l = v_loss_fn(vpred, value)

                p_l = torch.tensor(0.0, device=dev)
                # supervised policy (من زيارات MCTS) لو متاحة
                if policy is not None:
                    p_l = policy_ce_from_vector(plogits, policy)

                # KL إلى ستوكفيش (لو مُفعّلة وعندنا مسار المحرك وفينات)
                if kl_to_stockfish and stockfish_path and isinstance(fens, list):
                    # نبني توزيع هدف من ستوكفيش لكل FEN
                    sf_targets: List[np.ndarray] = []
                    for fen in fens:
                        if fen is None:
                            sf_targets.append(np.zeros((SimplePVNet.POLICY_SIZE,), dtype=np.float32))
                            continue
                        try:
                            scores = stockfish_eval_moves(fen, stockfish_path)
                        except Exception:
                            scores = {}
                        # softmax على cp/200.0 لتوزيع معقول
                        if not scores:
                            sf_targets.append(np.zeros((SimplePVNet.POLICY_SIZE,), dtype=np.float32))
                            continue
                        moves = list(scores.keys()); cps = np.array([scores[m] for m in moves], dtype=np.float32)
                        cps = cps / 200.0
                        exp = np.exp(cps - cps.max())
                        probs = exp / (exp.sum() + 1e-8)
                        # NOTE: هنا بنسقطها على رأس 4672: من دون ترميز محدد للمؤشرات، هنحطها في أول len(moves) مواقع فقط
                        # لو عندك mapping ثابت للـ4672، بدّل هذا الجزء.
                        target = np.zeros((SimplePVNet.POLICY_SIZE,), dtype=np.float32)
                        target = np.zeros((POLICY_SIZE,), dtype=np.float32)
                        for mv, pr in zip(moves, probs):
                            idx = uci_to_index(mv, fen=fen)  # مرر الـ FEN عشان نعرف الأبيض/الأسود
                            if 0 <= idx < POLICY_SIZE:
                                target[idx] = pr

                        sf_targets.append(target)
                    if sf_targets:
                        q = torch.tensor(np.stack(sf_targets), dtype=torch.float32, device=dev)
                        kl = kl_divergence(plogits, q)
                        p_l = p_l + kl_weight * kl

                loss = v_loss_weight * v_l + p_loss_weight * p_l

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()

            running += loss.item(); nb += 1

        sched.step()
        ep_loss = running / max(1, nb)
        log(f"Epoch {ep}/{epochs} | train_loss={ep_loss:.4f} | lr={sched.get_last_lr()[0]:.6f}")

        if ep_loss < best:
            best = ep_loss
            save_checkpoint({"model": model.state_dict(), "epoch": ep, "loss": ep_loss}, ckpt)
            torch.save(model.state_dict(), out_model_path.as_posix())
            log(f"✅ Saved best RL model to {out_model_path}")

    log("Done.")

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Self-play dir containing *.jsonl / episodes.jsonl")
    ap.add_argument("--net", required=True, help="Init net path (.pt/.ckpt)")
    ap.add_argument("--out", required=True, help="Output model path (.pt)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--kl-to-stockfish", type=str, default="false", help="true/false")
    ap.add_argument("--kl-weight", type=float, default=0.1)
    ap.add_argument("--stockfish", type=str, default=os.environ.get("STOCKFISH_PATH", ""))
    ap.add_argument("--model", type=str, default="auto", help="Import path 'pkg:Class' or 'auto'")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_rl(
        data_dir=Path(args.data),
        init_net_path=Path(args.net),
        out_model_path=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kl_to_stockfish=(str(args.kl_to_stockfish).lower() in ["1","true","yes","y"]),
        kl_weight=args.kl_weight,
        stockfish_path=args.stockfish if args.stockfish else None,
        model_name=args.model,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        seed=args.seed,
    )

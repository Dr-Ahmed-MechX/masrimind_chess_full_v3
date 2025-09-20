from __future__ import annotations
import os, json, math, time, random, argparse
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def log(msg: str):
    print(f"[train_distill_real] {msg}", flush=True)

def save_checkpoint(state: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path.as_posix())

# =========================
# Model
# =========================
class SimplePVNet(nn.Module):
    """
    Fallback CNN إذا موديلك الحقيقي مش متاح للاستيراد.
    Produces:
      - policy_logits: (N, 4672)  -> 64*64 moves + ترقيات (N/B/R) = 4672
      - value: (N, 1) in [-1,1]
    """
    POLICY_SIZE = POLICY_SIZE  # استخدم نفس الثابت

    def __init__(self, in_channels: int = 13, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head_policy = nn.Sequential(
            nn.Conv2d(hidden, 32, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, self.POLICY_SIZE),
        )
        self.head_value = nn.Sequential(
            nn.Conv2d(hidden, 16, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),  # output in [-1,1]
        )

    def forward(self, x):
        z = self.enc(x)
        p = self.head_policy(z)
        v = self.head_value(z)
        return p, v

def load_user_model_or_fallback(model_name: Optional[str] = None, in_channels: int = 13) -> nn.Module:
    """
    يحاول يستورد موديلك PVNetPolicyValue لو عندك، وإلا يرجع SimplePVNet.
    - جرّب مسارات شائعة: models.pvnet، training.models، models
    """
    if model_name and model_name.lower() != "auto":
        # محاولة تحميل كلاس كامل من مسار معيّن
        pkg, cls = model_name.rsplit(":", 1) if ":" in model_name else (model_name, "PVNetPolicyValue")
        try:
            mod = __import__(pkg, fromlist=[cls])
            Model = getattr(mod, cls)
            return Model()
        except Exception as e:
            log(f"⚠️ Failed to import custom model '{model_name}': {e}. Falling back.")

    candidates: List[Tuple[str,str]] = [
        ("models.pvnet", "PVNetPolicyValue"),
        ("training.models.pvnet", "PVNetPolicyValue"),
        ("models", "PVNetPolicyValue"),
    ]
    for pkg, cls in candidates:
        try:
            mod = __import__(pkg, fromlist=[cls])
            Model = getattr(mod, cls)
            log(f"Loaded user model: {pkg}.{cls}")
            return Model()
        except Exception:
            pass

    log("Using SimplePVNet fallback.")
    return SimplePVNet(in_channels=in_channels)

# =========================
# Data
# =========================
def fen_to_planes(fen: str) -> np.ndarray:
    """
    Encoder بسيط: 12 قنوات للقطع + 1 قناة للدور (أبيض/أسود) = 13x8x8
    """
    import chess
    board = chess.Board(fen)
    planes = np.zeros((13, 8, 8), dtype=np.float32)
    # piece planes
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for sq, piece in board.piece_map().items():
        r = 7 - (sq // 8)
        c = sq % 8
        base = 0 if piece.color == chess.WHITE else 6
        planes[base + piece_map[piece.piece_type], r, c] = 1.0
    # side to move
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return planes

class DistillDataset(Dataset):
    """
    يدعم:
      - NPZ: يجب أن يحتوي على واحدة أو أكثر مما يلي:
         planes: (N, C, 8, 8)
         value:  (N,) or (N,1) in [-1,1]   (أو cp ثم نطبّق tanh_scale)
         policy: (N, 4672) توزيع اختياري (إن وُجد)
      - JSONL: كل سطر dict ممكن يحتوي: fen, value, cp, policy_topk
         policy_topk: [{"uci": "e2e4", "p": 0.35}, ...]
    """
    def __init__(self, path: Path, tanh_scale: Optional[float] = None):
        self.items: List[Dict[str, Any]] = []
        self.tanh_scale = tanh_scale
        p = Path(path)
        if p.suffix.lower() == ".npz":
            data = np.load(p, allow_pickle=True)
            planes = data.get("planes")
            value = data.get("value", None)
            cp = data.get("cp", None)
            policy = data.get("policy", None)

            n = planes.shape[0] if planes is not None else (value.shape[0] if value is not None else policy.shape[0])
            for i in range(n):
                it: Dict[str, Any] = {}
                if planes is not None:
                    it["planes"] = planes[i]
                if value is not None:
                    v = value[i].item() if hasattr(value[i], "item") else float(value[i])
                    it["value"] = float(v)
                elif cp is not None:
                    # سي بي إلى قيمة -1..1 باستخدام tanh
                    cpi = cp[i].item() if hasattr(cp[i], "item") else float(cp[i])
                    it["value"] = math.tanh(cpi / (self.tanh_scale or 400.0))
                if policy is not None:
                    it["policy"] = policy[i]
                self.items.append(it)
        else:
            # JSONL
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    it: Dict[str, Any] = {}
                    if "planes" in j:
                        it["planes"] = np.array(j["planes"], dtype=np.float32)
                    elif "fen" in j:
                        it["planes"] = fen_to_planes(j["fen"])
                    if "value" in j:
                        it["value"] = float(j["value"])
                    elif "cp" in j:
                        it["value"] = math.tanh(float(j["cp"]) / (self.tanh_scale or 400.0))
                    if "policy" in j:
                        it["policy"] = np.array(j["policy"], dtype=np.float32)
                    elif "policy_topk" in j:
                        # نبني توزيع 4672 لاحقًا من topk (إن وُجد)
                        it["policy_topk"] = j["policy_topk"]
                    self.items.append(it)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        planes = it["planes"].astype(np.float32)
        value = np.array([it.get("value", 0.0)], dtype=np.float32)
        # policy: قد تكون vector 4672 أو topk أو غير موجودة
        pol = it.get("policy", None)
        if pol is not None:
            pol = pol.astype(np.float32)
        else:
            pol = None
        topk = it.get("policy_topk", None)
        return planes, value, pol, topk

# =========================
# Losses
# =========================
class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    def forward(self, pred, target):
        err = pred - target
        abs_err = torch.abs(err)
        quad = torch.minimum(abs_err, torch.tensor(self.delta, device=pred.device))
        # 0.5 * quad^2 + delta*(|e|-delta) for large e
        loss = 0.5 * quad**2 + self.delta * (abs_err - quad)
        return loss.mean()

def policy_ce_from_vector(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """
    CE بين logits (N,4672) و target_dist (N,4672) normalized.
    """
    target = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-8)
    logp = F.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()

# =========================
# Train
# =========================
def _ensure_policy_size(policy_tensor: torch.Tensor) -> torch.Tensor:
    """
    يضمن إن policy لها الحجم POLICY_SIZE. لو أكبر: هنقص.
    لو أصغر: هنـpad بزيرو لحد ما نوصل الحجم المطلوب.
    """
    n, m = policy_tensor.shape
    if m == POLICY_SIZE:
        return policy_tensor
    if m > POLICY_SIZE:
        return policy_tensor[:, :POLICY_SIZE]
    # pad
    pad = torch.zeros((n, POLICY_SIZE - m), dtype=policy_tensor.dtype, device=policy_tensor.device)
    return torch.cat([policy_tensor, pad], dim=1)

def train_distill(
    dataset_path: Path,
    out_path: Path,
    epochs: int = 2,
    batch_size: int = 1024,
    lr: float = 1e-3,
    huber_delta: float = 1.0,
    tanh_scale: Optional[float] = None,
    model_name: Optional[str] = "auto",
    grad_clip: float = 1.0,
    num_workers: int = 0,
    seed: int = 42,
):
    set_seed(seed)
    dev = device_auto()
    ds = DistillDataset(dataset_path, tanh_scale=tanh_scale)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty or unreadable.")

    # استنتاج عدد القنوات من أول عنصر
    planes0, _, _, _ = ds[0]
    in_ch = planes0.shape[0]

    model = load_user_model_or_fallback(model_name, in_channels=in_ch)
    model.to(dev)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))

    # خسائر
    v_loss_fn = HuberLoss(delta=huber_delta)
    p_loss_weight = 1.0
    v_loss_weight = 1.0

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    # داتالودر
    pin_mem = (dev.type == "cuda")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)

    best_loss = float("inf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(".ckpt.pt")

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for planes, value, policy, policy_topk in dl:
            planes = torch.tensor(planes, dtype=torch.float32, device=dev)
            value  = torch.tensor(value, dtype=torch.float32, device=dev)

            # policy vector (لو موجودة) أو بناءها من policy_topk
            if policy is not None:
                policy = torch.tensor(policy, dtype=torch.float32, device=dev)
                policy = _ensure_policy_size(policy)
            elif policy_topk is not None:
                # policy_topk هو "قائمة من القوائم" (batch of lists)
                batch_vecs = []
                for topk in policy_topk:
                    vec = np.zeros((POLICY_SIZE,), dtype=np.float32)
                    if topk:
                        for it in topk:
                            mv = it.get("uci")
                            pr = float(it.get("p", 0.0))
                            idx = uci_to_index(mv, fen=None)  # ممكن تمرّر fen لو متاحة
                            if 0 <= idx < POLICY_SIZE:
                                vec[idx] = pr
                    batch_vecs.append(vec)
                policy = torch.tensor(np.stack(batch_vecs), dtype=torch.float32, device=dev)
            else:
                policy = None

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev.type=="cuda")):
                plogits, vpred = model(planes)
                # value loss
                v_l = v_loss_fn(vpred, value)

                # policy loss (لو vector موجودة)
                if policy is not None:
                    # تأكدنا إن حجمها POLICY_SIZE
                    p_l = policy_ce_from_vector(plogits, policy)
                else:
                    p_l = torch.tensor(0.0, device=dev)

                loss = v_loss_weight * v_l + p_loss_weight * p_l

            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            n_batches += 1

        sched.step()
        ep_loss = running / max(1, n_batches)
        log(f"Epoch {ep}/{epochs} | train_loss={ep_loss:.4f} | lr={sched.get_last_lr()[0]:.6f}")

        # حفظ الأفضل
        if ep_loss < best_loss:
            best_loss = ep_loss
            save_checkpoint(
                {"model": model.state_dict(), "epoch": ep, "loss": ep_loss},
                ckpt_path
            )
            torch.save(model.state_dict(), out_path.as_posix())  # الوزن فقط
            log(f"✅ Saved best to {out_path}")

    log("Done.")

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to .npz or .jsonl")
    ap.add_argument("--out", required=True, help="Output model .pt path")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--tanh-scale", type=float, default=None, help="If cp provided, value=tanh(cp/scale). E.g., 400")
    ap.add_argument("--model", type=str, default="auto", help="Import path to your model 'pkg:Class' or 'auto'")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_distill(
        dataset_path=Path(args.dataset),
        out_path=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        huber_delta=args.huber_delta,
        tanh_scale=args.tanh_scale,
        model_name=args.model,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        seed=args.seed,
    )

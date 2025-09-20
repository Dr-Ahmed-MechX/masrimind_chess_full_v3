"""
MasriMind Chess — Unified Training Orchestrator
------------------------------------------------
Usage:
    python -m training.run_all --cfg training/config.yaml

I can override config keys using:
    --set key=value
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from .utils import log, ensure_dir, run_cmd, load_yaml, dump_yaml, timestamp_dir, symlink_or_copy

def deep_set(d: Dict[str, Any], dotted: str, value: Any):
    keys = dotted.split('.')
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    v = value
    if isinstance(value, str):
        if value.lower() in ("true", "false"):
            v = value.lower() == "true"
        else:
            try:
                v = float(v) if '.' in v else int(v)
            except:
                pass
    cur[keys[-1]] = v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="training/config.yaml")
    ap.add_argument("--set", nargs='*', default=[])
    args = ap.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        log(f"Config not found: {cfg_path}")
        sys.exit(1)

    cfg = load_yaml(cfg_path)
    for s in args.set:
        if '=' not in s:
            continue
        k, v = s.split('=', 1)
        deep_set(cfg, k, v)

    out_root = Path(cfg.get("out_root", "training"))
    run_dir = timestamp_dir(out_root / "runs", cfg.get("run_tag", "run"))
    dump_yaml(cfg, run_dir / "resolved_config.yaml")

    python = sys.executable

    # 1) Dataset creation
    if cfg.get("distill", {}).get("make_dataset"):
        dd = cfg["distill"]
        cmd = [python, "-m", dd.get("make_dataset_script", "training.make_dataset_distill"),
               "--pgn", dd["pgn"], "--out", str(run_dir / "distill_dataset.npz"),
               "--max-plies", str(dd["max_plies"]), "--depth", str(dd["depth"])]
        run_cmd(cmd)

    # 2) Distillation training
    if cfg.get("distill", {}).get("enabled", True):
        dd = cfg["distill"]
        model_out = str(run_dir / "pvnet_best.pt")
        dataset = dd.get("dataset") or str(run_dir / "distill_dataset.npz")
        cmd = [python, "-m", dd["train_script"],
               "--dataset", dataset,
               "--epochs", str(dd["epochs"]),
               "--batch-size", str(dd["batch_size"]),
               "--lr", str(dd["lr"]),
               "--out", model_out]
        if dd.get("huber_delta"):
            cmd += ["--huber-delta", str(dd["huber_delta"])]
        if dd.get("tanh_scale"):
            cmd += ["--tanh-scale", str(dd["tanh_scale"])]
        run_cmd(cmd)
        symlink_or_copy(Path(model_out), Path(dd["best_model_name"]))

    # 3) Self-play
    if cfg.get("selfplay", {}).get("enabled", True):
        sp = cfg["selfplay"]
        cmd = [python, "-m", sp["script"],
               "--games", str(sp["games"]),
               "--mcts-simulations", str(sp["mcts_simulations"]),
               "--net", cfg["distill"]["best_model_name"],
               "--out", str(run_dir / "selfplay")]
        run_cmd(cmd)

    # 4) RL training
    if cfg.get("rl", {}).get("enabled", True):
        rl = cfg["rl"]
        cmd = [python, "-m", rl["train_script"],
               "--data", rl.get("data") or str(run_dir / "selfplay"),
               "--net", cfg["distill"]["best_model_name"],
               "--out", str(run_dir / "pvnet_pv_rl.pt"),
               "--epochs", str(rl["epochs"]),
               "--batch-size", str(rl["batch_size"]),
               "--lr", str(rl["lr"])]
        if rl.get("kl_to_stockfish"):
            cmd += ["--kl-to-stockfish", "true"]
        if rl.get("kl_weight"):
            cmd += ["--kl-weight", str(rl["kl_weight"])]
        run_cmd(cmd)
        symlink_or_copy(Path(run_dir / "pvnet_pv_rl.pt"), Path(rl["active_model_name"]))

    # 5) Evaluation
    if cfg.get("eval", {}).get("enabled", False):
        ev = cfg["eval"]
        cmd = [python, "-m", ev["script"],
               "--net", cfg["rl"]["active_model_name"],
               "--games", str(ev["games"]),
               "--out", str(run_dir / "eval.json")]
        if ev.get("stockfish_path"):
            cmd += ["--stockfish", ev["stockfish_path"]]
        run_cmd(cmd)

    log("✅ Training complete. Check training/models/")

if __name__ == "__main__":
    main()

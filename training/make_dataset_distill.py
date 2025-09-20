from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import List, Dict, Optional

import chess
import chess.pgn
import chess.engine

import yaml

from training.train_data_utils import cp_to_value_tanh

def load_cfg(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def evaluate_position(engine: chess.engine.SimpleEngine,
                      board: chess.Board,
                      limit: chess.engine.Limit) -> Dict:
    """
    Returns dict with:
      - 'cp': centipawns from side-to-move (int) or None
      - 'mate': signed ply-to-mate if mate available else None
      - 'top_moves': list of tuples (uci, score_cp) for top moves
    """
    # MultiPV to get top K moves
    info = engine.analyse(board, limit=limit, multipv=5)
    top_moves = []
    cp = None
    mate = None
    # info may be a list (MultiPV) or dict (single)
    if isinstance(info, list):
        for it in info:
            move = it.get("pv", [None])[0]
            if move is None:
                continue
            sc = it.get("score")
            score_cp = None
            if sc is not None:
                if sc.is_mate():
                    mate = sc.mate()  # keep last mate seen
                    # map mate to large cp for policy softmax
                    score_cp = 100000 if mate and mate > 0 else -100000
                else:
                    score_cp = sc.white().score(mate_score=100000)
                    # adapt perspective to side-to-move
                    if not board.turn:  # black to move
                        score_cp = -score_cp
                    if cp is None:
                        cp = score_cp
            top_moves.append((move.uci(), score_cp if score_cp is not None else 0))
    else:
        it = info
        move = it.get("pv", [None])[0]
        sc = it.get("score")
        if sc is not None:
            if sc.is_mate():
                mate = sc.mate()
                cp = 100000 if mate and mate > 0 else -100000
            else:
                cp = sc.white().score(mate_score=100000)
                if not board.turn:
                    cp = -cp
        if move is not None:
            top_moves = [(move.uci(), cp if cp is not None else 0)]

    return {"cp": cp, "mate": mate, "top_moves": top_moves}

def softmax(xs: List[float], temp: float = 400.0) -> List[float]:
    # temperature in cp units; larger -> flatter
    import math
    mx = max(xs)
    exps = [math.exp((x - mx) / max(1e-6, temp)) for x in xs]
    s = sum(exps)
    if s <= 0:
        n = len(xs)
        return [1.0 / n] * n
    return [e / s for e in exps]

def generate_from_pgn(pgn_path: str,
                      out_jsonl: str,
                      cfg: dict,
                      max_positions: int = 10000,
                      every_halfmove: int = 1,
                      sf_depth: int = 16,
                      topk: int = 5,
                      value_scale: float = 600.0):
    """
    - pgn_path: مصدر الألعاب
    - out_jsonl: ملف الناتج (JSONL)
    - every_halfmove: 1 يعني كل نقلة، 2 يعني كل نقلة أخرى، ...
    """
    paths = cfg.get("paths", {}) or {}
    sf_path = paths.get("stockfish", "assets/sf/stockfish.exe")
    assert Path(sf_path).exists(), f"Stockfish not found at {sf_path}"

    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = out.open("w", encoding="utf-8")

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    # Use depth or time; depth is deterministic-ish
    limit = chess.engine.Limit(depth=sf_depth)

    total = 0
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()

            ply = 0
            for mv in game.mainline_moves():
                board.push(mv)
                ply += 1
                if ply % every_halfmove != 0:
                    continue

                info = evaluate_position(engine, board, limit)
                cp = info["cp"]
                top = info["top_moves"][:topk]
                # Build policy_dict via softmax over cp scores
                scores = [sc for (_, sc) in top]
                if len(scores) == 0:
                    continue
                probs = softmax(scores, temp=400.0)
                policy_dict = {uci: float(p) for (uci, _), p in zip(top, probs)}

                # Scale value via tanh(cp/scale)
                if cp is None:
                    # if mate inferred, sign only
                    v = 1.0 if info["mate"] and info["mate"] > 0 else (-1.0 if info["mate"] and info["mate"] < 0 else 0.0)
                else:
                    v = cp_to_value_tanh(float(cp), scale=value_scale)

                line = {
                    "fen": board.fen(),
                    "policy_dict": policy_dict,  # sparse (better للـ IO)
                    "value": float(v)
                }
                w.write(json.dumps(line, ensure_ascii=False) + "\n")
                total += 1
                if total >= max_positions:
                    break
            if total >= max_positions:
                break

    engine.quit()
    w.close()
    print(f"Done. Wrote {total} positions to {out_jsonl}")

def cli():
    ap = argparse.ArgumentParser("make_dataset_distill")
    ap.add_argument("--pgn", required=True, help="Input PGN file")
    ap.add_argument("--out", default="training/data/distill_raw.jsonl", help="Output JSONL")
    ap.add_argument("--max_positions", type=int, default=10000)
    ap.add_argument("--every_halfmove", type=int, default=1)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--value_scale", type=float, default=600.0)
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--split", action="store_true", help="Split into train/val after generation")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    generate_from_pgn(
        pgn_path=args.pgn,
        out_jsonl=args.out,
        cfg=cfg,
        max_positions=args.max_positions,
        every_halfmove=args.every_halfmove,
        sf_depth=args.depth,
        topk=args.topk,
        value_scale=args.value_scale
    )

    if args.split:
        from training.train_data_utils import split_jsonl
        out = Path(args.out)
        train_out = out.with_name(out.stem.replace("_raw", "") + "_train.jsonl")
        val_out = out.with_name(out.stem.replace("_raw", "") + "_val.jsonl")
        split_jsonl(str(out), str(train_out), str(val_out), val_ratio=args.val_ratio)

if __name__ == "__main__":
    cli()

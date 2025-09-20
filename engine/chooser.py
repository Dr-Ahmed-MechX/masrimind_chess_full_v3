# engine/chooser.py
import os
import pathlib
from dataclasses import dataclass
from typing import Optional, Dict, Any

import chess
import chess.engine

# ============== Config ==============
STOCKFISH_PATHS = [
    "assets/sf/stockfish.exe",          # Windows
    "assets/sf/stockfish",              # Linux/Mac (لو أضفت بناري)
]
DEFAULT_DEPTH = 18
DEFAULT_SKILL = 20

MODEL_CANDIDATES = [
    "training/pvnet_pv_rl.pt",
    "training/pvnet_best.pt",
]

@dataclass
class Explain:
    score_cp: Optional[int] = None
    score_mate: Optional[int] = None
    pv: Optional[str] = None

class HybridChooser:
    def __init__(self, model_path: Optional[str] = None, depth:int=DEFAULT_DEPTH, skill:int=DEFAULT_SKILL):
        self.depth = depth
        self.skill = skill
        self.model_path = self._resolve_model(model_path)
        self.engine = self._open_stockfish()
        self.last_explain: Optional[Explain] = None
        # NOTE: مكان تحميل الشبكة (إن وجدت) — سايبه placeholder لحد ما تكمّل تدريبك.
        self.nn_loaded = self.model_path is not None and os.path.exists(self.model_path)

    def _resolve_model(self, explicit: Optional[str]) -> Optional[str]:
        if explicit and os.path.exists(explicit):
            return explicit
        for p in MODEL_CANDIDATES:
            if os.path.exists(p):
                return p
        return None  # هنكمل بستوكفيش فقط

    def _open_stockfish(self):
        for p in STOCKFISH_PATHS:
            if os.path.exists(p):
                try:
                    return chess.engine.SimpleEngine.popen_uci(p)
                except Exception:
                    pass
        raise RuntimeError("Stockfish binary not found. Put it in assets/sf/")

    def close(self):
        try:
            self.engine.close()
        except Exception:
            pass

    def legal_moves(self, fen: str):
        board = chess.Board(fen)
        return [m.uci() for m in board.legal_moves]

    def best_move(self, fen: str) -> Dict[str, Any]:
        """يرجّع dict فيه uci, san, score, pv."""
        board = chess.Board(fen)
        # ضبط إعدادات ستوكفيش
        try:
            self.engine.configure({"Skill Level": self.skill})
        except Exception:
            pass

        # نطلب تحليل
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=1)
        move = info.get("pv", [None])[0]
        if move is None:
            # لو مفيش PV لأي سبب، خُد أول legal
            lm = list(board.legal_moves)
            move = lm[0] if lm else None
        if move is None:
            return {"uci": None, "san": None, "score": None, "pv": None}

        score = info.get("score")
        cp = score.white().score(mate_score=100000) if score else None
        mate = score.white().mate() if score and score.is_mate() else None
        pv_line = info.get("pv")
        pv_san = None
        if pv_line:
            tmp = board.copy()
            pv_san = []
            for m in pv_line:
                pv_san.append(tmp.san(m))
                tmp.push(m)
            pv_san = " ".join(pv_san)

        # حفظ آخر تفسير
        self.last_explain = Explain(score_cp=cp if mate is None else None,
                                    score_mate=mate,
                                    pv=pv_san)

        san = board.san(move)
        return {
            "uci": move.uci(),
            "san": san,
            "score": {"cp": cp, "mate": mate},
            "pv": pv_san,
            "used_model": bool(self.nn_loaded),
            "model_path": self.model_path,
        }

    def explain_last(self) -> Dict[str, Any]:
        e = self.last_explain or Explain()
        return {"score_cp": e.score_cp, "score_mate": e.score_mate, "pv": e.pv}

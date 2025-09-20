# engine/uci_stockfish.py
# Lightweight UCI wrapper for Stockfish with safe option configuration
# Works with Stockfish 17+ (no 'Contempt' option)

import os
import yaml
import chess
import chess.engine

# ---------- Load & normalize config ----------
# Resolve project root ( .. from this file => repo root )
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _load_cfg():
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = _load_cfg()

def _resolve_sf_path(path_str: str) -> str:
    """Return absolute path to Stockfish binary."""
    if not path_str:
        return ""
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(PROJECT_ROOT, path_str)

# ---------- UCI wrapper ----------
class StockfishUCI:
    """
    Small helper around python-chess engine API.
    - Safely sets supported options only (ignores unsupported like 'Contempt' in SF17).
    - Move time/depth/nodes limits configurable from config.yaml:
        stockfish:
          path: assets/sf/stockfish.exe
          threads: 4
          hash_mb: 256
          skill_level: 20
          # optional:
          # move_time_ms: 700
          # depth: 12
          # nodes: 200000
    """
    def __init__(self, verbose: bool = False):
        sf_cfg = CFG.get("stockfish", {}) or {}
        sf_path = _resolve_sf_path(sf_cfg.get("path", ""))

        if not sf_path or not os.path.exists(sf_path):
            raise FileNotFoundError(
                f"Stockfish not found at: {sf_path or '<empty path>'}. "
                f"Update 'stockfish.path' in config.yaml or place the engine at assets/sf/stockfish.exe"
            )

        # Launch engine process
        self.engine = chess.engine.SimpleEngine.popen_uci(sf_path)

        # Configure supported options safely (ignore unsupported)
        def _safe_config(name, value):
            try:
                self.engine.configure({name: value})
            except Exception:
                # Option not supported by this engine build/version -> ignore
                if verbose:
                    print(f"[StockfishUCI] Option '{name}' not supported; skipped.")

        _safe_config("Skill Level", int(sf_cfg.get("skill_level", 20)))
        _safe_config("Threads",     int(sf_cfg.get("threads", 4)))
        _safe_config("Hash",        int(sf_cfg.get("hash_mb", 256)))
        # DO NOT set "Contempt" – removed in Stockfish 17+

        # Build thinking limit: prefer depth > nodes > time
        depth = sf_cfg.get("depth", None)
        nodes = sf_cfg.get("nodes", None)
        mt_ms = sf_cfg.get("move_time_ms", 700)

        if isinstance(depth, int) and depth > 0:
            self.limit = chess.engine.Limit(depth=depth)
        elif isinstance(nodes, int) and nodes > 0:
            self.limit = chess.engine.Limit(nodes=nodes)
        else:
            self.limit = chess.engine.Limit(time=max(0.05, float(mt_ms) / 1000.0))

        if verbose:
            print(f"[StockfishUCI] Engine: {sf_path}")
            print(f"[StockfishUCI] Limit: {self.limit}")

    # Context manager convenience
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------- API ----------
    def best_move(self, board: chess.Board) -> chess.Move:
        """Return best move under configured limit."""
        return self.engine.play(board, self.limit).move

    def evaluate_cp(self, board: chess.Board, depth: int = None) -> int:
        """
        Return centipawn eval POV side-to-move.
        If depth is provided, it overrides the default limit for this call.
        Mates are mapped to large centipawn values via mate_score.
        """
        if depth is not None and depth > 0:
            info = self.engine.analyse(board, chess.engine.Limit(depth=int(depth)))
        else:
            # fall back to the same limit family used for best_move
            info = self.engine.analyse(board, self.limit)
        return info["score"].pov(board.turn).score(mate_score=100000)

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass

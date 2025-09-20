import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("assets/sf/stockfish.exe")
board = chess.Board()
info = engine.analyse(board, chess.engine.Limit(depth=10))
print("✅ Evaluation:", info["score"])
engine.quit()

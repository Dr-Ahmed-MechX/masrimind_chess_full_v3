import chess
class Board:
    def __init__(self): self.board = chess.Board()
    def push_uci(self, uci:str): self.board.push_uci(uci)
    def is_game_over(self): return self.board.is_game_over()
    def result(self): return self.board.result()
    def fen(self): return self.board.fen()

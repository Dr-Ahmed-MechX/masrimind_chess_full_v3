# training/encode.py
import numpy as np
import chess

PIECE_TO_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}
N_PLANES = 12  # 6 أبيض + 6 أسود
EXTRA = 3     # side-to-move, castling, no-progress (اختياري)
BOARD_SHAPE = (N_PLANES + EXTRA, 8, 8)

def fen_to_tensor(fen: str) -> np.ndarray:
    """Encode FEN to simple planes (C,H,W)."""
    board = chess.Board(fen)
    x = np.zeros(BOARD_SHAPE, dtype=np.float32)

    # قطع
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece: continue
        p = PIECE_TO_PLANE.get(piece.piece_type, None)
        if p is None: continue
        row = 7 - chess.square_rank(sq)
        col = chess.square_file(sq)
        idx = p if piece.color == chess.WHITE else p + 6
        x[idx, row, col] = 1.0

    # side-to-move
    x[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # castling rights (ببساطة: عدد الحقوق / 4)
    cr = sum([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ])
    x[13, :, :] = cr / 4.0

    # halfmove clock scaled
    x[14, :, :] = min(board.halfmove_clock, 50) / 50.0

    return x

# training/move_index.py
from __future__ import annotations
import chess

# مساحة الأفعال 4672 = 4096 (from-to) + 576 (ترقيات 9 طيارات × 64 مربع منشأ)
POLICY_SIZE = 4672
FROM_TO_SIZE = 64 * 64
PROMO_PLANES_PER_FROM = 9  # (forward|left|right) × (N,B,R)

PROMO_ORDER = ["n", "b", "r"]  # بنحط N,B,R بس كـ planes، والـ Queen نرميها في from-to

def square_to_rc(sq: int):
    return divmod(sq, 8)  # (rank, file) 0..7, 0..7

def uci_to_index(uci: str, fen: str | None = None) -> int:
    """
    Mapping عملي وسهل:
    - أول 4096: from*64 + to  (وده يشمل أي حركة غير الترقية أو ترقية لملكة)
    - آخر 576:  ترقيات (N,B,R فقط) = 9 planes/منشأ:
        planes order:
            0: forward+N, 1: forward+B, 2: forward+R,
            3: left+N,    4: left+B,    5: left+R,
            6: right+N,   7: right+B,   8: right+R
      direction relative to side-to-move (لو عايز تعرف اللون استخدم fen).
    - ترقية لـ Queen نرميها على الـ from-to العادي (هتلاقي لها خانة).
    """
    try:
        mv = chess.Move.from_uci(uci)
    except Exception:
        return -1

    from_sq, to_sq, promo = mv.from_square, mv.to_square, mv.promotion
    base = from_sq * 64 + to_sq

    # لو مش ترقية، أو ترقية لـ Queen: استخدم from-to المباشر
    if promo is None or promo == chess.QUEEN:
        return base  # 0..4095

    # ترقيات N/B/R: لازم نتعرف الاتجاه (أبيض ولا أسود)
    # محتاجين fen عشان نحدد الدور الحالي. لو مش موجودة هنحاول نخمن من اتجاه الحركة.
    side_white = True
    if fen:
        try:
            b = chess.Board(fen)
            side_white = b.turn == chess.WHITE
        except Exception:
            side_white = True
    # اتجاه "للأمام" للابيض = +8 رتب، للأسود = -8
    fr_rank, fr_file = divmod(from_sq, 8)
    to_rank, to_file = divmod(to_sq, 8)
    dr = to_rank - fr_rank
    df = to_file - fr_file

    # حدّد اتجاه أمام/يسار/يمين بالنسبة للّون اللي هيلعب:
    # الأبيض: forward = +1 rank (dr=+1)، left = df=-1، right = df=+1
    # الأسود: forward = -1 rank (dr=-1)، left/right معكوسين بالملف؟
    # نعتبر "left" هو القبلة‑على‑يسار اللاعب؛
    if side_white:
        forward = (dr == 1 and df == 0)
        left    = (dr == 1 and df == -1)
        right   = (dr == 1 and df == +1)
    else:
        forward = (dr == -1 and df == 0)
        # لليسار بالنسبة لأسود: التحرك ناحية +file (عكس الأبيض)
        left    = (dr == -1 and df == +1)
        right   = (dr == -1 and df == -1)

    # حدّد أي plane (0..8)
    dir_idx = None
    if forward: dir_idx = 0
    elif left:  dir_idx = 1
    elif right: dir_idx = 2
    else:
        # لو الاتجاه مش واحد من دول (نادر)، ارجع from-to
        return base

    promo_ch = chess.piece_symbol(promo).lower()
    if promo_ch not in PROMO_ORDER:
        # أي ترقية غير N/B/R (زي Q) → from-to
        return base

    piece_idx = PROMO_ORDER.index(promo_ch)  # 0..2
    plane_idx = dir_idx * 3 + piece_idx      # 0..8

    return FROM_TO_SIZE + (from_sq * PROMO_PLANES_PER_FROM) + plane_idx  # 4096..4671

def index_to_uci(index: int, fen: str | None = None) -> str | None:
    """
    عكس المابات (تقريبي).
    - لأول 4096: استرجاع من/to.
    - لPlanes الترقية: بيبني حركة ترقية (N/B/R) في اتجاه forward/left/right حسب الدور من الـFEN (أو يفترض أبيض).
    """
    if not (0 <= index < POLICY_SIZE):
        return None
    if index < FROM_TO_SIZE:
        from_sq = index // 64
        to_sq   = index % 64
        return chess.Move(from_sq, to_sq).uci()

    # ترقيات
    rel = index - FROM_TO_SIZE
    from_sq = rel // PROMO_PLANES_PER_FROM
    plane_idx = rel % PROMO_PLANES_PER_FROM
    dir_idx, piece_idx = divmod(plane_idx, 3)

    side_white = True
    if fen:
        try:
            b = chess.Board(fen)
            side_white = b.turn == chess.WHITE
        except Exception:
            side_white = True

    fr_rank, fr_file = divmod(from_sq, 8)
    if side_white:
        if dir_idx == 0:  # forward
            to_rank, to_file = fr_rank + 1, fr_file
        elif dir_idx == 1:  # left
            to_rank, to_file = fr_rank + 1, fr_file - 1
        else:  # right
            to_rank, to_file = fr_rank + 1, fr_file + 1
    else:
        if dir_idx == 0:
            to_rank, to_file = fr_rank - 1, fr_file
        elif dir_idx == 1:
            to_rank, to_file = fr_rank - 1, fr_file + 1
        else:
            to_rank, to_file = fr_rank - 1, fr_file - 1

    if not (0 <= to_rank < 8 and 0 <= to_file < 8):
        return None
    to_sq = to_rank * 8 + to_file
    promo_piece = {"n": chess.KNIGHT, "b": chess.BISHOP, "r": chess.ROOK}[PROMO_ORDER[piece_idx]]
    return chess.Move(from_sq, to_sq, promotion=promo_piece).uci()

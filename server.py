# server.py
import os, io, json, time, ssl, smtplib, yaml, sqlite3
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from functools import wraps
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash

import chess
import chess.pgn

# شبكتك (اختياري)
try:
    from engine.network_engine import NetworkEngine, network_available
except Exception:
    def network_available(): return False
    class NetworkEngine:
        ready = False

# --- مشروعنا ---
from db import init_schema, connect

# واجهة ستوكفيش الخفيفة (موجودة في engine/uci_stockfish.py)
try:
    from engine.uci_stockfish import StockfishUCI
except Exception:
    StockfishUCI = None  # fallback لاحقًا

APP_ROOT = Path(__file__).resolve().parent

# -------- تحميل الكونفيج مع دعم متغيرات البيئة داخل ${VAR:default} ----------
def _expand_env(val: str) -> str:
    if not isinstance(val, str):
        return val
    out = val
    for _ in range(3):
        import re
        def repl(m):
            key = m.group(1)
            default = m.group(2) or ""
            return os.getenv(key, default)
        out = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:]?(.*?)\}", repl, out)
    return out

with open(APP_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# وسّع المتغيرات داخل القيم النصيّة
for section in list(cfg.keys()):
    if isinstance(cfg[section], dict):
        for k, v in list(cfg[section].items()):
            cfg[section][k] = _expand_env(v)

# -------- Flask --------
app = Flask(
    __name__,
    template_folder=str(APP_ROOT / "web" / "templates"),
    static_folder=str(APP_ROOT / "web" / "static"),
)
app.secret_key = cfg["app"]["secret_key"]

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# DB
DB_URL = cfg["database"]["url"]
init_schema(DB_URL)

# -------- إعداد ستوكفيش (مع غطاء احتياطي) --------
stockfish = None
sf_error = None
if StockfishUCI is not None:
    try:
        stockfish = StockfishUCI(verbose=False)
    except Exception as e:
        sf_error = str(e)
else:
    sf_error = "StockfishUCI module unavailable"

def best_move_stockfish(board: chess.Board) -> str:
    """أفضل نقلة من ستوكفيش أو fallback عشوائي (قانوني)."""
    # حاول ستوكفيش
    if stockfish:
        try:
            mv = stockfish.best_move(board)  # قد يرجّع Move أو UCI
            if isinstance(mv, chess.Move):
                return mv.uci()
            if isinstance(mv, str):
                return mv
        except Exception:
            pass
    # Fallback: نقلة قانونية عشوائية
    import random
    return random.choice(list(board.legal_moves)).uci()

# الشبكة (لو متاحة)
net_engine = None
try:
    if network_available():
        model_path = cfg["engine"].get("network_model", "training/pvnet_pv_rl.pt")
        net_engine = NetworkEngine(model_path=model_path)
except Exception:
    net_engine = None

def best_move_ai(board: chess.Board) -> str:
    # الشبكة أولاً (لو جاهزة)
    try:
        if net_engine and getattr(net_engine, "ready", False):
            mv = net_engine.best_move(board)
            if isinstance(mv, chess.Move):
                return mv.uci()
            if isinstance(mv, str):
                return mv
    except Exception:
        pass
    # ثم ستوكفيش
    return best_move_stockfish(board)

# -------- أدوات الأمان --------
def login_required(fn):
    @wraps(fn)
    def _wrap(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return _wrap

# -------- Routes: Auth --------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        pw = request.form.get("password") or ""
        if not email or not pw:
            flash("Email & password required.")
            return redirect(url_for("signup"))
        with connect(DB_URL) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE email=?", (email,))
            if cur.fetchone():
                flash("Email already registered.")
                return redirect(url_for("signup"))
            cur.execute("INSERT INTO users(email, pw_hash) VALUES (?,?)",
                        (email, generate_password_hash(pw)))
            conn.commit()
        flash("Account created. Please log in.")
        return redirect(url_for("login"))
    return render_template("signup.html", brand_name=cfg["app"]["brand"])

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        pw = request.form.get("password") or ""
        with connect(DB_URL) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, pw_hash FROM users WHERE email=?", (email,))
            row = cur.fetchone()
        if not row or not check_password_hash(row["pw_hash"], pw):
            flash("Invalid credentials.")
            return redirect(url_for("login"))
        session["user_id"] = row["id"]
        session["email"] = email
        return redirect(url_for("home"))
    return render_template("login.html", brand_name=cfg["app"]["brand"])

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# -------- Routes: Home/Play --------
@app.route("/")
@login_required
def home():
    return render_template(
        "index.html",
        brand_name=cfg["app"]["brand"],
        sf_status=("OK" if stockfish else f"DISABLED: {sf_error or 'N/A'}")
    )

# -------- Socket.IO: بدء مباراة + استقبال الحركات --------
@socketio.on("new_game")
def on_new_game(data):
    uid = session.get("user_id")
    if not uid:
        return {"ok": False, "error": "Unauthorized"}  # ACK

    color = (data or {}).get("color") or "white"
    with connect(DB_URL) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO games(user_id, color, start_fen) VALUES (?,?,?)",
                    (uid, color, chess.STARTING_FEN))
        gid = cur.lastrowid
        conn.commit()

    ai_first_move = None

    # لو المستخدم اختار أسود، الـ AI (أبيض) يبدأ فورًا
    if color == "black":
        board = chess.Board()
        ai_uci = best_move_ai(board)
        board.push_uci(ai_uci)
        with connect(DB_URL) as conn:
            cur = conn.cursor()
            cur.execute("""INSERT INTO moves(game_id, ply, side, fen_before, move_uci, move_san)
                           VALUES (?,?,?,?,?,?)""",
                        (gid, 1, "ai", chess.STARTING_FEN, ai_uci, board.peek().uci()))
            conn.commit()
        ai_first_move = ai_uci
        # نبعته أيضًا كإيفنت (اختياري، للـRT updates)
        emit("ai_move", {"game_id": gid, "move_uci": ai_uci})

    # رجّع ACK فيه أول نقلة لو موجودة
    return {"ok": True, "game_id": gid, "your_color": color, "ai_move": ai_first_move}


@socketio.on("user_move")
def on_user_move(data):
    uid = session.get("user_id")
    if not uid:
        return {"ok": False, "error": "Unauthorized"}  # ACK

    try:
        gid = int((data or {}).get("game_id") or 0)
    except Exception:
        gid = 0
    uci = (data or {}).get("move_uci")
    if not gid or not uci:
        return {"ok": False, "error": "Bad payload"}  # ACK

    # حمّل آخر حالة من الMoves لإعادة بناء البورد
    with connect(DB_URL) as conn:
        cur = conn.cursor()
        cur.execute("SELECT start_fen FROM games WHERE id=? AND user_id=?", (gid, uid))
        row = cur.fetchone()
        if not row:
            return {"ok": False, "error": "Game not found"}  # ACK
        start_fen = row["start_fen"]
        cur.execute("SELECT fen_before, move_uci FROM moves WHERE game_id=? ORDER BY id ASC", (gid,))
        moves = cur.fetchall()

    board = chess.Board(start_fen)
    for m in moves:
        board.push_uci(m["move_uci"])
    fen_before = board.fen()

    # تحقّق قانونية حركة المستخدم
    try:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return {"ok": False, "reason": "illegal"}  # ACK
        board.push(move)
    except Exception:
        return {"ok": False, "reason": "parse"}  # ACK

    user_san = board.peek().uci()

    # سجّل حركة المستخدم
    with connect(DB_URL) as conn:
        cur = conn.cursor()
        cur.execute("""INSERT INTO moves(game_id, ply, side, fen_before, move_uci, move_san)
                       VALUES (?,?,?,?,?,?)""",
                    (gid, board.fullmove_number*2 - (0 if board.turn else 1), "user",
                     fen_before, uci, user_san))
        conn.commit()

    # لو الجيم خلص بعد حركة اليوزر
    if board.is_game_over():
        return {"ok": True, "game_over": True}  # ACK

    # دور الـ AI
    ai_uci = best_move_ai(board)
    board.push_uci(ai_uci)
    with connect(DB_URL) as conn:
        cur = conn.cursor()
        cur.execute("""INSERT INTO moves(game_id, ply, side, fen_before, move_uci, move_san)
                       VALUES (?,?,?,?,?,?)""",
                    (gid, board.fullmove_number*2 - (1 if board.turn else 0),
                     "ai", fen_before, ai_uci, board.peek().uci()))
        conn.commit()

    # بعتنا حركة الـ AI كـ event للعميل
    emit("ai_move", {"game_id": gid, "move_uci": ai_uci})

    # ورجّعنا ACK بنجاح حركة اليوزر
    return {"ok": True}

# -------- تقرير PDF + إيميل --------
def send_mail_pdf(to_email: str, subject: str, body_html: str, pdf_bytes: bytes):
    if not cfg.get("smtp") or not cfg["smtp"].get("host"):
        raise RuntimeError("SMTP not configured")
    msg = MIMEMultipart()
    msg["From"] = f'{cfg["smtp"]["from_name"]} <{cfg["smtp"]["from_email"]}>'
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body_html, "html"))
    att = MIMEApplication(pdf_bytes, _subtype="pdf")
    att.add_header("Content-Disposition", "attachment", filename="MasriMind-Report.pdf")
    msg.attach(att)
    context = ssl.create_default_context()
    with smtplib.SMTP(cfg["smtp"]["host"], int(cfg["smtp"]["port"])) as server:
        if cfg["smtp"].get("use_tls", True):
            server.starttls(context=context)
        if cfg["smtp"].get("username"):
            server.login(cfg["smtp"]["username"], cfg["smtp"]["password"])
        server.send_message(msg)

def generate_game_report(game_id: int) -> bytes:
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    import cairosvg
    import chess.svg as csvg

    # حمّل الحركات
    with connect(DB_URL) as conn:
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE id=(SELECT user_id FROM games WHERE id=?)",(game_id,))
        user_row = cur.fetchone()
        cur.execute("SELECT * FROM moves WHERE game_id=? ORDER BY id ASC", (game_id,))
        ms = cur.fetchall()

    # نعيد بناء البورد خطوة بخطوة
    board = chess.Board()
    pages = []

    def svg_to_png_bytes(svg_str):
        return cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), output_width=800, output_height=800)

    for i, m in enumerate(ms, 1):
        fen_before = m["fen_before"]
        played_uci = m["move_uci"]
        b = chess.Board(fen_before)

        # صورة "قبل"
        try:
            move_obj = chess.Move.from_uci(played_uci)
            svg1 = csvg.board(board=b, lastmove=move_obj, size=800, coordinates=True)
        except Exception:
            svg1 = csvg.board(board=b, size=800, coordinates=True)
        img1 = ImageReader(io.BytesIO(svg_to_png_bytes(svg1)))

        # اقترح الأفضل
        def best_suggestion(bb: chess.Board):
            try:
                if net_engine and getattr(net_engine, "ready", False):
                    mv = net_engine.best_move(bb)
                    if isinstance(mv, chess.Move):
                        return mv
                    if isinstance(mv, str):
                        return chess.Move.from_uci(mv)
            except Exception:
                pass
            bm_uci = best_move_stockfish(bb)
            return chess.Move.from_uci(bm_uci)

        best_mv = best_suggestion(b.copy())
        svg2 = csvg.board(board=b, arrows=[(best_mv.from_square, best_mv.to_square)],
                          size=800, coordinates=True)
        img2 = ImageReader(io.BytesIO(svg_to_png_bytes(svg2)))

        pages.append((i, played_uci, best_mv.uci(), img1, img2))

        # طبّق الحركة الفعلية على البورد الرئيسي
        try:
            board.push_uci(played_uci)
        except Exception:
            pass

    # رسم الـPDF
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    for idx, played, bestu, img_before, img_best in pages:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, h-2.2*cm, f"Move #{idx}")
        c.setFont("Helvetica", 11)
        c.drawString(2*cm, h-2.8*cm, f"Played: {played}     |     Recommended: {bestu}")
        img_w = (w - 4*cm) / 2
        img_h = img_w
        y = h - 3.0*cm - img_h
        c.drawImage(img_before, 2*cm, y, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
        c.drawImage(img_best, 2*cm + img_w + 0.5*cm, y, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
        c.showPage()
    c.save()
    return buf.getvalue()

@app.route("/api/finish_game", methods=["POST"])
@login_required
def finish_game():
    data = request.get_json(silent=True) or {}
    gid = int(data.get("game_id") or 0)
    if not gid:
        return jsonify({"ok": False, "error": "Missing game_id"}), 400
    pdf_bytes = generate_game_report(gid)
    if cfg["report"].get("enable"):
        email_to = cfg["smtp"].get("to_override") or session.get("email")
        try:
            send_mail_pdf(email_to, "MasriMind — Game Report", "<p>Your game report is attached.</p>", pdf_bytes)
        except Exception as e:
            return jsonify({"ok": True, "mail_error": str(e)})
    return jsonify({"ok": True})

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "stockfish": bool(stockfish),
        "sf_error": sf_error
    }), 200

if __name__ == "__main__":
    print("[BOOT] starting server on 127.0.0.1:5000 ...")
    socketio.run(app, host="127.0.0.1", port=5000, allow_unsafe_werkzeug=True)

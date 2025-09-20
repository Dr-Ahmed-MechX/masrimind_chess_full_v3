# db.py
import os, sqlite3
from contextlib import contextmanager

def ensure_dirs(db_path: str):
    d = os.path.dirname(db_path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

@contextmanager
def connect(db_path: str):
    ensure_dirs(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_schema(db_path: str):
    ensure_dirs(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            pw_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS games(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            color TEXT CHECK(color IN ('white','black')) NOT NULL,
            start_fen TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS moves(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            ply INTEGER NOT NULL,
            side TEXT CHECK(side IN ('user','ai')) NOT NULL,
            fen_before TEXT NOT NULL,
            move_uci TEXT NOT NULL,
            move_san TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(game_id) REFERENCES games(id)
        )""")
        conn.commit()

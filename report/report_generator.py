import os, io
from typing import List, Tuple
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm
import chess
import chess.svg
import cairosvg  # يأتي ضمن reportlab? لا، نزود عبر pip cairoSVG. بدلاً منها: هنستخدم matplotlib + pillow
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import subprocess

# ملاحظة: لتفادي cairoSVG، هنستخدم python-chess svg -> PNG عبر cairosvg
# ضف في requirements لو مش متاح: cairosvg>=2.7
# لو مش تحب cairosvg، نقدر نرسم بماتبلوتليب مربعات + صور SVG القطع من high_quality_staunton_pieces، بس ده أطول.
# هنا هنستعمل cairosvg لتبسيط التحويل.

def board_png_from_fen_and_move(fen: str, uci_move: str, title: str, pieces_dir: str) -> Image.Image:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)
    # رسم السهم/التمييز:
    squares = []
    if move:
        squares = [move.from_square, move.to_square]
    svg = chess.svg.board(board=board, arrows=[(move.from_square, move.to_square)],
                          lastmove=move, size=480)
    png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
    return Image.open(io.BytesIO(png_bytes))

def render_page_pair(pdf: canvas.Canvas, left_img: Image.Image, right_img: Image.Image,
                     brand_name: str, brand_logo_path: str, move_title: str):
    W, H = A4  # (595x842pt)
    margin = 1.2 * cm
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(margin, H - margin, f"{brand_name} — Post‑Game Move Report")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(margin, H - margin - 14, move_title)
    # لوجو
    if brand_logo_path and os.path.isfile(brand_logo_path):
        pdf.drawImage(brand_logo_path, W - margin - 2.5*cm, H - margin - 2.5*cm, 2.5*cm, 2.5*cm, mask='auto')

    # صور اللوحين
    max_w = (W - 3*margin) / 2
    max_h = H - 5*margin
    # نحافظ على المربعات
    left_buf = io.BytesIO()
    left_img.save(left_buf, format="PNG")
    right_buf = io.BytesIO()
    right_img.save(right_buf, format="PNG")
    left_reader = ImageReader(io.BytesIO(left_buf.getvalue()))
    right_reader = ImageReader(io.BytesIO(right_buf.getvalue()))
    pdf.drawImage(left_reader, margin, margin*2, max_w, max_w, preserveAspectRatio=True, mask='auto')
    pdf.drawImage(right_reader, margin*2 + max_w, margin*2, max_w, max_w, preserveAspectRatio=True, mask='auto')
    pdf.showPage()

def generate_pdf_report(out_path: str,
                        brand_name: str,
                        brand_logo_path: str,
                        pieces_dir: str,
                        move_rows: List[Tuple[int,str,str,str]]):
    """
    move_rows: list of (ply, fen_before, user_uci, best_uci)
    """
    c = canvas.Canvas(out_path, pagesize=A4)
    for ply, fen_before, user_uci, best_uci in move_rows:
        left = board_png_from_fen_and_move(fen_before, user_uci, "Your Move", pieces_dir)
        right = board_png_from_fen_and_move(fen_before, best_uci, "Recommended", pieces_dir)
        move_title = f"Move #{ply}: You played {user_uci} — Recommended {best_uci}"
        render_page_pair(c, left, right, brand_name, brand_logo_path, move_title)
    c.save()
    return out_path

import os
import shutil

# ✅ 1. المصدر: مكان ملفاتك المحمّلة
src = r"C:/Users/Dr.Eng. Ahmed Fathy/Downloads/JohnPablok Cburnett Chess Zip/JohnPablok Cburnett Chess set/SVG with shadow"

# ✅ 2. الوجهة النهائية داخل مشروعك
dst = r"E:/Python Projects\MySon/masrimind_chess_full_v3/web/static/pieces"
os.makedirs(dst, exist_ok=True)

# ✅ 3. خريطة التحويل: (أصل الاسم → الاسم الجديد)
mapping = {
    "w_pawn_svg_withShadow": "wp.svg",
    "w_knight_svg_withShadow": "wn.svg",
    "w_bishop_svg_withShadow": "wb.svg",
    "w_rook_svg_withShadow": "wr.svg",
    "w_queen_svg_withShadow": "wq.svg",
    "w_king_svg_withShadow": "wk.svg",
    "b_pawn_svg_withShadow": "bp.svg",
    "b_knight_svg_withShadow": "bn.svg",
    "b_bishop_svg_withShadow": "bb.svg",
    "b_rook_svg_withShadow": "br.svg",
    "b_queen_svg_withShadow": "bq.svg",
    "b_king_svg_withShadow": "bk.svg",
}

# ✅ 4. تنفيذ النقل وإعادة التسمية
count = 0
for old_name, new_name in mapping.items():
    old_path = os.path.join(src, old_name + ".svg")
    new_path = os.path.join(dst, new_name)
    if os.path.exists(old_path):
        shutil.copyfile(old_path, new_path)
        print(f"✅ {old_name}.svg → {new_name}")
        count += 1
    else:
        print(f"❌ Not found: {old_name}.svg")

print(f"\n🎉 Done! {count}/12 pieces copied to: {dst}")

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

ROOT = Path(__file__).resolve().parent

SRC_DIR = (
    ROOT
    / "data"
    / "niveau2_error_analysis"
    / "hard_negative_candidates"
)

files = sorted(
    p for p in SRC_DIR.glob("*.jpg")
    if "mosaique" not in p.name
)

OUT = SRC_DIR / "mosaique_numerotee.jpg"
MAP = SRC_DIR / "correspondance_numeros.txt"

CELL_W = 360
CELL_H = 390
COLS = 4

rows = math.ceil(len(files) / COLS)

canvas = Image.new(
    "RGB",
    (COLS * CELL_W, rows * CELL_H),
    "white"
)

mapping = []

for i, path in enumerate(files, start=1):

    img = Image.open(path).convert("RGB")
    img.thumbnail((340, 340))

    cell = Image.new(
        "RGB",
        (CELL_W, CELL_H),
        "white"
    )

    x = (CELL_W - img.width) // 2
    cell.paste(img, (x, 35))

    draw = ImageDraw.Draw(cell)

    draw.text(
        (10, 5),
        f"CANDIDAT {i}",
        fill="black"
    )

    col = (i - 1) % COLS
    row = (i - 1) // COLS

    canvas.paste(
        cell,
        (col * CELL_W, row * CELL_H)
    )

    mapping.append(
        f"{i}\t{path.name}"
    )

canvas.save(
    OUT,
    quality=95
)

MAP.write_text(
    "\n".join(mapping),
    encoding="utf-8"
)

print("✅ Mosaïque :", OUT)
print("✅ Correspondance :", MAP)
print("Nombre de candidats :", len(files))
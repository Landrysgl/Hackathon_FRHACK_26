from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

ANNOTATIONS = DATA / "niveau2_manual_train" / "annotations.csv"
IMAGES_DIR = DATA / "niveau2_manual_train" / "images"

OUT_DIR = DATA / "niveau2_manual_train" / "audit_visuel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ANNOTATIONS)

df["status"] = (
    df["status"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

visible = df[df["status"] == "visible"].copy()

print("Bbox à auditer :", len(visible))

# ============================================================
# PARAMETRES PLANCHES
# ============================================================

THUMB = 320
HEADER = 55

COLS = 3
ROWS = 3

PER_PAGE = COLS * ROWS

font = ImageFont.load_default()


# ============================================================
# CREATION DES PLANCHES
# ============================================================

pages = math.ceil(len(visible) / PER_PAGE)

for page_idx in range(pages):

    chunk = visible.iloc[
        page_idx * PER_PAGE:
        (page_idx + 1) * PER_PAGE
    ]

    canvas = Image.new(
        "RGB",
        (COLS * THUMB, ROWS * (THUMB + HEADER)),
        "white"
    )

    draw_canvas = ImageDraw.Draw(canvas)

    for j, (_, row) in enumerate(chunk.iterrows()):

        r = j // COLS
        c = j % COLS

        x0 = c * THUMB
        y0 = r * (THUMB + HEADER)

        img_path = IMAGES_DIR / str(row["image"])

        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size

        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])

        # Dessin sur image originale
        draw = ImageDraw.Draw(img)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=5
        )

        # Redimensionnement
        img.thumbnail((THUMB, THUMB))

        # Fond noir pour centrer si nécessaire
        cell = Image.new(
            "RGB",
            (THUMB, THUMB),
            "black"
        )

        paste_x = (THUMB - img.width) // 2
        paste_y = (THUMB - img.height) // 2

        cell.paste(
            img,
            (paste_x, paste_y)
        )

        canvas.paste(
            cell,
            (x0, y0)
        )

        # Informations
        sup_id = row["SUP_ID"]
        nature = str(row["NAT_LB_NOM"])
        classe = str(row.get("classe_niveau2", ""))

        text1 = f"SUP_ID: {sup_id} | {classe}"
        text2 = nature[:48]

        draw_canvas.text(
            (x0 + 5, y0 + THUMB + 5),
            text1,
            fill="black",
            font=font
        )

        draw_canvas.text(
            (x0 + 5, y0 + THUMB + 25),
            text2,
            fill="black",
            font=font
        )

    out = OUT_DIR / f"planche_{page_idx + 1:02d}.jpg"

    canvas.save(
        out,
        quality=92
    )

    print("Créée :", out)


# ============================================================
# BBOX EXTREMES
# ============================================================

visible["width_px"] = visible["x2"] - visible["x1"]
visible["height_px"] = visible["y2"] - visible["y1"]

visible["area_pct"] = (
    visible["width_px"]
    * visible["height_px"]
    / (1024 * 1024)
    * 100
)

extremes_small = visible.nsmallest(
    8,
    "area_pct"
)

extremes_large = visible.nlargest(
    8,
    "area_pct"
)

cols = [
    "SUP_ID",
    "NAT_LB_NOM",
    "classe_niveau2",
    "image",
    "width_px",
    "height_px",
    "area_pct",
]

extremes_small[cols].to_csv(
    OUT_DIR / "bbox_plus_petites.csv",
    index=False
)

extremes_large[cols].to_csv(
    OUT_DIR / "bbox_plus_grandes.csv",
    index=False
)

print("\n===================================")
print("AUDIT VISUEL PREPARE")
print("===================================")

print("Planches :", pages)
print("Dossier :", OUT_DIR)

print("\n8 plus petites bbox :")
print(
    extremes_small[
        ["SUP_ID", "NAT_LB_NOM", "area_pct"]
    ].to_string(index=False)
)

print("\n8 plus grandes bbox :")
print(
    extremes_large[
        ["SUP_ID", "NAT_LB_NOM", "area_pct"]
    ].to_string(index=False)
)
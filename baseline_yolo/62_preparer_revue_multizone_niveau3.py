from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MULTI_DIR = (
    DATA_DIR
    / "niveau3_multizone"
)

CANDIDATES_PATH = (
    MULTI_DIR
    / "candidats"
    / "candidats_uniques.csv"
)

ZONES_DIR = (
    MULTI_DIR
    / "zones"
)

OUT_DIR = (
    MULTI_DIR
    / "candidats"
    / "revue"
)

CROPS_DIR = (
    OUT_DIR
    / "crops"
)

CONTEXTS_DIR = (
    OUT_DIR
    / "contextes"
)

TO_REVIEW_CSV = (
    OUT_DIR
    / "a_examiner.csv"
)


CONTEXT_MARGIN = 140


# ============================================================
# CHARGEMENT
# ============================================================

if not CANDIDATES_PATH.exists():
    raise FileNotFoundError(
        CANDIDATES_PATH
    )


df = pd.read_csv(
    CANDIDATES_PATH
)


df["review_status"] = (
    df["review_status"]
    .fillna("")
    .astype("object")
)

df["review_comment"] = (
    df["review_comment"]
    .fillna("")
    .astype("object")
)


to_review = df[
    df["review_status"]
    .astype(str)
    .str.strip()
    .eq("")
].copy()


print("====================================")
print("PREPARATION REVUE MULTIZONE")
print("====================================")

print(
    "Candidats totaux :",
    len(df)
)

print(
    "Déjà examinés :",
    len(df) - len(to_review)
)

print(
    "À examiner :",
    len(to_review)
)

print("\nPar zone :")
print(
    to_review[
        "zone_id"
    ].value_counts()
)


if len(to_review) != 25:
    print(
        "\n⚠️ On attend normalement 25 candidats."
    )


# ============================================================
# DOSSIERS
# ============================================================

CROPS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONTEXTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Nettoyage d'anciens visuels éventuels.
for folder in [
    CROPS_DIR,
    CONTEXTS_DIR,
]:

    for p in folder.glob(
        "*.jpg"
    ):
        p.unlink()


# ============================================================
# GENERATION DES VISUELS
# ============================================================

for _, row in to_review.iterrows():

    uid = str(
        row["unique_candidate_id"]
    )

    zone_id = str(
        row["zone_id"]
    )

    image_name = str(
        row["image"]
    )


    image_path = (
        ZONES_DIR
        / zone_id
        / "images"
        / image_name
    )


    if not image_path.exists():

        raise FileNotFoundError(
            image_path
        )


    with Image.open(
        image_path
    ) as im:

        image = im.convert(
            "RGB"
        )


    width, height = image.size


    x1 = int(
        round(
            float(row["x1"])
        )
    )

    y1 = int(
        round(
            float(row["y1"])
        )
    )

    x2 = int(
        round(
            float(row["x2"])
        )
    )

    y2 = int(
        round(
            float(row["y2"])
        )
    )


    # ----------------------------------------
    # CROP SERRE
    # ----------------------------------------

    crop_x1 = max(
        0,
        x1
    )

    crop_y1 = max(
        0,
        y1
    )

    crop_x2 = min(
        width,
        x2
    )

    crop_y2 = min(
        height,
        y2
    )


    crop = image.crop(
        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        )
    )


    crop.save(
        CROPS_DIR
        / f"{uid}.jpg",
        quality=95,
    )


    # ----------------------------------------
    # CONTEXTE
    # ----------------------------------------

    ctx_x1 = max(
        0,
        x1 - CONTEXT_MARGIN
    )

    ctx_y1 = max(
        0,
        y1 - CONTEXT_MARGIN
    )

    ctx_x2 = min(
        width,
        x2 + CONTEXT_MARGIN
    )

    ctx_y2 = min(
        height,
        y2 + CONTEXT_MARGIN
    )


    context = image.crop(
        (
            ctx_x1,
            ctx_y1,
            ctx_x2,
            ctx_y2,
        )
    )


    draw = ImageDraw.Draw(
        context
    )


    draw.rectangle(
        (
            x1 - ctx_x1,
            y1 - ctx_y1,
            x2 - ctx_x1,
            y2 - ctx_y1,
        ),
        outline="red",
        width=4,
    )


    label = (
        f"{uid} | "
        f"{zone_id} | "
        f"conf={float(row['confidence']):.3f} | "
        f"ANFR="
        f"{float(row['nearest_ANFR_distance_m']):.1f}m"
    )


    draw.rectangle(
        (
            0,
            0,
            min(
                context.width,
                620
            ),
            30,
        ),
        fill="black",
    )


    draw.text(
        (
            5,
            6,
        ),
        label,
        fill="white",
    )


    context.save(
        CONTEXTS_DIR
        / f"{uid}.jpg",
        quality=95,
    )


# ============================================================
# CSV DE REVUE
# ============================================================

to_review.to_csv(
    TO_REVIEW_CSV,
    index=False,
)


# ============================================================
# AUDIT
# ============================================================

n_crops = len(
    list(
        CROPS_DIR.glob(
            "*.jpg"
        )
    )
)

n_contexts = len(
    list(
        CONTEXTS_DIR.glob(
            "*.jpg"
        )
    )
)


print("\n====================================")
print("REVUE MULTIZONE PRETE")
print("====================================")

print(
    "Candidats à examiner :",
    len(to_review)
)

print(
    "Crops :",
    n_crops
)

print(
    "Contextes :",
    n_contexts
)


if (
    n_crops != len(to_review)
    or
    n_contexts != len(to_review)
):

    raise RuntimeError(
        "Nombre de visuels incorrect."
    )


print(
    "\nCSV :",
    TO_REVIEW_CSV
)

print(
    "Crops :",
    CROPS_DIR
)

print(
    "Contextes :",
    CONTEXTS_DIR
)

print(
    "\n✅ Uniquement les candidats non examinés."
)
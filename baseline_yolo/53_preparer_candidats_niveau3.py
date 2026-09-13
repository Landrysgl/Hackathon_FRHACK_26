from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PILOT_DIR = (
    DATA_DIR
    / "niveau3_pilote"
)

DETECTIONS_PATH = (
    PILOT_DIR
    / "detections"
    / "detections_geolocalisees.csv"
)

IMAGES_DIR = (
    PILOT_DIR
    / "images"
)

OUT_DIR = (
    PILOT_DIR
    / "candidats"
)

CROPS_DIR = (
    OUT_DIR
    / "crops"
)

CONTEXT_DIR = (
    OUT_DIR
    / "contextes"
)

OUT_CSV = (
    OUT_DIR
    / "candidats.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

DISTANCE_MIN_M = 150.0

# On ne filtre PAS sur la confiance :
# même une détection peu confiante mais très éloignée
# peut être intéressante à examiner.
CONTEXT_MARGIN = 120


# ============================================================
# VERIFICATIONS
# ============================================================

if not DETECTIONS_PATH.exists():
    raise FileNotFoundError(
        DETECTIONS_PATH
    )

if not IMAGES_DIR.exists():
    raise FileNotFoundError(
        IMAGES_DIR
    )


CROPS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONTEXT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHARGEMENT
# ============================================================

detections = pd.read_csv(
    DETECTIONS_PATH
)


print("====================================")
print("PREPARATION CANDIDATS NIVEAU 3")
print("====================================")

print(
    "Détections totales :",
    len(detections)
)


# ============================================================
# FILTRE DISTANCE
# ============================================================

candidates = detections[
    detections[
        "nearest_ANFR_distance_m"
    ] >= DISTANCE_MIN_M
].copy()


# Priorité :
# 1. distance ANFR élevée
# 2. confiance élevée
candidates = candidates.sort_values(
    [
        "nearest_ANFR_distance_m",
        "confidence",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(
    drop=True
)


candidates.insert(
    0,
    "candidate_id",
    [
        f"C{i:03d}"
        for i in range(
            1,
            len(candidates) + 1
        )
    ],
)


# Colonnes pour revue humaine
candidates[
    "review_status"
] = ""

candidates[
    "review_comment"
] = ""


print(
    "Distance minimale :",
    DISTANCE_MIN_M,
    "m"
)

print(
    "Candidats retenus :",
    len(candidates)
)


# ============================================================
# CREATION DES IMAGES DE REVUE
# ============================================================

for _, row in candidates.iterrows():

    candidate_id = str(
        row["candidate_id"]
    )

    image_name = str(
        row["image"]
    )

    image_path = (
        IMAGES_DIR
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
        / f"{candidate_id}.jpg",
        quality=95,
    )


    # ----------------------------------------
    # CONTEXTE AUTOUR DE LA DETECTION
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


    local_x1 = (
        x1 - ctx_x1
    )

    local_y1 = (
        y1 - ctx_y1
    )

    local_x2 = (
        x2 - ctx_x1
    )

    local_y2 = (
        y2 - ctx_y1
    )


    draw.rectangle(
        (
            local_x1,
            local_y1,
            local_x2,
            local_y2,
        ),
        outline="red",
        width=4,
    )


    text = (
        f"{candidate_id} | "
        f"conf={float(row['confidence']):.3f} | "
        f"ANFR={float(row['nearest_ANFR_distance_m']):.1f}m"
    )


    draw.rectangle(
        (
            0,
            0,
            min(
                context.width,
                520
            ),
            28,
        ),
        fill="black",
    )


    draw.text(
        (
            5,
            5,
        ),
        text,
        fill="white",
    )


    context.save(
        CONTEXT_DIR
        / f"{candidate_id}.jpg",
        quality=95,
    )


# ============================================================
# SAUVEGARDE CSV
# ============================================================

candidates.to_csv(
    OUT_CSV,
    index=False
)


# ============================================================
# AUDIT FINAL
# ============================================================

print("\n====================================")
print("CANDIDATS NIVEAU 3 PRETS")
print("====================================")

print(
    "Candidats :",
    len(candidates)
)

print(
    "Crops :",
    len(
        list(
            CROPS_DIR.glob("*.jpg")
        )
    )
)

print(
    "Contextes :",
    len(
        list(
            CONTEXT_DIR.glob("*.jpg")
        )
    )
)


print(
    "\n===== REPARTITION CONFIANCE ====="
)

print(
    candidates[
        "confidence"
    ].describe()
)


print(
    "\n===== TOP CANDIDATS ====="
)

cols = [
    "candidate_id",
    "detection_id",
    "image",
    "confidence",
    "nearest_ANFR_distance_m",
    "latitude",
    "longitude",
]

print(
    candidates[
        cols
    ]
    .head(30)
    .to_string(
        index=False
    )
)


print(
    "\nCSV :",
    OUT_CSV
)

print(
    "Crops :",
    CROPS_DIR
)

print(
    "Contextes :",
    CONTEXT_DIR
)

print(
    "\n✅ Seuil fondé sur la calibration humaine."
)

print(
    "✅ Aucun candidat déclaré automatiquement comme site réel."
)
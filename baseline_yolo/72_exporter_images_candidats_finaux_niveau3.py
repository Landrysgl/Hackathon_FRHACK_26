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
    / "resultats_finaux"
    / "candidats_finaux_images"
)

MONTAGE_PATH = (
    MULTI_DIR
    / "resultats_finaux"
    / "candidats_finaux_montage.jpg"
)


FINAL_IDS = [
    "D003",
    "D006",
    "D007",
]

MARGIN = 220
TARGET_HEIGHT = 600
SEPARATOR = 20


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


selected = df[
    df[
        "unique_candidate_id"
    ].isin(
        FINAL_IDS
    )
].copy()


if len(selected) != 3:
    raise RuntimeError(
        f"3 candidats attendus, trouvé {len(selected)}."
    )


# Ordre final D003, D006, D007
selected[
    "_order"
] = selected[
    "unique_candidate_id"
].map(
    {
        uid: i
        for i, uid
        in enumerate(
            FINAL_IDS
        )
    }
)


selected = selected.sort_values(
    "_order"
).drop(
    columns=[
        "_order"
    ]
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Nettoyer seulement les JPG de ce dossier final.
for path in OUT_DIR.glob(
    "*.jpg"
):
    path.unlink()


# ============================================================
# GENERATION DES 3 IMAGES
# ============================================================

final_images = []


for _, row in selected.iterrows():

    uid = str(
        row[
            "unique_candidate_id"
        ]
    )

    zone_id = str(
        row[
            "zone_id"
        ]
    )

    support_type = str(
        row[
            "support_type"
        ]
    )

    image_name = str(
        row[
            "image"
        ]
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


    x1 = int(
        round(
            float(
                row["x1"]
            )
        )
    )

    y1 = int(
        round(
            float(
                row["y1"]
            )
        )
    )

    x2 = int(
        round(
            float(
                row["x2"]
            )
        )
    )

    y2 = int(
        round(
            float(
                row["y2"]
            )
        )
    )


    # --------------------------------------------------------
    # CROP CONTEXTE
    # --------------------------------------------------------

    cx1 = max(
        0,
        x1 - MARGIN
    )

    cy1 = max(
        0,
        y1 - MARGIN
    )

    cx2 = min(
        image.width,
        x2 + MARGIN
    )

    cy2 = min(
        image.height,
        y2 + MARGIN
    )


    context = image.crop(
        (
            cx1,
            cy1,
            cx2,
            cy2,
        )
    )


    # --------------------------------------------------------
    # ANNOTATION
    # --------------------------------------------------------

    draw = ImageDraw.Draw(
        context
    )


    # Bbox fine pour ne pas cacher l'objet
    draw.rectangle(
        (
            x1 - cx1,
            y1 - cy1,
            x2 - cx1,
            y2 - cy1,
        ),
        outline="red",
        width=2,
    )


    title = (
        f"{uid} | "
        f"{support_type}"
    )

    info = (
        f"conf={float(row['confidence']):.3f} | "
        f"distance ANFR="
        f"{float(row['nearest_ANFR_distance_m']):.1f} m"
    )

    coords = (
        f"{float(row['latitude']):.6f}, "
        f"{float(row['longitude']):.6f}"
    )


    draw.rectangle(
        (
            0,
            0,
            context.width,
            72,
        ),
        fill="black",
    )


    draw.text(
        (
            8,
            7,
        ),
        title,
        fill="white",
    )

    draw.text(
        (
            8,
            28,
        ),
        info,
        fill="white",
    )

    draw.text(
        (
            8,
            49,
        ),
        coords,
        fill="white",
    )


    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    output_path = (
        OUT_DIR
        / f"{uid}.jpg"
    )


    context.save(
        output_path,
        quality=95,
    )


    final_images.append(
        (
            uid,
            context.copy(),
        )
    )


# ============================================================
# MONTAGE DES 3
# ============================================================

resized = []


for uid, image in final_images:

    ratio = (
        TARGET_HEIGHT
        / image.height
    )

    target_width = int(
        image.width
        * ratio
    )


    img = image.resize(
        (
            target_width,
            TARGET_HEIGHT,
        )
    )


    resized.append(
        (
            uid,
            img,
        )
    )


montage_width = (
    sum(
        img.width
        for _, img
        in resized
    )
    +
    SEPARATOR
    * (
        len(resized) - 1
    )
)


montage = Image.new(
    "RGB",
    (
        montage_width,
        TARGET_HEIGHT,
    ),
    "white",
)


x = 0


for uid, img in resized:

    montage.paste(
        img,
        (
            x,
            0,
        )
    )

    x += (
        img.width
        + SEPARATOR
    )


montage.save(
    MONTAGE_PATH,
    quality=95,
)


# ============================================================
# SORTIE
# ============================================================

print("====================================")
print("IMAGES CANDIDATS FINAUX")
print("====================================")

for uid in FINAL_IDS:

    print(
        uid,
        "->",
        OUT_DIR / f"{uid}.jpg"
    )


print(
    "\nMontage :",
    MONTAGE_PATH
)

print(
    "\n✅ D003, D006 et D007 exportés."
)

print(
    "✅ Une image propre par candidat."
)

print(
    "✅ Bbox fine pour ne pas masquer l'objet."
)
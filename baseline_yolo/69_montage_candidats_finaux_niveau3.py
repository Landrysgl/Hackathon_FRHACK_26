from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MULTI_DIR = DATA_DIR / "niveau3_multizone"

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
)

OUT_IMAGE = (
    OUT_DIR
    / "comparaison_candidats_finaux.jpg"
)

OUT_CSV = (
    OUT_DIR
    / "candidats_a_comparer.csv"
)


N_CANDIDATS = 8
MARGIN = 220
CELL_WIDTH = 600
CELL_HEIGHT = 500
SEPARATOR = 20


# ============================================================
# CHARGEMENT
# ============================================================

df = pd.read_csv(
    CANDIDATES_PATH
)

for col in [
    "review_status",
    "support_type",
]:

    df[col] = (
        df[col]
        .fillna("")
        .astype(str)
    )


# ============================================================
# UNIQUEMENT :
# PLAUSIBLE + TYPE DE SUPPORT DEFINI
# ============================================================

eligible = df[
    (
        df["review_status"]
        == "plausible"
    )
    &
    (
        df["support_type"]
        .str.strip()
        .ne("")
    )
    &
    (
        df["support_type"]
        != "indetermine"
    )
].copy()


print(
    "Candidats éligibles :",
    len(eligible)
)

print("\nTypes :")

print(
    eligible[
        "support_type"
    ].value_counts()
)


# ============================================================
# D003 TOUJOURS PRESENT
# ============================================================

d003 = eligible[
    eligible[
        "unique_candidate_id"
    ] == "D003"
].copy()


if len(d003) != 1:

    raise RuntimeError(
        "D003 doit être présent "
        "exactement une fois."
    )


# ============================================================
# AUTRES CANDIDATS
# ============================================================

others = eligible[
    eligible[
        "unique_candidate_id"
    ] != "D003"
].copy()


others = others.sort_values(
    [
        "confidence",
        "nearest_ANFR_distance_m",
    ],
    ascending=[
        False,
        False,
    ],
)


selected = pd.concat(
    [
        d003,
        others.head(
            N_CANDIDATS - 1
        ),
    ],
    ignore_index=True,
)


selected.to_csv(
    OUT_CSV,
    index=False,
)


print("\n====================================")
print("CANDIDATS A COMPARER")
print("====================================")

print(
    selected[
        [
            "unique_candidate_id",
            "zone_id",
            "support_type",
            "confidence",
            "nearest_ANFR_distance_m",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# CARTES VISUELLES
# ============================================================

cards = []


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


    draw = ImageDraw.Draw(
        context
    )


    # bbox fine
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
        f"{row['support_type']} | "
        f"{zone_id}"
    )

    info = (
        f"conf="
        f"{float(row['confidence']):.3f} | "
        f"ANFR="
        f"{float(row['nearest_ANFR_distance_m']):.1f} m"
    )


    draw.rectangle(
        (
            0,
            0,
            context.width,
            52,
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
            27,
        ),
        info,
        fill="white",
    )


    context.thumbnail(
        (
            CELL_WIDTH,
            CELL_HEIGHT,
        )
    )


    card = Image.new(
        "RGB",
        (
            CELL_WIDTH,
            CELL_HEIGHT,
        ),
        "white",
    )


    px = (
        CELL_WIDTH
        - context.width
    ) // 2

    py = (
        CELL_HEIGHT
        - context.height
    ) // 2


    card.paste(
        context,
        (
            px,
            py,
        )
    )


    cards.append(
        card
    )


# ============================================================
# MONTAGE 4 x 2
# ============================================================

cols = 4
rows = 2


montage_width = (
    cols * CELL_WIDTH
    +
    (cols - 1)
    * SEPARATOR
)

montage_height = (
    rows * CELL_HEIGHT
    +
    (rows - 1)
    * SEPARATOR
)


montage = Image.new(
    "RGB",
    (
        montage_width,
        montage_height,
    ),
    "white",
)


for i, card in enumerate(
    cards
):

    row = i // cols
    col = i % cols

    x = (
        col
        * (
            CELL_WIDTH
            + SEPARATOR
        )
    )

    y = (
        row
        * (
            CELL_HEIGHT
            + SEPARATOR
        )
    )


    montage.paste(
        card,
        (
            x,
            y,
        )
    )


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


montage.save(
    OUT_IMAGE,
    quality=95,
)


print(
    "\nMontage :",
    OUT_IMAGE
)

print(
    "CSV :",
    OUT_CSV
)

print(
    "\n✅ Seulement des candidats "
    "plausibles et typés."
)
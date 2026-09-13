from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MULTI_DIR = DATA_DIR / "niveau3_multizone"

DETECTIONS_PATH = (
    MULTI_DIR
    / "detections"
    / "detections_geolocalisees.csv"
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
    / "audit_avant_dedup"
)

IMAGES_DIR = (
    OUT_DIR
    / "contextes"
)

INDEX_CSV = (
    OUT_DIR
    / "index_detections.csv"
)


DISTANCE_MIN = 150.0
MARGIN = 170


# ============================================================
# CHARGEMENT
# ============================================================

detections = pd.read_csv(
    DETECTIONS_PATH
)

candidates = pd.read_csv(
    CANDIDATES_PATH
)


far = detections[
    detections[
        "nearest_ANFR_distance_m"
    ] >= DISTANCE_MIN
].copy()


print("====================================")
print("AUDIT AVANT DEDUPLICATION")
print("====================================")

print(
    "Détections >=150 m :",
    len(far)
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for p in IMAGES_DIR.glob("*.jpg"):
    p.unlink()


# ============================================================
# ASSOCIER DETECTION -> CANDIDAT UNIQUE
# ============================================================

mapping = {}


for _, row in candidates.iterrows():

    ids = str(
        row[
            "cluster_detection_ids"
        ]
    ).split(",")

    for det_id in ids:

        det_id = det_id.strip()

        if det_id:
            mapping[int(det_id)] = {
                "unique_candidate_id":
                    row[
                        "unique_candidate_id"
                    ],

                "review_status":
                    row[
                        "review_status"
                    ],

                "support_type":
                    row.get(
                        "support_type",
                        "",
                    ),
            }


# ============================================================
# GENERATION
# ============================================================

records = []


for _, row in far.iterrows():

    det_id = int(
        row[
            "detection_id"
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

        image = im.convert("RGB")


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


    draw.rectangle(
        (
            x1 - cx1,
            y1 - cy1,
            x2 - cx1,
            y2 - cy1,
        ),
        outline="red",
        width=5,
    )


    info = mapping.get(
        det_id,
        {}
    )


    uid = str(
        info.get(
            "unique_candidate_id",
            "?"
        )
    )

    status = str(
        info.get(
            "review_status",
            ""
        )
    )

    support_type = str(
        info.get(
            "support_type",
            ""
        )
    )


    label = (
        f"DET {det_id} | {uid} | {zone_id} | "
        f"conf={float(row['confidence']):.3f} | "
        f"ANFR={float(row['nearest_ANFR_distance_m']):.0f}m"
    )


    draw.rectangle(
        (
            0,
            0,
            context.width,
            32,
        ),
        fill="black",
    )


    draw.text(
        (
            6,
            8,
        ),
        label,
        fill="white",
    )


    filename = (
        f"{zone_id}"
        f"_DET{det_id:03d}"
        f"_{uid}.jpg"
    )


    context.save(
        IMAGES_DIR
        / filename,
        quality=95,
    )


    records.append(
        {
            "detection_id":
                det_id,

            "zone_id":
                zone_id,

            "unique_candidate_id":
                uid,

            "confidence":
                float(
                    row[
                        "confidence"
                    ]
                ),

            "distance_anfr_m":
                float(
                    row[
                        "nearest_ANFR_distance_m"
                    ]
                ),

            "review_status":
                status,

            "support_type":
                support_type,

            "image_audit":
                filename,
        }
    )


# ============================================================
# CSV
# ============================================================

index = pd.DataFrame(
    records
)


index.to_csv(
    INDEX_CSV,
    index=False,
)


print(
    "\nImages générées :",
    len(
        list(
            IMAGES_DIR.glob(
                "*.jpg"
            )
        )
    )
)

print(
    "Index :",
    INDEX_CSV
)

print(
    "Contextes :",
    IMAGES_DIR
)


print(
    "\n===== CLUSTERS MULTIPLES ====="
)


multiple = candidates[
    candidates[
        "detections_in_cluster"
    ] > 1
]


print(
    multiple[
        [
            "unique_candidate_id",
            "zone_id",
            "detections_in_cluster",
            "cluster_detection_ids",
            "review_status",
            "support_type",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n✅ Les 53 détections originales "
    "sont maintenant inspectables."
)
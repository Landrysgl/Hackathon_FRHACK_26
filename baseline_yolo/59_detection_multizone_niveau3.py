from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MULTI_DIR = (
    DATA_DIR
    / "niveau3_multizone"
)

METADATA_PATH = (
    MULTI_DIR
    / "metadata_toutes_zones.csv"
)

ANFR_PATH = (
    DATA_DIR
    / "supports_yvelines.csv"
)

WEIGHTS = (
    ROOT
    / "runs"
    / "niveau2_renforce_coco_sans_rotation"
    / "weights"
    / "best.pt"
)

OUT_DIR = (
    MULTI_DIR
    / "detections"
)

OUT_CSV = (
    OUT_DIR
    / "detections_geolocalisees.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "resume_par_zone.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

ZOOM = 19
TILE_SIZE = 256

IMGSZ = 1024

CONF = 0.05
IOU_NMS = 0.50

EARTH_RADIUS_M = 6371008.8


# ============================================================
# VERIFICATIONS
# ============================================================

for path in [
    METADATA_PATH,
    ANFR_PATH,
    WEIGHTS,
]:
    if not path.exists():
        raise FileNotFoundError(path)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHARGEMENT
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

anfr = pd.read_csv(
    ANFR_PATH
)


if len(metadata) != 75:
    raise RuntimeError(
        f"75 patches attendus, trouvé {len(metadata)}."
    )


for col in [
    "latitude",
    "longitude",
]:
    anfr[col] = pd.to_numeric(
        anfr[col],
        errors="coerce",
    )


anfr = anfr.dropna(
    subset=[
        "latitude",
        "longitude",
    ]
).reset_index(drop=True)


print("====================================")
print("NIVEAU 3 - INFERENCE MULTIZONE")
print("====================================")

print(
    "Patches :",
    len(metadata)
)

print(
    "Zones :",
    metadata["zone_id"].nunique()
)

print(
    "Catalogue ANFR :",
    len(anfr)
)

print(
    "Modèle :",
    WEIGHTS
)

print(
    "Conf :",
    CONF
)


# ============================================================
# WEB MERCATOR
# ============================================================

def world_pixel_to_latlon(
    pixel_x,
    pixel_y,
    zoom,
):

    world_size = (
        TILE_SIZE
        * (2 ** zoom)
    )

    longitude = (
        pixel_x
        / world_size
        * 360.0
        - 180.0
    )

    n = (
        math.pi
        - 2.0
        * math.pi
        * pixel_y
        / world_size
    )

    latitude = math.degrees(
        math.atan(
            math.sinh(n)
        )
    )

    return (
        latitude,
        longitude,
    )


# ============================================================
# ANFR LE PLUS PROCHE
# ============================================================

anfr_lat_rad = np.radians(
    anfr["latitude"].to_numpy()
)

anfr_lon_rad = np.radians(
    anfr["longitude"].to_numpy()
)


def nearest_anfr(
    latitude,
    longitude,
):

    lat1 = math.radians(
        latitude
    )

    lon1 = math.radians(
        longitude
    )

    dlat = (
        anfr_lat_rad
        - lat1
    )

    dlon = (
        anfr_lon_rad
        - lon1
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        math.cos(lat1)
        * np.cos(anfr_lat_rad)
        * np.sin(dlon / 2.0) ** 2
    )

    c = (
        2.0
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(1.0 - a),
        )
    )

    distances = (
        EARTH_RADIUS_M
        * c
    )

    idx = int(
        np.argmin(
            distances
        )
    )

    return (
        idx,
        float(
            distances[idx]
        ),
    )


# ============================================================
# MODELE
# ============================================================

device = (
    0
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device :",
    device
)


model = YOLO(
    str(WEIGHTS)
)


# ============================================================
# INFERENCE
# ============================================================

detections = []

global_detection_id = 0


for i, patch in metadata.iterrows():

    zone_id = str(
        patch["zone_id"]
    )

    image_name = str(
        patch["image"]
    )

    relative_path = Path(
        str(
            patch[
                "relative_image_path"
            ]
        )
    )

    image_path = (
        MULTI_DIR
        / relative_path
    )


    if not image_path.exists():
        raise FileNotFoundError(
            image_path
        )


    print(
        f"[{i + 1:02d}/75] "
        f"{zone_id} | "
        f"{image_name}"
    )


    results = model.predict(
        source=str(image_path),
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU_NMS,
        device=device,
        verbose=False,
    )


    result = results[0]

    boxes = result.boxes


    if boxes is None:
        continue


    for box in boxes:

        global_detection_id += 1


        xyxy = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
        )


        x1, y1, x2, y2 = map(
            float,
            xyxy
        )


        confidence = float(
            box.conf[0]
            .detach()
            .cpu()
        )


        xc = (
            x1 + x2
        ) / 2.0

        yc = (
            y1 + y2
        ) / 2.0


        # ----------------------------------------
        # PIXEL PATCH -> PIXEL MONDE
        # ----------------------------------------

        world_x = (
            float(
                patch[
                    "world_left"
                ]
            )
            + xc
        )

        world_y = (
            float(
                patch[
                    "world_top"
                ]
            )
            + yc
        )


        latitude, longitude = (
            world_pixel_to_latlon(
                world_x,
                world_y,
                ZOOM,
            )
        )


        # ----------------------------------------
        # SUPPORT ANFR LE PLUS PROCHE
        # ----------------------------------------

        nearest_idx, distance_m = (
            nearest_anfr(
                latitude,
                longitude,
            )
        )


        nearest = anfr.iloc[
            nearest_idx
        ]


        detections.append(
            {
                "detection_id":
                    global_detection_id,

                "zone_id":
                    zone_id,

                "image":
                    image_name,

                "row":
                    int(
                        patch["row"]
                    ),

                "col":
                    int(
                        patch["col"]
                    ),

                "confidence":
                    confidence,

                "x1":
                    x1,

                "y1":
                    y1,

                "x2":
                    x2,

                "y2":
                    y2,

                "center_x":
                    xc,

                "center_y":
                    yc,

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "nearest_SUP_ID":
                    nearest["SUP_ID"],

                "nearest_ANFR_latitude":
                    nearest["latitude"],

                "nearest_ANFR_longitude":
                    nearest["longitude"],

                "nearest_ANFR_distance_m":
                    distance_m,

                "nearest_ANFR_nature":
                    nearest.get(
                        "NAT_LB_NOM",
                        None,
                    ),

                "nearest_ANFR_height":
                    nearest.get(
                        "SUP_NM_HAUT",
                        None,
                    ),
            }
        )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    detections
)


if len(df) == 0:
    raise RuntimeError(
        "Aucune détection sur les 75 patches."
    )


df.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# RESUME PAR ZONE
# ============================================================

summaries = []


for zone_id in [
    "dense",
    "intermediaire",
    "peu_dense",
]:

    z = df[
        df["zone_id"]
        == zone_id
    ]


    summaries.append(
        {
            "zone_id":
                zone_id,

            "patches":
                int(
                    (
                        metadata[
                            "zone_id"
                        ]
                        == zone_id
                    ).sum()
                ),

            "detections":
                len(z),

            "confidence_mean":
                (
                    float(
                        z[
                            "confidence"
                        ].mean()
                    )
                    if len(z)
                    else 0.0
                ),

            "confidence_max":
                (
                    float(
                        z[
                            "confidence"
                        ].max()
                    )
                    if len(z)
                    else 0.0
                ),

            "distance_anfr_mean_m":
                (
                    float(
                        z[
                            "nearest_ANFR_distance_m"
                        ].mean()
                    )
                    if len(z)
                    else 0.0
                ),

            "gt_50m":
                int(
                    (
                        z[
                            "nearest_ANFR_distance_m"
                        ]
                        > 50
                    ).sum()
                ),

            "gt_75m":
                int(
                    (
                        z[
                            "nearest_ANFR_distance_m"
                        ]
                        > 75
                    ).sum()
                ),

            "gt_100m":
                int(
                    (
                        z[
                            "nearest_ANFR_distance_m"
                        ]
                        > 100
                    ).sum()
                ),

            "gt_150m":
                int(
                    (
                        z[
                            "nearest_ANFR_distance_m"
                        ]
                        >= 150
                    ).sum()
                ),

            "gt_200m":
                int(
                    (
                        z[
                            "nearest_ANFR_distance_m"
                        ]
                        > 200
                    ).sum()
                ),

            "gt_300m":
                int(
                    (
                        z[
                            "nearest_ANFR_distance_m"
                        ]
                        > 300
                    ).sum()
                ),
        }
    )


summary = pd.DataFrame(
    summaries
)


summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# RESULTATS
# ============================================================

print("\n====================================")
print("RESULTATS MULTIZONE NIVEAU 3")
print("====================================")

print(
    "Détections totales :",
    len(df)
)


print("\n===== PAR ZONE =====")

print(
    summary.to_string(
        index=False
    )
)


print("\n===== DISTANCE ANFR GLOBALE =====")

print(
    df[
        "nearest_ANFR_distance_m"
    ].describe()
)


print("\n===== CANDIDATS >= 150 m =====")

far = df[
    df[
        "nearest_ANFR_distance_m"
    ] >= 150
].copy()


print(
    "Total :",
    len(far)
)


print(
    "\nPar zone :"
)

print(
    far[
        "zone_id"
    ].value_counts()
)


print(
    "\n===== TOP 20 DISTANCE ====="
)

cols = [
    "detection_id",
    "zone_id",
    "image",
    "confidence",
    "latitude",
    "longitude",
    "nearest_SUP_ID",
    "nearest_ANFR_distance_m",
]


print(
    df.sort_values(
        [
            "nearest_ANFR_distance_m",
            "confidence",
        ],
        ascending=[
            False,
            False,
        ],
    )[
        cols
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nCSV :",
    OUT_CSV
)

print(
    "Résumé :",
    SUMMARY_PATH
)


print(
    "\n✅ 75 patches analysés."
)

print(
    "✅ Résultats séparés par zone."
)

print(
    "✅ Distance ANFR calculée sur le catalogue complet."
)
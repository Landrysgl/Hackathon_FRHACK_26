from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PILOT_DIR = (
    DATA_DIR
    / "niveau3_pilote"
)

IMAGES_DIR = (
    PILOT_DIR
    / "images"
)

METADATA_PATH = (
    PILOT_DIR
    / "metadata_patches.csv"
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
    PILOT_DIR
    / "detections"
)

OUT_CSV = (
    OUT_DIR
    / "detections_geolocalisees.csv"
)

ANNOTATED_DIR = (
    OUT_DIR
    / "images_annotees"
)


# ============================================================
# CONFIGURATION
# ============================================================

ZOOM = 19
TILE_SIZE = 256

IMGSZ = 1024

# volontairement bas pour la recherche de candidats
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


if not IMAGES_DIR.exists():
    raise FileNotFoundError(IMAGES_DIR)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANNOTATED_DIR.mkdir(
    parents=True,
    exist_ok=True
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


anfr["latitude"] = pd.to_numeric(
    anfr["latitude"],
    errors="coerce"
)

anfr["longitude"] = pd.to_numeric(
    anfr["longitude"],
    errors="coerce"
)


anfr = anfr.dropna(
    subset=[
        "latitude",
        "longitude",
    ]
).reset_index(drop=True)


print("====================================")
print("NIVEAU 3 - DETECTION GEOLOCALISEE")
print("====================================")

print(
    "Patches :",
    len(metadata)
)

print(
    "Supports ANFR référence :",
    len(anfr)
)

print(
    "Poids :",
    WEIGHTS
)

print(
    "Seuil confiance :",
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
# DISTANCE HAVERSINE VERS TOUS LES SUPPORTS
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

detection_id = 0


for i, patch in metadata.iterrows():

    image_name = str(
        patch["image"]
    )

    image_path = (
        IMAGES_DIR
        / image_name
    )


    if not image_path.exists():
        raise FileNotFoundError(
            image_path
        )


    print(
        f"[{i + 1:02d}/{len(metadata)}] "
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


    # Sauvegarde visuelle
    result.save(
        filename=str(
            ANNOTATED_DIR
            / image_name
        )
    )


    boxes = result.boxes


    if boxes is None:
        continue


    for box in boxes:

        detection_id += 1


        xyxy = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
        )


        x1, y1, x2, y2 = (
            map(
                float,
                xyxy
            )
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
        # PIXEL LOCAL -> PIXEL MONDE
        # ----------------------------------------

        world_x = (
            float(
                patch["world_left"]
            )
            + xc
        )

        world_y = (
            float(
                patch["world_top"]
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
        # ANFR LE PLUS PROCHE
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
                    detection_id,

                "image":
                    image_name,

                "row":
                    int(patch["row"]),

                "col":
                    int(patch["col"]),

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
# SAUVEGARDE
# ============================================================

df = pd.DataFrame(
    detections
)


if len(df) > 0:

    df = df.sort_values(
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


df.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# AUDIT
# ============================================================

print("\n====================================")
print("RESULTATS PILOTE NIVEAU 3")
print("====================================")

print(
    "Détections totales :",
    len(df)
)


if len(df) > 0:

    print("\n===== CONFIANCE =====")

    print(
        df["confidence"]
        .describe()
    )


    print("\n===== DISTANCE ANFR =====")

    print(
        df[
            "nearest_ANFR_distance_m"
        ]
        .describe()
    )


    print("\n===== NOMBRE PAR DISTANCE =====")

    for threshold in [
        50,
        100,
        150,
        200,
        300,
    ]:

        count = int(
            (
                df[
                    "nearest_ANFR_distance_m"
                ]
                > threshold
            ).sum()
        )

        print(
            f"> {threshold:3d} m :",
            count
        )


    print(
        "\n===== TOP 15 PLUS ELOIGNEES ANFR ====="
    )

    cols = [
        "detection_id",
        "image",
        "confidence",
        "latitude",
        "longitude",
        "nearest_SUP_ID",
        "nearest_ANFR_distance_m",
    ]

    print(
        df[
            cols
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


print(
    "\nCSV :",
    OUT_CSV
)

print(
    "Images annotées :",
    ANNOTATED_DIR
)

print(
    "\n✅ Chaque détection possède des coordonnées GPS."
)

print(
    "✅ Distance calculée vers le catalogue ANFR complet."
)

print(
    "✅ Aucun seuil de candidat définitif appliqué pour l'instant."
)
from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

ANNOTATION_FILES = [
    DATA_DIR
    / "niveau2_manual_train"
    / "annotations.csv",

    DATA_DIR
    / "niveau2_manual_train_lot2"
    / "annotations.csv",

    DATA_DIR
    / "niveau2_manual_val"
    / "annotations.csv",
]

ANFR_PATH = (
    DATA_DIR
    / "supports_yvelines.csv"
)

OUT_PATH = (
    DATA_DIR
    / "niveau3_pilote"
    / "calibration_distance_anfr.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

ZOOM = 19
TILE_SIZE = 256
IMAGE_SIZE = 1024

EARTH_RADIUS_M = 6371008.8


# ============================================================
# VERIFICATIONS
# ============================================================

for path in ANNOTATION_FILES:
    if not path.exists():
        raise FileNotFoundError(path)

if not ANFR_PATH.exists():
    raise FileNotFoundError(ANFR_PATH)


# ============================================================
# CHARGEMENT ANNOTATIONS
# ============================================================

frames = []

for path in ANNOTATION_FILES:

    df = pd.read_csv(path)

    df["source"] = path.parent.name

    frames.append(df)


annotations = pd.concat(
    frames,
    ignore_index=True
)


visible = annotations[
    annotations["status"] == "visible"
].copy()


print("====================================")
print("CALIBRATION DISTANCE ANFR - NIVEAU 3")
print("====================================")

print(
    "Annotations totales :",
    len(annotations)
)

print(
    "Supports visibles utilisés :",
    len(visible)
)


# On attend :
# train lot1 : 51
# train lot2 : 93
# val        : 31
# total      : 175

if len(visible) != 175:

    print(
        "\n⚠️ Nombre attendu normalement : 175"
    )


# ============================================================
# VALIDATION COLONNES
# ============================================================

required = [
    "SUP_ID",
    "latitude",
    "longitude",
    "x1",
    "y1",
    "x2",
    "y2",
    "classe_niveau2",
]

for col in required:

    if col not in visible.columns:
        raise RuntimeError(
            f"Colonne absente : {col}"
        )


for col in [
    "latitude",
    "longitude",
    "x1",
    "y1",
    "x2",
    "y2",
]:

    visible[col] = pd.to_numeric(
        visible[col],
        errors="coerce"
    )


if visible[
    [
        "latitude",
        "longitude",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
].isna().any().any():

    raise RuntimeError(
        "Valeurs géographiques ou bbox manquantes."
    )


# ============================================================
# CHARGEMENT CATALOGUE ANFR COMPLET
# ============================================================

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


print(
    "Catalogue ANFR :",
    len(anfr)
)


# ============================================================
# WEB MERCATOR
# ============================================================

def latlon_to_world_pixel(
    latitude,
    longitude,
    zoom,
):

    latitude_rad = math.radians(
        latitude
    )

    world_size = (
        TILE_SIZE
        * (2 ** zoom)
    )

    pixel_x = (
        longitude + 180.0
    ) / 360.0 * world_size

    pixel_y = (
        (
            1.0
            - math.asinh(
                math.tan(
                    latitude_rad
                )
            )
            / math.pi
        )
        / 2.0
        * world_size
    )

    return (
        pixel_x,
        pixel_y,
    )


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
# DISTANCES
# ============================================================

def haversine_scalar(
    lat1,
    lon1,
    lat2,
    lon2,
):

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2.0) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2.0) ** 2
    )

    c = (
        2.0
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1.0 - a),
        )
    )

    return EARTH_RADIUS_M * c


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
        np.argmin(distances)
    )

    return (
        idx,
        float(
            distances[idx]
        ),
    )


# ============================================================
# GEOLOCALISATION DU CENTRE DES BBOX HUMAINES
# ============================================================

rows = []


for _, row in visible.iterrows():

    official_lat = float(
        row["latitude"]
    )

    official_lon = float(
        row["longitude"]
    )


    bbox_cx = (
        float(row["x1"])
        + float(row["x2"])
    ) / 2.0

    bbox_cy = (
        float(row["y1"])
        + float(row["y2"])
    ) / 2.0


    official_px_x, official_px_y = (
        latlon_to_world_pixel(
            official_lat,
            official_lon,
            ZOOM,
        )
    )


    # Les images corrigées Niveau 1
    # sont centrées sur la coordonnée ANFR.
    visual_px_x = (
        official_px_x
        + bbox_cx
        - IMAGE_SIZE / 2.0
    )

    visual_px_y = (
        official_px_y
        + bbox_cy
        - IMAGE_SIZE / 2.0
    )


    visual_lat, visual_lon = (
        world_pixel_to_latlon(
            visual_px_x,
            visual_px_y,
            ZOOM,
        )
    )


    distance_own = (
        haversine_scalar(
            official_lat,
            official_lon,
            visual_lat,
            visual_lon,
        )
    )


    nearest_idx, nearest_distance = (
        nearest_anfr(
            visual_lat,
            visual_lon,
        )
    )


    nearest = anfr.iloc[
        nearest_idx
    ]


    rows.append(
        {
            "SUP_ID":
                row["SUP_ID"],

            "source":
                row["source"],

            "classe_niveau2":
                row["classe_niveau2"],

            "official_latitude":
                official_lat,

            "official_longitude":
                official_lon,

            "bbox_center_x":
                bbox_cx,

            "bbox_center_y":
                bbox_cy,

            "visual_latitude":
                visual_lat,

            "visual_longitude":
                visual_lon,

            "distance_to_own_anfr_m":
                distance_own,

            "nearest_SUP_ID":
                nearest["SUP_ID"],

            "distance_to_nearest_anfr_m":
                nearest_distance,
        }
    )


result = pd.DataFrame(
    rows
)


result.to_csv(
    OUT_PATH,
    index=False
)


# ============================================================
# STATISTIQUES
# ============================================================

distance = result[
    "distance_to_nearest_anfr_m"
]


print("\n====================================")
print("DISTANCE VISUELLE -> ANFR LE PLUS PROCHE")
print("====================================")


print(
    distance.describe()
)


print("\n===== PERCENTILES =====")

for p in [
    50,
    75,
    90,
    95,
    97.5,
    99,
    100,
]:

    value = float(
        np.percentile(
            distance,
            p
        )
    )

    print(
        f"P{p:>4} : "
        f"{value:.2f} m"
    )


print("\n===== PAR CLASSE =====")

for cls, group in result.groupby(
    "classe_niveau2"
):

    d = group[
        "distance_to_nearest_anfr_m"
    ]

    print(
        f"\n{cls} "
        f"(n={len(group)})"
    )

    print(
        f"  médiane : "
        f"{d.median():.2f} m"
    )

    print(
        f"  P90     : "
        f"{np.percentile(d, 90):.2f} m"
    )

    print(
        f"  P95     : "
        f"{np.percentile(d, 95):.2f} m"
    )

    print(
        f"  max     : "
        f"{d.max():.2f} m"
    )


print("\n===== DEPASSEMENTS =====")

for threshold in [
    25,
    50,
    75,
    100,
    125,
    150,
    200,
]:

    n = int(
        (
            distance
            > threshold
        ).sum()
    )

    pct = (
        100.0
        * n
        / len(distance)
    )

    print(
        f"> {threshold:3d} m : "
        f"{n:3d} "
        f"({pct:.1f} %)"
    )


print(
    "\nCSV :",
    OUT_PATH
)

print(
    "\n✅ Calibration basée uniquement sur TRAIN + VAL humains."
)

print(
    "✅ Aucun holdout final utilisé."
)
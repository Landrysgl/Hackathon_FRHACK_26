from pathlib import Path
from io import BytesIO
import math
import shutil
import time

import numpy as np
import pandas as pd
import requests
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SUPPORTS_PATH = (
    DATA_DIR
    / "supports_yvelines.csv"
)

OUT_DIR = (
    DATA_DIR
    / "niveau3_pilote"
)

IMAGES_DIR = OUT_DIR / "images"

METADATA_PATH = (
    OUT_DIR
    / "metadata_patches.csv"
)

ANFR_ZONE_PATH = (
    OUT_DIR
    / "anfr_dans_zone.csv"
)

TILE_CACHE_DIR = (
    DATA_DIR
    / "cache_ign_tiles"
)


# ============================================================
# CONFIGURATION
# ============================================================

WMTS_URL = "https://data.geopf.fr/wmts"
LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"

ZOOM = 19
WMTS_TILE_SIZE = 256

PATCH_SIZE = 1024

# 5 x 5 patches ≈ 1 km x 1 km à cette latitude.
GRID_SIZE = 5

# Sert seulement à choisir une zone pilote urbaine/dense.
DENSITY_RADIUS_M = 500

DELAI = 0.03

EARTH_RADIUS_M = 6378137.0


# ============================================================
# PREPARATION
# ============================================================

if not SUPPORTS_PATH.exists():
    raise FileNotFoundError(
        SUPPORTS_PATH
    )


if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)


IMAGES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TILE_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHARGEMENT ANFR
# ============================================================

supports = pd.read_csv(
    SUPPORTS_PATH
)


required = [
    "SUP_ID",
    "latitude",
    "longitude",
]

for col in required:

    if col not in supports.columns:
        raise RuntimeError(
            f"Colonne absente : {col}"
        )


supports["latitude"] = pd.to_numeric(
    supports["latitude"],
    errors="coerce"
)

supports["longitude"] = pd.to_numeric(
    supports["longitude"],
    errors="coerce"
)


supports = supports[
    supports["latitude"].between(
        -90,
        90
    )
    &
    supports["longitude"].between(
        -180,
        180
    )
].copy()


supports = supports.dropna(
    subset=[
        "latitude",
        "longitude",
    ]
).reset_index(drop=True)


print("====================================")
print("NIVEAU 3 - ZONE PILOTE")
print("====================================")

print(
    "Supports ANFR disponibles :",
    len(supports)
)


# ============================================================
# CHOIX AUTOMATIQUE D'UNE ZONE DENSE
# ============================================================

lat_rad = np.radians(
    supports["latitude"].to_numpy()
)

lon_rad = np.radians(
    supports["longitude"].to_numpy()
)

lat_ref = float(
    lat_rad.mean()
)


# Approximation métrique locale suffisante
# pour sélectionner notre zone pilote.
x_m = (
    EARTH_RADIUS_M
    * lon_rad
    * math.cos(lat_ref)
)

y_m = (
    EARTH_RADIUS_M
    * lat_rad
)


radius2 = (
    DENSITY_RADIUS_M
    ** 2
)


counts = np.zeros(
    len(supports),
    dtype=int
)


for i in range(
    len(supports)
):

    dx = x_m - x_m[i]
    dy = y_m - y_m[i]

    counts[i] = int(
        np.sum(
            dx * dx
            + dy * dy
            <= radius2
        )
    )


best_index = int(
    np.argmax(counts)
)

centre = supports.iloc[
    best_index
]


centre_lat = float(
    centre["latitude"]
)

centre_lon = float(
    centre["longitude"]
)


print("\n===== CENTRE PILOTE =====")

print(
    "SUP_ID :",
    centre["SUP_ID"]
)

print(
    "Latitude :",
    centre_lat
)

print(
    "Longitude :",
    centre_lon
)

print(
    f"Supports dans un rayon de "
    f"{DENSITY_RADIUS_M} m :",
    int(counts[best_index])
)


# ============================================================
# WEB MERCATOR : LAT/LON <-> PIXEL MONDE
# ============================================================

def latlon_to_world_pixel(
    latitude,
    longitude,
    zoom,
):

    lat_rad = math.radians(
        latitude
    )

    world_size = (
        WMTS_TILE_SIZE
        * (2 ** zoom)
    )

    pixel_x = (
        (longitude + 180.0)
        / 360.0
        * world_size
    )

    pixel_y = (
        (
            1.0
            - math.asinh(
                math.tan(lat_rad)
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
        WMTS_TILE_SIZE
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
# CACHE WMTS
# ============================================================

def cache_path(
    tile_x,
    tile_y,
    zoom,
):

    return (
        TILE_CACHE_DIR
        / str(zoom)
        / str(tile_x)
        / f"{tile_y}.jpg"
    )


def read_cached_tile(
    path,
):

    if not path.exists():
        return None

    try:

        with Image.open(path) as im:

            im.load()

            if im.size != (
                WMTS_TILE_SIZE,
                WMTS_TILE_SIZE,
            ):
                return None

            return im.convert("RGB")

    except Exception:

        return None


def download_tile(
    session,
    tile_x,
    tile_y,
    zoom,
):

    path = cache_path(
        tile_x,
        tile_y,
        zoom,
    )

    cached = read_cached_tile(
        path
    )

    if cached is not None:
        return cached


    params = {
        "SERVICE": "WMTS",
        "VERSION": "1.0.0",
        "REQUEST": "GetTile",
        "LAYER": LAYER,
        "STYLE": "normal",
        "FORMAT": "image/jpeg",
        "TILEMATRIXSET": "PM",
        "TILEMATRIX": zoom,
        "TILEROW": tile_y,
        "TILECOL": tile_x,
    }


    for attempt in range(
        1,
        4
    ):

        try:

            response = session.get(
                WMTS_URL,
                params=params,
                timeout=30,
            )

            response.raise_for_status()

            image = Image.open(
                BytesIO(
                    response.content
                )
            )

            image.load()
            image = image.convert(
                "RGB"
            )

            if image.size != (
                WMTS_TILE_SIZE,
                WMTS_TILE_SIZE,
            ):

                raise RuntimeError(
                    "Taille WMTS inattendue : "
                    f"{image.size}"
                )


            path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            image.save(
                path,
                format="JPEG",
                quality=95,
            )

            time.sleep(
                DELAI
            )

            return image


        except Exception as exc:

            print(
                f"Erreur tuile "
                f"{tile_x}/{tile_y} "
                f"tentative "
                f"{attempt}/3 : "
                f"{exc}"
            )

            time.sleep(2)


    raise RuntimeError(
        f"Impossible de récupérer "
        f"la tuile {tile_x}/{tile_y}"
    )


# ============================================================
# CREATION D'UN PATCH 1024x1024
# ============================================================

def create_patch(
    session,
    left,
    top,
):

    right = (
        left
        + PATCH_SIZE
    )

    bottom = (
        top
        + PATCH_SIZE
    )


    tile_x_min = math.floor(
        left
        / WMTS_TILE_SIZE
    )

    tile_x_max = math.floor(
        (right - 1)
        / WMTS_TILE_SIZE
    )

    tile_y_min = math.floor(
        top
        / WMTS_TILE_SIZE
    )

    tile_y_max = math.floor(
        (bottom - 1)
        / WMTS_TILE_SIZE
    )


    mosaic_width = (
        tile_x_max
        - tile_x_min
        + 1
    ) * WMTS_TILE_SIZE

    mosaic_height = (
        tile_y_max
        - tile_y_min
        + 1
    ) * WMTS_TILE_SIZE


    mosaic = Image.new(
        "RGB",
        (
            mosaic_width,
            mosaic_height,
        ),
    )


    for tx in range(
        tile_x_min,
        tile_x_max + 1,
    ):

        for ty in range(
            tile_y_min,
            tile_y_max + 1,
        ):

            tile = download_tile(
                session,
                tx,
                ty,
                ZOOM,
            )

            paste_x = (
                tx - tile_x_min
            ) * WMTS_TILE_SIZE

            paste_y = (
                ty - tile_y_min
            ) * WMTS_TILE_SIZE

            mosaic.paste(
                tile,
                (
                    paste_x,
                    paste_y,
                ),
            )


    origin_x = (
        tile_x_min
        * WMTS_TILE_SIZE
    )

    origin_y = (
        tile_y_min
        * WMTS_TILE_SIZE
    )


    crop_left = int(
        round(
            left - origin_x
        )
    )

    crop_top = int(
        round(
            top - origin_y
        )
    )


    patch = mosaic.crop(
        (
            crop_left,
            crop_top,
            crop_left
            + PATCH_SIZE,
            crop_top
            + PATCH_SIZE,
        )
    )


    if patch.size != (
        PATCH_SIZE,
        PATCH_SIZE,
    ):

        raise RuntimeError(
            f"Patch invalide : "
            f"{patch.size}"
        )


    return patch


# ============================================================
# EMPRISE DE LA ZONE PILOTE
# ============================================================

centre_px_x, centre_px_y = (
    latlon_to_world_pixel(
        centre_lat,
        centre_lon,
        ZOOM,
    )
)


total_pixels = (
    GRID_SIZE
    * PATCH_SIZE
)


zone_left = int(
    round(
        centre_px_x
        - total_pixels / 2
    )
)

zone_top = int(
    round(
        centre_px_y
        - total_pixels / 2
    )
)


zone_right = (
    zone_left
    + total_pixels
)

zone_bottom = (
    zone_top
    + total_pixels
)


north_lat, west_lon = (
    world_pixel_to_latlon(
        zone_left,
        zone_top,
        ZOOM,
    )
)

south_lat, east_lon = (
    world_pixel_to_latlon(
        zone_right,
        zone_bottom,
        ZOOM,
    )
)


print("\n===== EMPRISE =====")

print(
    "Nord :",
    north_lat
)

print(
    "Sud :",
    south_lat
)

print(
    "Ouest :",
    west_lon
)

print(
    "Est :",
    east_lon
)


# ============================================================
# SUPPORTS ANFR DANS LA ZONE
# ============================================================

anfr_zone = supports[
    supports["latitude"].between(
        south_lat,
        north_lat
    )
    &
    supports["longitude"].between(
        west_lon,
        east_lon
    )
].copy()


anfr_zone.to_csv(
    ANFR_ZONE_PATH,
    index=False,
)


print(
    "\nSupports ANFR dans la zone :",
    len(anfr_zone)
)


# ============================================================
# TELECHARGEMENT DES 25 PATCHES
# ============================================================

session = requests.Session()

metadata = []


print("\n===== TELECHARGEMENT =====")


for row in range(
    GRID_SIZE
):

    for col in range(
        GRID_SIZE
    ):

        patch_left = (
            zone_left
            + col * PATCH_SIZE
        )

        patch_top = (
            zone_top
            + row * PATCH_SIZE
        )

        patch_right = (
            patch_left
            + PATCH_SIZE
        )

        patch_bottom = (
            patch_top
            + PATCH_SIZE
        )


        filename = (
            f"pilot_r{row:02d}"
            f"_c{col:02d}.jpg"
        )


        output_path = (
            IMAGES_DIR
            / filename
        )


        print(
            f"[{row * GRID_SIZE + col + 1:02d}"
            f"/{GRID_SIZE * GRID_SIZE}] "
            f"{filename}"
        )


        patch = create_patch(
            session,
            patch_left,
            patch_top,
        )


        patch.save(
            output_path,
            format="JPEG",
            quality=95,
        )


        center_lat_patch, center_lon_patch = (
            world_pixel_to_latlon(
                (
                    patch_left
                    + PATCH_SIZE / 2
                ),
                (
                    patch_top
                    + PATCH_SIZE / 2
                ),
                ZOOM,
            )
        )


        north_patch, west_patch = (
            world_pixel_to_latlon(
                patch_left,
                patch_top,
                ZOOM,
            )
        )

        south_patch, east_patch = (
            world_pixel_to_latlon(
                patch_right,
                patch_bottom,
                ZOOM,
            )
        )


        metadata.append(
            {
                "image": filename,
                "row": row,
                "col": col,
                "zoom": ZOOM,

                "world_left": patch_left,
                "world_top": patch_top,
                "world_right": patch_right,
                "world_bottom": patch_bottom,

                "center_latitude":
                    center_lat_patch,

                "center_longitude":
                    center_lon_patch,

                "north":
                    north_patch,

                "south":
                    south_patch,

                "west":
                    west_patch,

                "east":
                    east_patch,
            }
        )


metadata_df = pd.DataFrame(
    metadata
)


metadata_df.to_csv(
    METADATA_PATH,
    index=False,
)


# ============================================================
# AUDIT FINAL
# ============================================================

created = list(
    IMAGES_DIR.glob(
        "*.jpg"
    )
)


print("\n====================================")
print("ZONE PILOTE NIVEAU 3 PRETE")
print("====================================")

print(
    "Grille :",
    f"{GRID_SIZE} x {GRID_SIZE}"
)

print(
    "Patches attendus :",
    GRID_SIZE * GRID_SIZE
)

print(
    "Patches créés :",
    len(created)
)

print(
    "Taille patch :",
    f"{PATCH_SIZE}x{PATCH_SIZE}"
)

print(
    "Supports ANFR zone :",
    len(anfr_zone)
)

print(
    "\nImages :",
    IMAGES_DIR
)

print(
    "Métadonnées :",
    METADATA_PATH
)

print(
    "Catalogue ANFR local :",
    ANFR_ZONE_PATH
)

if len(created) != (
    GRID_SIZE * GRID_SIZE
):
    raise RuntimeError(
        "Nombre de patches incorrect."
    )


print(
    "\n✅ Chaque patch est géoréférencé."
)

print(
    "✅ Catalogue complet Yvelines utilisé."
)

print(
    "✅ Aucun entraînement supplémentaire."
)
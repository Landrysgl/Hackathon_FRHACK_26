from pathlib import Path
from io import BytesIO
import math
import shutil
import time

import pandas as pd
import requests
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MULTI_DIR = (
    DATA_DIR
    / "niveau3_multizone"
)

ZONES_PATH = (
    MULTI_DIR
    / "zones_selectionnees.csv"
)

SUPPORTS_PATH = (
    DATA_DIR
    / "supports_yvelines.csv"
)

ZONES_ROOT = (
    MULTI_DIR
    / "zones"
)

ALL_METADATA_PATH = (
    MULTI_DIR
    / "metadata_toutes_zones.csv"
)

SUMMARY_PATH = (
    MULTI_DIR
    / "resume_zones.csv"
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

GRID_SIZE = 5

DELAI = 0.03


# ============================================================
# VERIFICATIONS
# ============================================================

for path in [
    ZONES_PATH,
    SUPPORTS_PATH,
]:

    if not path.exists():
        raise FileNotFoundError(path)


zones = pd.read_csv(
    ZONES_PATH
)

supports = pd.read_csv(
    SUPPORTS_PATH
)


if len(zones) != 3:
    raise RuntimeError(
        f"3 zones attendues, trouvé {len(zones)}."
    )


for col in [
    "latitude",
    "longitude",
]:

    supports[col] = pd.to_numeric(
        supports[col],
        errors="coerce",
    )


supports = supports.dropna(
    subset=[
        "latitude",
        "longitude",
    ]
).reset_index(drop=True)


# Ne supprimer que les zones générées,
# pas zones_selectionnees.csv.
if ZONES_ROOT.exists():
    shutil.rmtree(
        ZONES_ROOT
    )


ZONES_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

TILE_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("====================================")
print("NIVEAU 3 - PREPARATION MULTIZONE")
print("====================================")

print(
    "Zones :",
    len(zones)
)

print(
    "Catalogue ANFR :",
    len(supports)
)

print(
    "Patches par zone :",
    GRID_SIZE * GRID_SIZE
)

print(
    "Patches totaux attendus :",
    len(zones)
    * GRID_SIZE
    * GRID_SIZE
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
        WMTS_TILE_SIZE
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
# WMTS / CACHE
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

            return im.convert(
                "RGB"
            )

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

            content_type = (
                response.headers
                .get(
                    "Content-Type",
                    ""
                )
                .lower()
            )

            if "image" not in content_type:
                raise RuntimeError(
                    "La réponse WMTS "
                    "n'est pas une image : "
                    f"{content_type}"
                )


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
                exist_ok=True,
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
                f"    tentative "
                f"{attempt}/3 "
                f"échouée "
                f"({tile_x}, {tile_y}) : "
                f"{exc}"
            )

            time.sleep(2)


    raise RuntimeError(
        f"Impossible de télécharger "
        f"la tuile "
        f"{tile_x}/{tile_y}."
    )


# ============================================================
# CREATION PATCH
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
            crop_left + PATCH_SIZE,
            crop_top + PATCH_SIZE,
        )
    )


    if patch.size != (
        PATCH_SIZE,
        PATCH_SIZE,
    ):

        raise RuntimeError(
            f"Patch incorrect : "
            f"{patch.size}"
        )


    return patch


# ============================================================
# GENERATION DES TROIS ZONES
# ============================================================

session = requests.Session()

all_metadata = []
zone_summaries = []


for zone_index, zone in zones.iterrows():

    zone_id = str(
        zone["zone_id"]
    )

    centre_lat = float(
        zone["latitude"]
    )

    centre_lon = float(
        zone["longitude"]
    )


    print(
        "\n===================================="
    )

    print(
        f"ZONE {zone_index + 1}/"
        f"{len(zones)} : "
        f"{zone_id}"
    )

    print(
        "===================================="
    )

    print(
        "Centre :",
        centre_lat,
        centre_lon
    )

    print(
        "Densité 500 m :",
        int(
            zone["supports_500m"]
        )
    )


    zone_dir = (
        ZONES_ROOT
        / zone_id
    )

    images_dir = (
        zone_dir
        / "images"
    )

    metadata_path = (
        zone_dir
        / "metadata_patches.csv"
    )

    anfr_zone_path = (
        zone_dir
        / "anfr_dans_zone.csv"
    )


    images_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # EMPRISE
    # --------------------------------------------------------

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


    print(
        "Emprise :",
        f"N={north_lat:.6f}",
        f"S={south_lat:.6f}",
        f"W={west_lon:.6f}",
        f"E={east_lon:.6f}",
    )


    # --------------------------------------------------------
    # ANFR DANS LA ZONE
    # --------------------------------------------------------

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
        anfr_zone_path,
        index=False,
    )


    print(
        "Supports ANFR dans l'emprise :",
        len(anfr_zone)
    )


    # --------------------------------------------------------
    # PATCHES
    # --------------------------------------------------------

    zone_metadata = []


    for row in range(
        GRID_SIZE
    ):

        for col in range(
            GRID_SIZE
        ):

            n = (
                row * GRID_SIZE
                + col
                + 1
            )


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
                f"{zone_id}"
                f"_r{row:02d}"
                f"_c{col:02d}.jpg"
            )


            output_path = (
                images_dir
                / filename
            )


            print(
                f"[{n:02d}/25] "
                f"{filename}"
            )


            patch_image = (
                create_patch(
                    session,
                    patch_left,
                    patch_top,
                )
            )


            patch_image.save(
                output_path,
                format="JPEG",
                quality=95,
            )


            center_lat_patch, center_lon_patch = (
                world_pixel_to_latlon(
                    patch_left
                    + PATCH_SIZE / 2,
                    patch_top
                    + PATCH_SIZE / 2,
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


            record = {
                "zone_id":
                    zone_id,

                "image":
                    filename,

                "relative_image_path":
                    str(
                        Path("zones")
                        / zone_id
                        / "images"
                        / filename
                    ),

                "row":
                    row,

                "col":
                    col,

                "zoom":
                    ZOOM,

                "world_left":
                    patch_left,

                "world_top":
                    patch_top,

                "world_right":
                    patch_right,

                "world_bottom":
                    patch_bottom,

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


            zone_metadata.append(
                record
            )

            all_metadata.append(
                record
            )


    zone_metadata_df = pd.DataFrame(
        zone_metadata
    )


    zone_metadata_df.to_csv(
        metadata_path,
        index=False,
    )


    created = len(
        list(
            images_dir.glob(
                "*.jpg"
            )
        )
    )


    if created != 25:
        raise RuntimeError(
            f"{zone_id}: "
            f"{created} patches "
            "au lieu de 25."
        )


    zone_summaries.append(
        {
            "zone_id":
                zone_id,

            "centre_SUP_ID":
                zone["SUP_ID"],

            "centre_latitude":
                centre_lat,

            "centre_longitude":
                centre_lon,

            "supports_500m":
                int(
                    zone[
                        "supports_500m"
                    ]
                ),

            "supports_dans_emprise":
                len(anfr_zone),

            "patches":
                created,

            "north":
                north_lat,

            "south":
                south_lat,

            "west":
                west_lon,

            "east":
                east_lon,
        }
    )


# ============================================================
# EXPORT GLOBAL
# ============================================================

all_metadata_df = pd.DataFrame(
    all_metadata
)

summary_df = pd.DataFrame(
    zone_summaries
)


all_metadata_df.to_csv(
    ALL_METADATA_PATH,
    index=False,
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# AUDIT
# ============================================================

print(
    "\n===================================="
)

print(
    "MULTIZONE NIVEAU 3 PRET"
)

print(
    "===================================="
)


print(
    summary_df[
        [
            "zone_id",
            "supports_500m",
            "supports_dans_emprise",
            "patches",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nPatches totaux :",
    len(all_metadata_df)
)


if len(
    all_metadata_df
) != 75:

    raise RuntimeError(
        "75 patches attendus."
    )


print(
    "\nMétadonnées globales :",
    ALL_METADATA_PATH
)

print(
    "Résumé zones :",
    SUMMARY_PATH
)

print(
    "\n✅ 3 zones préparées."
)

print(
    "✅ 75 patches géoréférencés."
)

print(
    "✅ Structure uniforme pour l'inférence."
)
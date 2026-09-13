from pathlib import Path
from io import BytesIO
import math
import time

import pandas as pd
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SUPPORTS_PATH = DATA_DIR / "supports_yvelines.csv"
OUTPUT_DIR = DATA_DIR / "test_ign"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WMTS_URL = "https://data.geopf.fr/wmts"
LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"
ZOOM = 19
TILE_SIZE = 256
IMAGE_SIZE = 1024

if not SUPPORTS_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {SUPPORTS_PATH}\n"
        "Lance d'abord 02_choix_zone.py."
    )

supports = pd.read_csv(
    SUPPORTS_PATH,
    dtype={
        "SUP_ID": "string",
        "STA_NM_ANFR": "string",
        "COM_CD_INSEE": "string",
    },
)

print("===== DONNEES =====")
print("Nombre de supports disponibles :", len(supports))

support = supports.iloc[0]
sup_id = str(support["SUP_ID"])
latitude = float(support["latitude"])
longitude = float(support["longitude"])

print("\n===== SUPPORT TESTE =====")
print("SUP_ID    :", sup_id)
print("Latitude  :", latitude)
print("Longitude :", longitude)
if "NAT_LB_NOM" in support.index:
    print("Nature    :", support["NAT_LB_NOM"])
if "SUP_NM_HAUT" in support.index:
    print("Hauteur   :", support["SUP_NM_HAUT"])


def coordonnees_vers_pixels(latitude, longitude, zoom):
    latitude_rad = math.radians(latitude)
    world_size = TILE_SIZE * (2 ** zoom)
    pixel_x = (longitude + 180.0) / 360.0 * world_size
    pixel_y = (
        (1.0 - math.asinh(math.tan(latitude_rad)) / math.pi)
        / 2.0
        * world_size
    )
    return pixel_x, pixel_y


def telecharger_tuile(session, tile_x, tile_y, zoom):
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

    for tentative in range(1, 4):
        try:
            response = session.get(WMTS_URL, params=params, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "image" not in content_type:
                raise ValueError(
                    "La réponse IGN n'est pas une image : "
                    f"{content_type}"
                )

            image = Image.open(BytesIO(response.content))
            image.load()
            return image.convert("RGB")

        except Exception as erreur:
            print(
                f"Erreur tuile ({tile_x}, {tile_y}) "
                f"- tentative {tentative}/3 : {erreur}"
            )
            time.sleep(2)

    return None


def creer_image_centree(session, latitude, longitude, zoom, output_size):
    pixel_x, pixel_y = coordonnees_vers_pixels(latitude, longitude, zoom)
    half_size = output_size / 2

    left_pixel = pixel_x - half_size
    top_pixel = pixel_y - half_size
    right_pixel = pixel_x + half_size
    bottom_pixel = pixel_y + half_size

    tile_x_min = math.floor(left_pixel / TILE_SIZE)
    tile_x_max = math.floor((right_pixel - 1) / TILE_SIZE)
    tile_y_min = math.floor(top_pixel / TILE_SIZE)
    tile_y_max = math.floor((bottom_pixel - 1) / TILE_SIZE)

    mosaic_width = (tile_x_max - tile_x_min + 1) * TILE_SIZE
    mosaic_height = (tile_y_max - tile_y_min + 1) * TILE_SIZE

    print("\n===== TUILES NECESSAIRES =====")
    print("Colonnes :", tile_x_min, "->", tile_x_max)
    print("Lignes   :", tile_y_min, "->", tile_y_max)
    print(
        "Nombre de tuiles :",
        (tile_x_max - tile_x_min + 1) * (tile_y_max - tile_y_min + 1),
    )

    mosaic = Image.new("RGB", (mosaic_width, mosaic_height))

    for tile_y in range(tile_y_min, tile_y_max + 1):
        for tile_x in range(tile_x_min, tile_x_max + 1):
            image_tile = telecharger_tuile(
                session, tile_x, tile_y, zoom
            )
            if image_tile is None:
                return None

            paste_x = (tile_x - tile_x_min) * TILE_SIZE
            paste_y = (tile_y - tile_y_min) * TILE_SIZE
            mosaic.paste(image_tile, (paste_x, paste_y))

    origin_x = tile_x_min * TILE_SIZE
    origin_y = tile_y_min * TILE_SIZE
    center_x_mosaic = pixel_x - origin_x
    center_y_mosaic = pixel_y - origin_y

    left = round(center_x_mosaic - half_size)
    top = round(center_y_mosaic - half_size)
    right = left + output_size
    bottom = top + output_size

    return mosaic.crop((left, top, right, bottom))


print("\nTéléchargement IGN...")
session = requests.Session()
session.headers.update({"User-Agent": "FRHack-ANFR-ISEP/1.0"})

image = creer_image_centree(
    session=session,
    latitude=latitude,
    longitude=longitude,
    zoom=ZOOM,
    output_size=IMAGE_SIZE,
)

if image is None:
    raise RuntimeError("Impossible de créer l'image IGN.")

if image.size != (IMAGE_SIZE, IMAGE_SIZE):
    raise ValueError(f"Dimensions incorrectes : {image.size}")

print("\n===== IMAGE =====")
print("Dimensions :", image.size)
print(
    "Point ANFR attendu au pixel :",
    (IMAGE_SIZE // 2, IMAGE_SIZE // 2),
)

image_path = OUTPUT_DIR / f"support_{sup_id}.jpg"
image.save(image_path, quality=95)
print("\nImage IGN enregistrée :", image_path)

control_image = image.copy()
draw = ImageDraw.Draw(control_image)
center = IMAGE_SIZE // 2
cross_size = 25

draw.line(
    (center - cross_size, center, center + cross_size, center),
    fill="red",
    width=5,
)
draw.line(
    (center, center - cross_size, center, center + cross_size),
    fill="red",
    width=5,
)

control_path = OUTPUT_DIR / f"support_{sup_id}_controle.jpg"
control_image.save(control_path, quality=95)
print("Image de contrôle :", control_path)

meters_per_pixel = (
    156543.03392804097
    * math.cos(math.radians(latitude))
    / (2 ** ZOOM)
)
ground_width = meters_per_pixel * IMAGE_SIZE

print("\n===== RESOLUTION APPROXIMATIVE =====")
print(f"Résolution : {meters_per_pixel:.3f} m/pixel")
print(f"Largeur au sol : {ground_width:.1f} m")
print(f"Hauteur au sol : {ground_width:.1f} m")

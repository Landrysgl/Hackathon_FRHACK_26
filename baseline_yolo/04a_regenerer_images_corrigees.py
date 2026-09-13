from pathlib import Path
from io import BytesIO
import math
import time

import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SUPPORTS_PATH = DATA_DIR / "supports_yvelines.csv"
IMAGES_DIR = DATA_DIR / "images_ign_niveau1_corrige"
SELECTION_PATH = DATA_DIR / "supports_selectionnes_niveau1.csv"
METADATA_PATH = IMAGES_DIR / "metadonnees.csv"
TILE_CACHE_DIR = DATA_DIR / "cache_ign_tiles"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

WMTS_URL = "https://data.geopf.fr/wmts"
LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"
ZOOM = 19
TILE_SIZE = 256
IMAGE_SIZE = 1024

NOMBRE_IMAGES = 300
RANDOM_SEED = 42
DELAI_ENTRE_REQUETES = 0.03

NATURES_NON_VISIBLES = {
    "Tunnel",
    "Intérieur sous-terrain",
    "Intérieur galerie",
}

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

print("===== SUPPORTS DISPONIBLES =====")
print("Nombre de supports :", len(supports))
print("Nombre de SUP_ID uniques :", supports["SUP_ID"].nunique())

if supports["SUP_ID"].duplicated().any():
    doublons = supports[supports["SUP_ID"].duplicated(keep=False)]
    raise ValueError(
        "supports_yvelines.csv contient encore des SUP_ID dupliqués.\n"
        f"Nombre de lignes concernées : {len(doublons)}"
    )

print("Doublons SUP_ID : 0")

if "NAT_LB_NOM" not in supports.columns:
    raise ValueError("La colonne NAT_LB_NOM est absente.")

masque_non_visible = supports["NAT_LB_NOM"].isin(NATURES_NON_VISIBLES)
supports_exclus = supports[masque_non_visible].copy()
supports_candidats = supports[~masque_non_visible].copy()

print("\n===== FILTRAGE VISUEL =====")
print("Supports exclus :", len(supports_exclus))
if len(supports_exclus) > 0:
    print(supports_exclus["NAT_LB_NOM"].value_counts())
print("\nSupports candidats restants :", len(supports_candidats))

supports_candidats["latitude"] = pd.to_numeric(
    supports_candidats["latitude"], errors="coerce"
)
supports_candidats["longitude"] = pd.to_numeric(
    supports_candidats["longitude"], errors="coerce"
)

coordonnees_invalides = supports_candidats[
    supports_candidats["latitude"].isna()
    | supports_candidats["longitude"].isna()
    | ~supports_candidats["latitude"].between(-90, 90)
    | ~supports_candidats["longitude"].between(-180, 180)
]

if len(coordonnees_invalides) > 0:
    raise ValueError(
        f"{len(coordonnees_invalides)} supports ont des coordonnées invalides."
    )

print("Coordonnées invalides :", len(coordonnees_invalides))

supports_candidats = (
    supports_candidats
    .sort_values("SUP_ID")
    .reset_index(drop=True)
)

if NOMBRE_IMAGES > len(supports_candidats):
    raise ValueError(
        f"Impossible de sélectionner {NOMBRE_IMAGES} supports : "
        f"seulement {len(supports_candidats)} disponibles."
    )

supports_selectionnes = (
    supports_candidats
    .sample(n=NOMBRE_IMAGES, random_state=RANDOM_SEED)
    .sort_values("SUP_ID")
    .reset_index(drop=True)
)

if supports_selectionnes["SUP_ID"].nunique() != len(supports_selectionnes):
    raise ValueError("La sélection contient des SUP_ID dupliqués.")

print("\n===== SELECTION =====")
print("Nombre de supports sélectionnés :", len(supports_selectionnes))
print("SUP_ID uniques :", supports_selectionnes["SUP_ID"].nunique())
print("\nTypes de supports sélectionnés :")
print(supports_selectionnes["NAT_LB_NOM"].value_counts())

supports_selectionnes.to_csv(SELECTION_PATH, index=False)
print("\nSélection sauvegardée :", SELECTION_PATH)

noms_attendus = {
    f"support_{sup_id}.jpg"
    for sup_id in supports_selectionnes["SUP_ID"]
}

images_supprimees = 0
for image_path in IMAGES_DIR.glob("support_*.jpg"):
    if image_path.name not in noms_attendus:
        image_path.unlink()
        images_supprimees += 1

print("Anciennes images hors sélection supprimées :", images_supprimees)


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


def cache_path(tile_x, tile_y, zoom):
    return TILE_CACHE_DIR / str(zoom) / str(tile_x) / f"{tile_y}.jpg"


def lire_tuile_cache(path):
    if not path.exists():
        return None
    try:
        with Image.open(path) as im:
            im.load()
            if im.size != (TILE_SIZE, TILE_SIZE):
                return None
            return im.convert("RGB")
    except Exception:
        return None


def telecharger_tuile(session, tile_x, tile_y, zoom):
    path_cache = cache_path(tile_x, tile_y, zoom)
    image_cache = lire_tuile_cache(path_cache)
    if image_cache is not None:
        return image_cache

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
            image = image.convert("RGB")

            if image.size != (TILE_SIZE, TILE_SIZE):
                raise ValueError(f"Taille de tuile inattendue : {image.size}")

            path_cache.parent.mkdir(parents=True, exist_ok=True)
            image.save(path_cache, format="JPEG", quality=95)
            time.sleep(DELAI_ENTRE_REQUETES)
            return image

        except Exception as erreur:
            print(
                f"      Tentative {tentative}/3 échouée "
                f"pour tuile ({tile_x}, {tile_y}) : {erreur}"
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
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height))

    for tile_y in range(tile_y_min, tile_y_max + 1):
        for tile_x in range(tile_x_min, tile_x_max + 1):
            tile_image = telecharger_tuile(
                session, tile_x, tile_y, zoom
            )
            if tile_image is None:
                return None

            paste_x = (tile_x - tile_x_min) * TILE_SIZE
            paste_y = (tile_y - tile_y_min) * TILE_SIZE
            mosaic.paste(tile_image, (paste_x, paste_y))

    origin_x = tile_x_min * TILE_SIZE
    origin_y = tile_y_min * TILE_SIZE
    center_x_mosaic = pixel_x - origin_x
    center_y_mosaic = pixel_y - origin_y

    left = round(center_x_mosaic - half_size)
    top = round(center_y_mosaic - half_size)
    right = left + output_size
    bottom = top + output_size

    return mosaic.crop((left, top, right, bottom))


def image_existante_valide(image_path):
    if not image_path.exists():
        return False
    try:
        with Image.open(image_path) as image:
            image.load()
            return image.size == (IMAGE_SIZE, IMAGE_SIZE)
    except Exception:
        return False


session = requests.Session()
session.headers.update({"User-Agent": "FRHack-ANFR-ISEP/1.0"})

resultats = []
nombre_ok = 0
nombre_existantes = 0
nombre_echecs = 0

for numero, support in enumerate(
    supports_selectionnes.itertuples(index=False),
    start=1,
):
    sup_id = str(support.SUP_ID)
    latitude = float(support.latitude)
    longitude = float(support.longitude)
    nom_image = f"support_{sup_id}.jpg"
    image_path = IMAGES_DIR / nom_image

    print(f"[{numero}/{len(supports_selectionnes)}] SUP_ID={sup_id}")

    if image_existante_valide(image_path):
        print("    Image déjà présente et valide.")
        nombre_existantes += 1
        resultats.append({
            "SUP_ID": sup_id,
            "latitude": latitude,
            "longitude": longitude,
            "NAT_LB_NOM": support.NAT_LB_NOM,
            "image": nom_image,
            "statut": "existante",
        })
        continue

    if image_path.exists():
        print("    Image existante invalide : nouveau téléchargement.")
        image_path.unlink()

    image = creer_image_centree(
        session=session,
        latitude=latitude,
        longitude=longitude,
        zoom=ZOOM,
        output_size=IMAGE_SIZE,
    )

    if image is None:
        print("    ECHEC du téléchargement.")
        nombre_echecs += 1
        resultats.append({
            "SUP_ID": sup_id,
            "latitude": latitude,
            "longitude": longitude,
            "NAT_LB_NOM": support.NAT_LB_NOM,
            "image": nom_image,
            "statut": "echec",
        })
        continue

    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        print("    ERREUR : dimensions incorrectes.")
        nombre_echecs += 1
        resultats.append({
            "SUP_ID": sup_id,
            "latitude": latitude,
            "longitude": longitude,
            "NAT_LB_NOM": support.NAT_LB_NOM,
            "image": nom_image,
            "statut": "echec_dimensions",
        })
        continue

    image.save(image_path, format="JPEG", quality=95)
    nombre_ok += 1

    resultats.append({
        "SUP_ID": sup_id,
        "latitude": latitude,
        "longitude": longitude,
        "NAT_LB_NOM": support.NAT_LB_NOM,
        "image": nom_image,
        "statut": "ok",
    })

    print("    Image enregistrée.")

table_resultats = pd.DataFrame(resultats)
table_resultats.to_csv(METADATA_PATH, index=False)

images_finales = list(IMAGES_DIR.glob("support_*.jpg"))
nombre_images_finales = len(images_finales)

print("\n==================================")
print("TELECHARGEMENT TERMINE")
print("==================================")
print("Téléchargées maintenant :", nombre_ok)
print("Déjà présentes et valides :", nombre_existantes)
print("Échecs :", nombre_echecs)
print("Images finales dans le dossier :", nombre_images_finales)
print("Métadonnées :", METADATA_PATH)

nombre_succes_total = nombre_ok + nombre_existantes

if nombre_echecs == 0:
    if nombre_succes_total != NOMBRE_IMAGES:
        raise ValueError(
            "Le nombre d'images valides ne correspond pas au nombre attendu."
        )

    if nombre_images_finales != NOMBRE_IMAGES:
        raise ValueError(
            "Le dossier d'images ne contient pas exactement 300 images."
        )

    print(
        "\n✅ Dataset d'images cohérent : "
        f"{NOMBRE_IMAGES} supports / {NOMBRE_IMAGES} images."
    )
else:
    print("\n⚠️ Certaines images ont échoué.")
    print(
        "Relance simplement le script : les images valides seront ignorées "
        "et seules les manquantes seront retentées."
    )

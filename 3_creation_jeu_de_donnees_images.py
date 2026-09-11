import os
import math
import time
import io
import boto3
import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

# Dossier TEMPORAIRE local sur le service Onyxia
# Les images ne resteront PAS ici après upload
LOCAL_OUTPUT_DIR = "/tmp/images_bd_ortho"

# Bucket S3
S3_BUCKET = "landry"

# Dossier dans le bucket S3
S3_PREFIX = "Challenge 4 - Antennes sur images/images_bd_ortho"

# Nombre de mètres couverts par l'image
# 100 = image couvrant environ 100 m x 100 m
IMAGE_SIZE_METERS = 100

# Taille finale de l'image en pixels
IMAGE_PIXELS = 512

# Pour faire un test
MAX_IMAGES = 10

# Pour tout télécharger :
# MAX_IMAGES = None

# Pause entre les requêtes
REQUEST_DELAY = 0.1

# Serveur WMTS IGN Géoplateforme
WMTS_URL = "https://data.geopf.fr/wmts"

# Couche BD ORTHO
WMTS_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"

WMTS_STYLE = "normal"
WMTS_FORMAT = "image/jpeg"
WMTS_MATRIX_SET = "PM"

# Nombre de pixels d'une tuile WMTS
TILE_SIZE = 256

# CRS utilisé par la matrice PM
# Web Mercator
EARTH_RADIUS = 6378137.0


# ============================================================
# CLIENT S3
# ============================================================

s3 = boto3.client("s3")


# ============================================================
# CONVERSION DMS -> DEGRES DECIMAUX
# ============================================================

def dms_to_decimal(degrees, minutes, seconds, direction):
    """
    Convertit :

        46° 00' 13" N

    en :

        46.003611...

    Fonctionne aussi pour S et W.
    """

    if pd.isna(degrees) or pd.isna(minutes) or pd.isna(seconds):
        return None

    degrees = float(degrees)
    minutes = float(minutes)
    seconds = float(seconds)

    decimal = degrees + minutes / 60 + seconds / 3600

    direction = str(direction).strip().upper()

    if direction in ["S", "W"]:
        decimal = -decimal

    return decimal


# ============================================================
# LATITUDE / LONGITUDE
# ============================================================

def add_coordinates(df):

    df["latitude"] = df.apply(
        lambda row: dms_to_decimal(
            row["COR_NB_DG_LAT"],
            row["COR_NB_MN_LAT"],
            row["COR_NB_SC_LAT"],
            row["COR_CD_NS_LAT"]
        ),
        axis=1
    )

    df["longitude"] = df.apply(
        lambda row: dms_to_decimal(
            row["COR_NB_DG_LON"],
            row["COR_NB_MN_LON"],
            row["COR_NB_SC_LON"],
            row["COR_CD_EW_LON"]
        ),
        axis=1
    )

    return df


# ============================================================
# WGS84 -> WEB MERCATOR
# ============================================================

def lonlat_to_webmercator(lon, lat):

    x = EARTH_RADIUS * math.radians(lon)

    # Évite les problèmes aux pôles
    lat = max(min(lat, 85.05112878), -85.05112878)

    y = EARTH_RADIUS * math.log(
        math.tan(math.pi / 4 + math.radians(lat) / 2)
    )

    return x, y


# ============================================================
# WEB MERCATOR -> TILE XYZ
# ============================================================

def webmercator_to_tile(x, y, zoom):

    world_size = 2 * math.pi * EARTH_RADIUS

    resolution = world_size / (
        TILE_SIZE * (2 ** zoom)
    )

    # coordonnées pixels globales
    pixel_x = (
        x + math.pi * EARTH_RADIUS
    ) / resolution

    pixel_y = (
        math.pi * EARTH_RADIUS - y
    ) / resolution

    tile_x = int(pixel_x // TILE_SIZE)
    tile_y = int(pixel_y // TILE_SIZE)

    return tile_x, tile_y, pixel_x, pixel_y


# ============================================================
# CHOIX DU ZOOM
# ============================================================

def choose_zoom(image_size_meters, image_pixels):

    # Résolution souhaitée en mètres / pixel
    desired_resolution = (
        image_size_meters / image_pixels
    )

    # Résolution Web Mercator au zoom 0
    initial_resolution = (
        2 * math.pi * EARTH_RADIUS / TILE_SIZE
    )

    zoom = round(
        math.log2(
            initial_resolution /
            desired_resolution
        )
    )

    # Zoom raisonnable pour l'orthophoto
    zoom = max(0, min(zoom, 19))

    return zoom


# ============================================================
# URL D'UNE TUILE IGN
# ============================================================

def get_tile_url(x, y, zoom):

    return (
        f"{WMTS_URL}"
        f"?SERVICE=WMTS"
        f"&REQUEST=GetTile"
        f"&VERSION=1.0.0"
        f"&LAYER={WMTS_LAYER}"
        f"&STYLE={WMTS_STYLE}"
        f"&TILEMATRIXSET={WMTS_MATRIX_SET}"
        f"&TILEMATRIX={zoom}"
        f"&TILEROW={y}"
        f"&TILECOL={x}"
        f"&FORMAT={WMTS_FORMAT}"
    )


# ============================================================
# TELECHARGEMENT D'UNE TUILE
# ============================================================

def download_tile(session, x, y, zoom):

    url = get_tile_url(x, y, zoom)

    try:

        response = session.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        return Image.open(
            io.BytesIO(response.content)
        ).convert("RGB")

    except Exception as e:

        print(
            f"\nErreur tuile "
            f"zoom={zoom}, x={x}, y={y} : {e}"
        )

        return None


# ============================================================
# CREATION DE L'IMAGE AUTOUR DU SUPPORT
# ============================================================

def download_ortho_image(
    session,
    longitude,
    latitude,
    output_path
):

    zoom = choose_zoom(
        IMAGE_SIZE_METERS,
        IMAGE_PIXELS
    )

    # --------------------------------------------------------
    # Position du support en Web Mercator
    # --------------------------------------------------------

    x, y = lonlat_to_webmercator(
        longitude,
        latitude
    )

    # --------------------------------------------------------
    # Position dans la grille de tuiles
    # --------------------------------------------------------

    tile_x, tile_y, pixel_x, pixel_y = (
        webmercator_to_tile(
            x,
            y,
            zoom
        )
    )

    # --------------------------------------------------------
    # Résolution au zoom choisi
    # --------------------------------------------------------

    world_size = 2 * math.pi * EARTH_RADIUS

    resolution = (
        world_size /
        (TILE_SIZE * (2 ** zoom))
    )

    # --------------------------------------------------------
    # Taille de l'image en mètres
    # --------------------------------------------------------

    half_size = IMAGE_SIZE_METERS / 2

    # pixels nécessaires
    half_pixels = half_size / resolution

    # --------------------------------------------------------
    # Bounding box en pixels globaux
    # --------------------------------------------------------

    min_pixel_x = pixel_x - half_pixels
    max_pixel_x = pixel_x + half_pixels

    min_pixel_y = pixel_y - half_pixels
    max_pixel_y = pixel_y + half_pixels

    # --------------------------------------------------------
    # Tuiles nécessaires
    # --------------------------------------------------------

    min_tile_x = int(
        min_pixel_x // TILE_SIZE
    )

    max_tile_x = int(
        max_pixel_x // TILE_SIZE
    )

    min_tile_y = int(
        min_pixel_y // TILE_SIZE
    )

    max_tile_y = int(
        max_pixel_y // TILE_SIZE
    )

    # --------------------------------------------------------
    # Création de la mosaïque temporaire
    # --------------------------------------------------------

    tiles_x = (
        max_tile_x -
        min_tile_x +
        1
    )

    tiles_y = (
        max_tile_y -
        min_tile_y +
        1
    )

    mosaic = Image.new(
        "RGB",
        (
            tiles_x * TILE_SIZE,
            tiles_y * TILE_SIZE
        )
    )

    # --------------------------------------------------------
    # Téléchargement des tuiles
    # --------------------------------------------------------

    for ty in range(
        min_tile_y,
        max_tile_y + 1
    ):

        for tx in range(
            min_tile_x,
            max_tile_x + 1
        ):

            tile = download_tile(
                session,
                tx,
                ty,
                zoom
            )

            if tile is None:
                return False

            paste_x = (
                tx - min_tile_x
            ) * TILE_SIZE

            paste_y = (
                ty - min_tile_y
            ) * TILE_SIZE

            mosaic.paste(
                tile,
                (
                    paste_x,
                    paste_y
                )
            )

            time.sleep(
                REQUEST_DELAY
            )

    # --------------------------------------------------------
    # Position du support dans la mosaïque
    # --------------------------------------------------------

    support_x = (
        pixel_x -
        min_tile_x * TILE_SIZE
    )

    support_y = (
        pixel_y -
        min_tile_y * TILE_SIZE
    )

    # --------------------------------------------------------
    # Crop autour du support
    # --------------------------------------------------------

    left = int(
        support_x - half_pixels
    )

    top = int(
        support_y - half_pixels
    )

    right = int(
        support_x + half_pixels
    )

    bottom = int(
        support_y + half_pixels
    )

    image = mosaic.crop(
        (
            left,
            top,
            right,
            bottom
        )
    )

    # --------------------------------------------------------
    # Redimensionnement
    # --------------------------------------------------------

    image = image.resize(
        (
            IMAGE_PIXELS,
            IMAGE_PIXELS
        ),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Sauvegarde temporaire
    # --------------------------------------------------------

    image.save(
        output_path,
        "JPEG",
        quality=95
    )

    return True


# ============================================================
# VERIFICATION EXISTENCE S3
# ============================================================

def s3_file_exists(s3_key):

    try:

        s3.head_object(
            Bucket=S3_BUCKET,
            Key=s3_key
        )

        return True

    except s3.exceptions.ClientError as e:

        error_code = e.response["Error"]["Code"]

        if error_code in ["404", "NoSuchKey"]:
            return False

        raise


# ============================================================
# UPLOAD VERS S3
# ============================================================

def upload_to_s3(
    local_path,
    s3_key
):

    s3.upload_file(
        local_path,
        S3_BUCKET,
        s3_key
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("TELECHARGEMENT BD ORTHO - IGN")
    print("=" * 60)

    # --------------------------------------------------------
    # Création du dossier temporaire
    # --------------------------------------------------------

    os.makedirs(
        LOCAL_OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Lecture du CSV depuis S3
    # --------------------------------------------------------

    print("\nLecture du CSV...")

    bucket = "landry"

    key = (
        "Challenge 4 - Antennes sur images/"
        "Données data.gouv/"
        "data/dataset_antennes/"
        "SUP_SUPPORT.txt"
    )

    obj = s3.get_object(
        Bucket=bucket,
        Key=key
    )

    df = pd.read_csv(
        obj["Body"],
        sep=";",
        encoding="utf-8",
        low_memory=False
    )

    print(
        f"{len(df):,} supports trouvés."
    )

    # --------------------------------------------------------
    # Vérification des colonnes
    # --------------------------------------------------------

    required_columns = [
        "SUP_ID",
        "COR_NB_DG_LAT",
        "COR_NB_MN_LAT",
        "COR_NB_SC_LAT",
        "COR_CD_NS_LAT",
        "COR_NB_DG_LON",
        "COR_NB_MN_LON",
        "COR_NB_SC_LON",
        "COR_CD_EW_LON"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Conversion coordonnées
    # --------------------------------------------------------

    print(
        "\nConversion des coordonnées..."
    )

    df = add_coordinates(df)

    # --------------------------------------------------------
    # Suppression des coordonnées invalides
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).copy()

    after = len(df)

    print(
        f"Supports avec coordonnées valides : "
        f"{after:,}/{before:,}"
    )

    # --------------------------------------------------------
    # Limitation éventuelle
    # --------------------------------------------------------

    if MAX_IMAGES is not None:

        df = df.head(
            MAX_IMAGES
        )

        print(
            f"Mode test : "
            f"{len(df)} images maximum."
        )

    # --------------------------------------------------------
    # Session HTTP
    # --------------------------------------------------------

    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "Hackathon-FRHACK-26/1.0"
    })

    # --------------------------------------------------------
    # Téléchargement
    # --------------------------------------------------------

    downloaded = 0
    already_exists = 0
    failed = 0

    print(
        "\nTéléchargement des images...\n"
    )

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="BD ORTHO"
    ):

        support_id = str(
            row["SUP_ID"]
        ).strip()

        latitude = float(
            row["latitude"]
        )

        longitude = float(
            row["longitude"]
        )

        # ----------------------------------------------------
        # Chemin S3 final
        # ----------------------------------------------------

        s3_key = (
            f"{S3_PREFIX}/"
            f"{support_id}.jpg"
        )

        # ----------------------------------------------------
        # Vérification si déjà dans S3
        # ----------------------------------------------------

        try:

            if s3_file_exists(s3_key):

                already_exists += 1
                continue

        except Exception as e:

            print(
                f"\nErreur vérification S3 "
                f"{support_id} : {e}"
            )

        # ----------------------------------------------------
        # Fichier temporaire local
        # ----------------------------------------------------

        local_path = os.path.join(
            LOCAL_OUTPUT_DIR,
            f"{support_id}.jpg"
        )

        try:

            # -----------------------------------------------
            # Télécharger BD ORTHO
            # -----------------------------------------------

            success = download_ortho_image(
                session,
                longitude,
                latitude,
                local_path
            )

            if not success:

                failed += 1
                continue

            # -----------------------------------------------
            # Upload S3
            # -----------------------------------------------

            upload_to_s3(
                local_path,
                s3_key
            )

            downloaded += 1

            # -----------------------------------------------
            # Suppression du fichier local
            # -----------------------------------------------

            if os.path.exists(
                local_path
            ):
                os.remove(
                    local_path
                )

        except Exception as e:

            failed += 1

            print(
                f"\nErreur support "
                f"{support_id} : {e}"
            )

            # Nettoyage même en cas d'erreur
            if os.path.exists(
                local_path
            ):
                os.remove(
                    local_path
                )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TERMINÉ")
    print("=" * 60)

    print(
        f"Images envoyées dans S3 : "
        f"{downloaded:,}"
    )

    print(
        f"Images déjà présentes : "
        f"{already_exists:,}"
    )

    print(
        f"Échecs : "
        f"{failed:,}"
    )

    print(
        "\nDestination S3 :"
    )

    print(
        f"s3://{S3_BUCKET}/"
        f"{S3_PREFIX}/"
    )

    print(
        f"\nTaille image : "
        f"{IMAGE_PIXELS} x "
        f"{IMAGE_PIXELS} px"
    )

    print(
        f"Emprise : "
        f"{IMAGE_SIZE_METERS} m x "
        f"{IMAGE_SIZE_METERS} m"
    )


if __name__ == "__main__":
    main()
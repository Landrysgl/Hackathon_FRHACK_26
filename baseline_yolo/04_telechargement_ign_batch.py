import math
import os
import time
from io import BytesIO

import pandas as pd
import requests
from PIL import Image


# ---------------------------------------------------------
# Paramètres
# ---------------------------------------------------------

chemin_csv = "data/supports_yvelines.csv"
dossier_sortie = "data/images_ign_yvelines"

niveau_zoom = 19
taille_tuile = 256
taille_image = 1024

nombre_images = 300

delai_entre_requetes = 0.05

os.makedirs(dossier_sortie, exist_ok=True)


# ---------------------------------------------------------
# Chargement des supports ANFR
# ---------------------------------------------------------

supports = pd.read_csv(
    chemin_csv,
    dtype={
        "STA_NM_ANFR": str,
        "COM_CD_INSEE": str,
    },
)

print("Nombre total de supports disponibles :", len(supports))


# ---------------------------------------------------------
# Sélection d'un échantillon reproductible
# ---------------------------------------------------------

if nombre_images is not None and nombre_images < len(supports):
    supports_selectionnes = supports.sample(
        n=nombre_images,
        random_state=42,
    ).copy()
else:
    supports_selectionnes = supports.copy()

supports_selectionnes = supports_selectionnes.reset_index(drop=True)

print(
    "Nombre de supports qui seront téléchargés :",
    len(supports_selectionnes),
)


# ---------------------------------------------------------
# Conversion GPS vers coordonnées pixel Web Mercator
# ---------------------------------------------------------


def coordonnees_vers_pixels(latitude, longitude, zoom):
    latitude_rad = math.radians(latitude)

    taille_monde = taille_tuile * (2**zoom)

    pixel_x = (longitude + 180.0) / 360.0 * taille_monde

    pixel_y = (1.0 - math.asinh(math.tan(latitude_rad)) / math.pi) / 2.0 * taille_monde

    return pixel_x, pixel_y


# ---------------------------------------------------------
# Téléchargement d'une tuile IGN
# ---------------------------------------------------------


def telecharger_tuile(session, x, y, zoom):
    url = (
        "https://data.geopf.fr/wmts"
        "?SERVICE=WMTS"
        "&VERSION=1.0.0"
        "&REQUEST=GetTile"
        "&LAYER=ORTHOIMAGERY.ORTHOPHOTOS"
        "&STYLE=normal"
        "&FORMAT=image/jpeg"
        "&TILEMATRIXSET=PM"
        f"&TILEMATRIX={zoom}"
        f"&TILEROW={y}"
        f"&TILECOL={x}"
    )

    for tentative in range(3):
        try:
            reponse = session.get(
                url,
                timeout=30,
            )

            reponse.raise_for_status()

            image = Image.open(BytesIO(reponse.content)).convert("RGB")

            return image

        except Exception as erreur:
            print(f"    Tentative {tentative + 1}/3 échouée : {erreur}")

            time.sleep(2)

    return None


# ---------------------------------------------------------
# Création d'une image centrée sur un support
# ---------------------------------------------------------


def creer_image_centree(
    session,
    latitude,
    longitude,
    zoom,
    taille_sortie,
):
    pixel_x, pixel_y = coordonnees_vers_pixels(
        latitude,
        longitude,
        zoom,
    )

    demi_taille = taille_sortie / 2

    pixel_gauche = pixel_x - demi_taille
    pixel_haut = pixel_y - demi_taille

    pixel_droite = pixel_x + demi_taille
    pixel_bas = pixel_y + demi_taille

    tuile_x_min = math.floor(pixel_gauche / taille_tuile)

    tuile_x_max = math.floor((pixel_droite - 1) / taille_tuile)

    tuile_y_min = math.floor(pixel_haut / taille_tuile)

    tuile_y_max = math.floor((pixel_bas - 1) / taille_tuile)

    largeur_mosaique = (tuile_x_max - tuile_x_min + 1) * taille_tuile

    hauteur_mosaique = (tuile_y_max - tuile_y_min + 1) * taille_tuile

    mosaique = Image.new(
        "RGB",
        (
            largeur_mosaique,
            hauteur_mosaique,
        ),
    )

    for tuile_y in range(
        tuile_y_min,
        tuile_y_max + 1,
    ):
        for tuile_x in range(
            tuile_x_min,
            tuile_x_max + 1,
        ):
            image_tuile = telecharger_tuile(
                session,
                tuile_x,
                tuile_y,
                zoom,
            )

            if image_tuile is None:
                return None

            position_x = (tuile_x - tuile_x_min) * taille_tuile

            position_y = (tuile_y - tuile_y_min) * taille_tuile

            mosaique.paste(
                image_tuile,
                (
                    position_x,
                    position_y,
                ),
            )

            time.sleep(delai_entre_requetes)

    origine_pixel_x = tuile_x_min * taille_tuile

    origine_pixel_y = tuile_y_min * taille_tuile

    centre_x_mosaique = pixel_x - origine_pixel_x

    centre_y_mosaique = pixel_y - origine_pixel_y

    gauche = round(centre_x_mosaique - demi_taille)

    haut = round(centre_y_mosaique - demi_taille)

    droite = gauche + taille_sortie
    bas = haut + taille_sortie

    image_finale = mosaique.crop(
        (
            gauche,
            haut,
            droite,
            bas,
        )
    )

    return image_finale


# ---------------------------------------------------------
# Téléchargement en batch
# ---------------------------------------------------------

session = requests.Session()

resultats = []

nombre_succes = 0
nombre_echecs = 0


for index, support in supports_selectionnes.iterrows():
    sup_id = support["SUP_ID"]

    latitude = float(support["latitude"])

    longitude = float(support["longitude"])

    nom_image = f"support_{sup_id}.jpg"

    chemin_image = os.path.join(
        dossier_sortie,
        nom_image,
    )

    print(f"[{index + 1}/{len(supports_selectionnes)}] Support {sup_id}")

    if os.path.exists(chemin_image):
        print("    Image déjà présente.")

        nombre_succes += 1

        resultats.append(
            {
                "SUP_ID": sup_id,
                "latitude": latitude,
                "longitude": longitude,
                "image": nom_image,
                "statut": "existante",
            }
        )

        continue

    image = creer_image_centree(
        session,
        latitude,
        longitude,
        niveau_zoom,
        taille_image,
    )

    if image is None:
        print("    Échec du téléchargement.")

        nombre_echecs += 1

        resultats.append(
            {
                "SUP_ID": sup_id,
                "latitude": latitude,
                "longitude": longitude,
                "image": nom_image,
                "statut": "echec",
            }
        )

        continue

    image.save(
        chemin_image,
        quality=95,
    )

    nombre_succes += 1

    resultats.append(
        {
            "SUP_ID": sup_id,
            "latitude": latitude,
            "longitude": longitude,
            "image": nom_image,
            "statut": "ok",
        }
    )

    print(f"    Image enregistrée : {chemin_image}")


# ---------------------------------------------------------
# Sauvegarde des métadonnées
# ---------------------------------------------------------

table_resultats = pd.DataFrame(resultats)

chemin_metadonnees = os.path.join(
    dossier_sortie,
    "metadonnees.csv",
)

table_resultats.to_csv(
    chemin_metadonnees,
    index=False,
)


# ---------------------------------------------------------
# Résumé
# ---------------------------------------------------------

print("\n----------------------------------")
print("Téléchargement terminé")
print("----------------------------------")

print("Succès :", nombre_succes)
print("Échecs :", nombre_echecs)

print(
    "Métadonnées enregistrées dans :",
    chemin_metadonnees,
)

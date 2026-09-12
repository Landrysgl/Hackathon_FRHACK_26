import math
import os
from io import BytesIO

import pandas as pd
import requests
from PIL import Image

# ---------------------------------------------------------
# Paramètres
# ---------------------------------------------------------

chemin_csv = "data/supports_yvelines.csv"
dossier_sortie = "data/test_ign"

niveau_zoom = 19
taille_tuile = 256

os.makedirs(dossier_sortie, exist_ok=True)


# ---------------------------------------------------------
# Chargement du premier support
# ---------------------------------------------------------

supports = pd.read_csv(chemin_csv, dtype={"STA_NM_ANFR": str, "COM_CD_INSEE": str})

support = supports.iloc[0]

latitude = float(support["latitude"])
longitude = float(support["longitude"])
sup_id = support["SUP_ID"]

print("Support testé :", sup_id)
print("Latitude :", latitude)
print("Longitude :", longitude)


# ---------------------------------------------------------
# Conversion latitude / longitude vers une tuile Web Mercator
# ---------------------------------------------------------


def coordonnees_vers_tuile(latitude, longitude, zoom):
    latitude_rad = math.radians(latitude)

    n = 2**zoom

    x = int((longitude + 180.0) / 360.0 * n)

    y = int((1.0 - math.asinh(math.tan(latitude_rad)) / math.pi) / 2.0 * n)

    return x, y


x, y = coordonnees_vers_tuile(latitude, longitude, niveau_zoom)

print("Tuile X :", x)
print("Tuile Y :", y)
print("Zoom :", niveau_zoom)


# ---------------------------------------------------------
# Téléchargement de la tuile IGN
# ---------------------------------------------------------

url = (
    "https://data.geopf.fr/wmts"
    "?SERVICE=WMTS"
    "&VERSION=1.0.0"
    "&REQUEST=GetTile"
    "&LAYER=ORTHOIMAGERY.ORTHOPHOTOS"
    "&STYLE=normal"
    "&FORMAT=image/jpeg"
    "&TILEMATRIXSET=PM"
    f"&TILEMATRIX={niveau_zoom}"
    f"&TILEROW={y}"
    f"&TILECOL={x}"
)

print("\nTéléchargement de l'image IGN...")

reponse = requests.get(url, timeout=30)

print("Code HTTP :", reponse.status_code)
print("Type de contenu :", reponse.headers.get("Content-Type"))

reponse.raise_for_status()


# ---------------------------------------------------------
# Sauvegarde de l'image
# ---------------------------------------------------------

image = Image.open(BytesIO(reponse.content))

chemin_image = os.path.join(dossier_sortie, f"support_{sup_id}.jpg")

image.save(chemin_image)

print("\nImage enregistrée :")
print(chemin_image)

print("Dimensions :", image.size)

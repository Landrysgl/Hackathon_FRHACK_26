from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SELECTION_PATH = DATA_DIR / "supports_selectionnes_niveau1.csv"
IMAGES_DIR = DATA_DIR / "images_ign_yvelines"
INSPECTION_DIR = DATA_DIR / "inspection_images"

NOMBRE_IMAGES_A_INSPECTER = 20
IMAGE_SIZE = 1024
CENTER_X = IMAGE_SIZE // 2
CENTER_Y = IMAGE_SIZE // 2
RANDOM_SEED = 42

if not SELECTION_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {SELECTION_PATH}\n"
        "Lance d'abord 04_telechargement_ign_batch.py."
    )

if not IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Dossier introuvable : {IMAGES_DIR}\n"
        "Lance d'abord 04_telechargement_ign_batch.py."
    )

if INSPECTION_DIR.exists():
    shutil.rmtree(INSPECTION_DIR)
INSPECTION_DIR.mkdir(parents=True, exist_ok=True)

supports = pd.read_csv(
    SELECTION_PATH,
    dtype={
        "SUP_ID": "string",
        "STA_NM_ANFR": "string",
        "COM_CD_INSEE": "string",
    },
)

print("===== SELECTION NIVEAU 1 =====")
print("Nombre de lignes :", len(supports))
print("Nombre de SUP_ID uniques :", supports["SUP_ID"].nunique())

if supports["SUP_ID"].duplicated().any():
    raise ValueError("Le fichier de sélection contient des SUP_ID dupliqués.")

if len(supports) != 300:
    raise ValueError(
        f"Le Niveau 1 devrait contenir 300 supports, "
        f"mais {len(supports)} ont été trouvés."
    )

print("Doublons SUP_ID : 0")

images_attendues = {
    f"support_{sup_id}.jpg"
    for sup_id in supports["SUP_ID"]
}
images_presentes = {
    image_path.name
    for image_path in IMAGES_DIR.glob("support_*.jpg")
}

images_manquantes = images_attendues - images_presentes
images_en_trop = images_presentes - images_attendues

print("\n===== CORRESPONDANCE =====")
print("Images attendues :", len(images_attendues))
print("Images présentes :", len(images_presentes))
print("Images manquantes :", len(images_manquantes))
print("Images en trop :", len(images_en_trop))

if images_manquantes:
    print("\nExemples d'images manquantes :")
    print(sorted(images_manquantes)[:10])

if images_en_trop:
    print("\nExemples d'images en trop :")
    print(sorted(images_en_trop)[:10])

if images_manquantes or images_en_trop:
    raise ValueError(
        "Les 300 supports et les images ne correspondent pas exactement."
    )

images_corrompues = []
dimensions_invalides = []

for nom_image in sorted(images_attendues):
    image_path = IMAGES_DIR / nom_image
    try:
        with Image.open(image_path) as image:
            image.load()
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                dimensions_invalides.append((nom_image, image.size))
    except Exception as erreur:
        images_corrompues.append((nom_image, str(erreur)))

print("\n===== INTEGRITE DES IMAGES =====")
print("Images corrompues :", len(images_corrompues))
print("Dimensions incorrectes :", len(dimensions_invalides))

if images_corrompues:
    print(images_corrompues[:10])
if dimensions_invalides:
    print(dimensions_invalides[:10])

if images_corrompues or dimensions_invalides:
    raise ValueError("Certaines images du dataset sont invalides.")

print("✅ Les 300 images sont lisibles et font 1024 x 1024.")

supports_tries = supports.sort_values("SUP_ID").reset_index(drop=True)

types_principaux = (
    supports_tries["NAT_LB_NOM"]
    .value_counts()
    .head(10)
    .index
)

echantillons_divers = []

for numero_type, type_support in enumerate(types_principaux):
    groupe = supports_tries[
        supports_tries["NAT_LB_NOM"] == type_support
    ]
    ligne = groupe.sample(
        n=1,
        random_state=RANDOM_SEED + numero_type,
    )
    echantillons_divers.append(ligne)

inspection = pd.concat(echantillons_divers, ignore_index=True)

ids_deja_selectionnes = set(inspection["SUP_ID"])
reste = supports_tries[
    ~supports_tries["SUP_ID"].isin(ids_deja_selectionnes)
]

nombre_a_ajouter = NOMBRE_IMAGES_A_INSPECTER - len(inspection)

if nombre_a_ajouter > 0:
    complement = reste.sample(
        n=nombre_a_ajouter,
        random_state=RANDOM_SEED,
    )
    inspection = pd.concat(
        [inspection, complement],
        ignore_index=True,
    )

inspection = (
    inspection
    .sort_values(["NAT_LB_NOM", "SUP_ID"])
    .reset_index(drop=True)
)

print("\n===== INSPECTION VISUELLE =====")
print("Images sélectionnées :", len(inspection))
print("\nTypes présents :")
print(inspection["NAT_LB_NOM"].value_counts())

resultats = []

for numero, support in inspection.iterrows():
    sup_id = str(support["SUP_ID"])
    nom_image = f"support_{sup_id}.jpg"
    image_path = IMAGES_DIR / nom_image

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        figure, axe = plt.subplots(figsize=(9, 9))
        axe.imshow(image)

        axe.scatter(
            CENTER_X,
            CENTER_Y,
            marker="x",
            s=250,
            linewidths=3,
            color="red",
            label="Coordonnée ANFR",
        )

        nature = support["NAT_LB_NOM"]
        hauteur = support.get("SUP_NM_HAUT", "inconnue")
        latitude = support["latitude"]
        longitude = support["longitude"]

        axe.set_title(
            f"SUP_ID : {sup_id}\n"
            f"Nature : {nature} | Hauteur : {hauteur} m\n"
            f"Lat : {latitude:.6f} | Lon : {longitude:.6f}",
            fontsize=12,
        )

        axe.legend(loc="upper right")
        axe.axis("off")

        nom_inspection = f"inspection_support_{sup_id}.jpg"
        inspection_path = INSPECTION_DIR / nom_inspection

        figure.savefig(
            inspection_path,
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(figure)

    resultats.append({
        "SUP_ID": sup_id,
        "image_originale": nom_image,
        "image_inspection": nom_inspection,
        "NAT_ID": support.get("NAT_ID", ""),
        "NAT_LB_NOM": nature,
        "SUP_NM_HAUT": hauteur,
        "latitude": latitude,
        "longitude": longitude,
        "controle_visuel": "",
        "commentaire": "",
    })

    print(
        f"[{numero + 1}/{len(inspection)}] "
        f"{nom_inspection}"
    )

table_resultats = pd.DataFrame(resultats)
RESUME_PATH = INSPECTION_DIR / "inspection_resume.csv"
table_resultats.to_csv(RESUME_PATH, index=False)

figure, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for axe, (_, support) in zip(axes, inspection.iterrows()):
    sup_id = str(support["SUP_ID"])
    image_path = IMAGES_DIR / f"support_{sup_id}.jpg"

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        axe.imshow(image)

    axe.scatter(
        CENTER_X,
        CENTER_Y,
        marker="x",
        s=100,
        linewidths=2,
        color="red",
    )

    axe.set_title(
        f"{sup_id}\n{support['NAT_LB_NOM']}",
        fontsize=9,
    )
    axe.axis("off")

MOSAIQUE_PATH = INSPECTION_DIR / "inspection_mosaique.jpg"

figure.tight_layout()
figure.savefig(MOSAIQUE_PATH, dpi=120)
plt.close(figure)

print("\n==================================")
print("INSPECTION PREPAREE")
print("==================================")
print("Dataset : 300 supports / 300 images valides")
print("Images à inspecter :", len(inspection))
print("Dossier :", INSPECTION_DIR)
print("Mosaïque :", MOSAIQUE_PATH)
print("Résumé CSV :", RESUME_PATH)

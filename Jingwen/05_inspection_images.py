import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


# ---------------------------------------------------------
# Paramètres
# ---------------------------------------------------------

chemin_supports = "data/supports_yvelines.csv"
chemin_natures = "data/raw/references/SUP_NATURE.txt"

dossier_images = "data/images_ign_yvelines"
dossier_inspection = "data/inspection_images"

nombre_images_a_inspecter = 20

os.makedirs(
    dossier_inspection,
    exist_ok=True,
)


# ---------------------------------------------------------
# Chargement des supports ANFR
# ---------------------------------------------------------

supports = pd.read_csv(
    chemin_supports,
    dtype={
        "STA_NM_ANFR": str,
        "COM_CD_INSEE": str,
    },
)


# ---------------------------------------------------------
# Chargement des natures de supports
# ---------------------------------------------------------

natures = pd.read_csv(
    chemin_natures,
    sep=";",
    header=None,
    names=[
        "NAT_ID",
        "NATURE_SUPPORT",
    ],
)

natures["NAT_ID"] = pd.to_numeric(
    natures["NAT_ID"],
    errors="coerce",
)

natures = natures.dropna(subset=["NAT_ID"])

natures["NAT_ID"] = natures["NAT_ID"].astype(int)


# ---------------------------------------------------------
# Association des supports avec leur nature
# ---------------------------------------------------------

supports = supports.merge(
    natures,
    on="NAT_ID",
    how="left",
)


# ---------------------------------------------------------
# Recherche des images disponibles
# ---------------------------------------------------------

images_disponibles = [
    fichier
    for fichier in os.listdir(dossier_images)
    if fichier.lower().endswith(".jpg")
]

print(
    "Nombre d'images disponibles :",
    len(images_disponibles),
)


# ---------------------------------------------------------
# Sélection reproductible de 20 images
# ---------------------------------------------------------

images_selectionnees = (
    pd.Series(images_disponibles)
    .sample(
        n=min(
            nombre_images_a_inspecter,
            len(images_disponibles),
        ),
        random_state=42,
    )
    .tolist()
)


# ---------------------------------------------------------
# Création des images d'inspection
# ---------------------------------------------------------

resultats = []

for numero, nom_image in enumerate(
    images_selectionnees,
    start=1,
):
    sup_id = nom_image.replace("support_", "").replace(".jpg", "")

    support_correspondant = supports[supports["SUP_ID"].astype(str) == str(sup_id)]

    if support_correspondant.empty:
        print(
            "Support introuvable :",
            sup_id,
        )
        continue

    support = support_correspondant.iloc[0]

    chemin_image = os.path.join(
        dossier_images,
        nom_image,
    )

    image = Image.open(chemin_image).convert("RGB")

    # -----------------------------------------------------
    # Création de la figure
    # -----------------------------------------------------

    figure, axe = plt.subplots(figsize=(10, 10))

    axe.imshow(image)

    # -----------------------------------------------------
    # Position théorique de la coordonnée ANFR
    # -----------------------------------------------------

    axe.scatter(
        512,
        512,
        marker="x",
        s=250,
        linewidths=3,
        label="Coordonnée ANFR",
    )

    # -----------------------------------------------------
    # Informations ANFR
    # -----------------------------------------------------

    nature_support = support["NATURE_SUPPORT"]
    hauteur_support = support["SUP_NM_HAUT"]

    axe.set_title(
        f"SUP_ID : {sup_id}\n"
        f"Nature du support : {nature_support}\n"
        f"Hauteur : {hauteur_support} m",
        fontsize=14,
    )

    axe.legend()

    axe.axis("off")

    # -----------------------------------------------------
    # Sauvegarde de l'image d'inspection
    # -----------------------------------------------------

    nom_image_inspection = f"inspection_support_{sup_id}.jpg"

    chemin_image_inspection = os.path.join(
        dossier_inspection,
        nom_image_inspection,
    )

    figure.savefig(
        chemin_image_inspection,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    # -----------------------------------------------------
    # Enregistrement des informations
    # -----------------------------------------------------

    resultats.append(
        {
            "SUP_ID": sup_id,
            "image_originale": nom_image,
            "image_inspection": nom_image_inspection,
            "NAT_ID": support["NAT_ID"],
            "NATURE_SUPPORT": nature_support,
            "SUP_NM_HAUT": hauteur_support,
            "latitude": support["latitude"],
            "longitude": support["longitude"],
        }
    )

    print(
        f"[{numero}/{len(images_selectionnees)}] Image créée : {nom_image_inspection}"
    )


# ---------------------------------------------------------
# Création du fichier récapitulatif
# ---------------------------------------------------------

table_resultats = pd.DataFrame(resultats)

chemin_resume = os.path.join(
    dossier_inspection,
    "inspection_resume.csv",
)

table_resultats.to_csv(
    chemin_resume,
    index=False,
)


# ---------------------------------------------------------
# Résumé
# ---------------------------------------------------------

print("\n----------------------------------")
print("Inspection terminée")
print("----------------------------------")

print(
    "Nombre d'images créées :",
    len(resultats),
)

print(
    "Dossier :",
    dossier_inspection,
)

print(
    "Fichier récapitulatif :",
    chemin_resume,
)

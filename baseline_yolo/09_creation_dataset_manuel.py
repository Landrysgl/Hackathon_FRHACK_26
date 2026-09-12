import os
import shutil
import pandas as pd


# ---------------------------------------------------------
# Paramètres
# ---------------------------------------------------------

dossier_dataset_source = "data/dataset_yolo"
dossier_annotations_manuelles = "data/annotations_manuelles"

chemin_resume = "data/annotations_manuelles_resume.csv"

dossier_dataset_sortie = "data/dataset_yolo_manuel"


# ---------------------------------------------------------
# Création de la structure
# ---------------------------------------------------------

for sous_ensemble in [
    "train",
    "val",
    "test",
]:

    os.makedirs(
        os.path.join(
            dossier_dataset_sortie,
            "images",
            sous_ensemble,
        ),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(
            dossier_dataset_sortie,
            "labels",
            sous_ensemble,
        ),
        exist_ok=True,
    )


# ---------------------------------------------------------
# Copie du dataset original
# ---------------------------------------------------------

for sous_ensemble in [
    "train",
    "val",
    "test",
]:

    dossier_images_source = os.path.join(
        dossier_dataset_source,
        "images",
        sous_ensemble,
    )

    dossier_labels_source = os.path.join(
        dossier_dataset_source,
        "labels",
        sous_ensemble,
    )

    dossier_images_sortie = os.path.join(
        dossier_dataset_sortie,
        "images",
        sous_ensemble,
    )

    dossier_labels_sortie = os.path.join(
        dossier_dataset_sortie,
        "labels",
        sous_ensemble,
    )

    for fichier in os.listdir(
        dossier_images_source
    ):

        shutil.copy2(
            os.path.join(
                dossier_images_source,
                fichier,
            ),
            os.path.join(
                dossier_images_sortie,
                fichier,
            ),
        )

    if os.path.exists(
        dossier_labels_source
    ):

        for fichier in os.listdir(
            dossier_labels_source
        ):

            shutil.copy2(
                os.path.join(
                    dossier_labels_source,
                    fichier,
                ),
                os.path.join(
                    dossier_labels_sortie,
                    fichier,
                ),
            )


# ---------------------------------------------------------
# Chargement des annotations manuelles
# ---------------------------------------------------------

table = pd.read_csv(
    chemin_resume
)


# ---------------------------------------------------------
# Remplacement des annotations faibles
# par les annotations manuelles
# ---------------------------------------------------------

nombre_remplacements = 0


for _, ligne in table.iterrows():

    statut = ligne["statut"]

    if statut != "annote":
        continue

    nom_image = ligne["image"]

    nom_label = os.path.splitext(
        nom_image
    )[0] + ".txt"

    chemin_label_manuel = os.path.join(
        dossier_annotations_manuelles,
        nom_label,
    )

    chemin_label_sortie = os.path.join(
        dossier_dataset_sortie,
        "labels",
        "train",
        nom_label,
    )

    if not os.path.exists(
        chemin_label_manuel
    ):

        print(
            "Annotation manuelle introuvable :",
            chemin_label_manuel,
        )

        continue

    shutil.copy2(
        chemin_label_manuel,
        chemin_label_sortie,
    )

    nombre_remplacements += 1

    print(
        "Annotation remplacée :",
        nom_label,
    )


# ---------------------------------------------------------
# Création du fichier YAML
# ---------------------------------------------------------

chemin_yaml = os.path.join(
    dossier_dataset_sortie,
    "data.yaml",
)


contenu_yaml = f"""
path: {os.path.abspath(dossier_dataset_sortie)}
train: images/train
val: images/val
test: images/test

names:
  0: support_radio
"""


with open(
    chemin_yaml,
    "w",
    encoding="utf-8",
) as fichier:

    fichier.write(
        contenu_yaml.strip()
    )


# ---------------------------------------------------------
# Résumé
# ---------------------------------------------------------

print()
print(
    "Nombre d'annotations manuelles utilisées :",
    nombre_remplacements,
)

print(
    "Dataset créé :",
    dossier_dataset_sortie,
)

print(
    "Fichier YAML :",
    chemin_yaml,
)
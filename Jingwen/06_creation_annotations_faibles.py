import os
import shutil
import pandas as pd


# ---------------------------------------------------------
# Paramètres
# ---------------------------------------------------------

chemin_supports = "data/supports_yvelines.csv"
dossier_images_source = "data/images_ign_yvelines"

dossier_dataset = "data/dataset_yolo"
dossier_images = os.path.join(dossier_dataset, "images")
dossier_labels = os.path.join(dossier_dataset, "labels")

taille_image = 1024
taille_boite = 160

proportion_train = 0.70
proportion_validation = 0.15
proportion_test = 0.15

graine_aleatoire = 42


# ---------------------------------------------------------
# Création des dossiers
# ---------------------------------------------------------

for sous_ensemble in ["train", "val", "test"]:
    os.makedirs(
        os.path.join(
            dossier_images,
            sous_ensemble,
        ),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(
            dossier_labels,
            sous_ensemble,
        ),
        exist_ok=True,
    )


# ---------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------

supports = pd.read_csv(
    chemin_supports,
)

supports["SUP_ID_TEXTE"] = supports["SUP_ID"].astype(str)


# ---------------------------------------------------------
# Recherche des images disponibles
# ---------------------------------------------------------

images_disponibles = [
    fichier
    for fichier in os.listdir(dossier_images_source)
    if fichier.lower().endswith(".jpg")
]


donnees = []

for nom_image in images_disponibles:
    sup_id = nom_image.replace("support_", "").replace(".jpg", "")

    if sup_id not in supports["SUP_ID_TEXTE"].values:
        continue

    donnees.append(
        {
            "SUP_ID": sup_id,
            "image": nom_image,
        }
    )


table = pd.DataFrame(donnees)


print(
    "Nombre d'images utilisables :",
    len(table),
)


# ---------------------------------------------------------
# Mélange reproductible
# ---------------------------------------------------------

table = table.sample(
    frac=1,
    random_state=graine_aleatoire,
).reset_index(drop=True)


# ---------------------------------------------------------
# Séparation train / validation / test
# ---------------------------------------------------------

nombre_total = len(table)

nombre_train = int(nombre_total * proportion_train)

nombre_validation = int(nombre_total * proportion_validation)

table_train = table.iloc[:nombre_train].copy()

table_validation = table.iloc[nombre_train : nombre_train + nombre_validation].copy()

table_test = table.iloc[nombre_train + nombre_validation :].copy()


ensembles = {
    "train": table_train,
    "val": table_validation,
    "test": table_test,
}


# ---------------------------------------------------------
# Conversion de la boîte en format YOLO
# ---------------------------------------------------------

centre_x = 512
centre_y = 512

largeur_boite = taille_boite
hauteur_boite = taille_boite


centre_x_normalise = centre_x / taille_image

centre_y_normalise = centre_y / taille_image

largeur_normalisee = largeur_boite / taille_image

hauteur_normalisee = hauteur_boite / taille_image


# ---------------------------------------------------------
# Création du dataset YOLO
# ---------------------------------------------------------

classe_support = 0


for nom_ensemble, table_ensemble in ensembles.items():
    print(f"\nCréation de l'ensemble : {nom_ensemble}")

    for _, ligne in table_ensemble.iterrows():
        nom_image = ligne["image"]

        chemin_image_source = os.path.join(
            dossier_images_source,
            nom_image,
        )

        chemin_image_destination = os.path.join(
            dossier_images,
            nom_ensemble,
            nom_image,
        )

        shutil.copy2(
            chemin_image_source,
            chemin_image_destination,
        )

        nom_label = os.path.splitext(nom_image)[0] + ".txt"

        chemin_label = os.path.join(
            dossier_labels,
            nom_ensemble,
            nom_label,
        )

        with open(
            chemin_label,
            "w",
            encoding="utf-8",
        ) as fichier_label:
            fichier_label.write(
                f"{classe_support} "
                f"{centre_x_normalise:.6f} "
                f"{centre_y_normalise:.6f} "
                f"{largeur_normalisee:.6f} "
                f"{hauteur_normalisee:.6f}\n"
            )


# ---------------------------------------------------------
# Création du fichier data.yaml
# ---------------------------------------------------------

chemin_yaml = os.path.join(
    dossier_dataset,
    "data.yaml",
)

contenu_yaml = f"""path: {os.path.abspath(dossier_dataset)}
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
) as fichier_yaml:
    fichier_yaml.write(contenu_yaml)


# ---------------------------------------------------------
# Résumé
# ---------------------------------------------------------

print("\n----------------------------------")
print("Dataset YOLO créé")
print("----------------------------------")

print(
    "Train :",
    len(table_train),
)

print(
    "Validation :",
    len(table_validation),
)

print(
    "Test :",
    len(table_test),
)

print(
    "Fichier YAML :",
    chemin_yaml,
)

print(
    "\nBoîte faible utilisée :",
    f"{taille_boite} x {taille_boite} pixels",
)

print(
    "Centre :",
    centre_x,
    centre_y,
)

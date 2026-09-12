import os
import pandas as pd


# ---------------------------------------------------------
# Paramètres
# ---------------------------------------------------------

chemin_resume = "data/annotations_manuelles_resume.csv"

dossier_dataset = "data/dataset_yolo/images"


# ---------------------------------------------------------
# Chargement du résumé manuel
# ---------------------------------------------------------

table = pd.read_csv(
    chemin_resume,
)


print("\n----------------------------------")
print("Statuts des annotations manuelles")
print("----------------------------------")

print(
    table["statut"].value_counts(
        dropna=False
    )
)


# ---------------------------------------------------------
# Recherche de l'ensemble original
# ---------------------------------------------------------

def trouver_ensemble(nom_image):

    for sous_ensemble in [
        "train",
        "val",
        "test",
    ]:

        chemin_image = os.path.join(
            dossier_dataset,
            sous_ensemble,
            nom_image,
        )

        if os.path.exists(
            chemin_image
        ):
            return sous_ensemble

    return "inconnu"


table["ensemble_original"] = table[
    "image"
].apply(
    trouver_ensemble
)


# ---------------------------------------------------------
# Résumé des ensembles
# ---------------------------------------------------------

print("\n----------------------------------")
print("Répartition train / val / test")
print("----------------------------------")

print(
    table["ensemble_original"].value_counts()
)


# ---------------------------------------------------------
# Croisement statut / ensemble
# ---------------------------------------------------------

print("\n----------------------------------")
print("Statut par ensemble")
print("----------------------------------")

print(
    pd.crosstab(
        table["ensemble_original"],
        table["statut"],
    )
)


# ---------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------

chemin_sortie = (
    "data/annotations_manuelles_analyse.csv"
)

table.to_csv(
    chemin_sortie,
    index=False,
)


print("\nFichier créé :", chemin_sortie)
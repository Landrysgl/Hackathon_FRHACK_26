import os

import pandas as pd

from ultralytics import YOLO


# ---------------------------------------------------------
# Paramètres
# ---------------------------------------------------------

chemin_resume = "data/annotations_validation_resume.csv"

dossier_images = "data/dataset_yolo/images/val"

dossier_labels_manuels = "data/annotations_validation"

chemin_modele = "runs/detect/runs/detect/antennes/weights/best.pt"

seuil_confiance = 0.05

seuil_iou = 0.50

# ---------------------------------------------------------
# Fonction IoU
# ---------------------------------------------------------


def calculer_iou(
    boite_a,
    boite_b,
):

    x1_a, y1_a, x2_a, y2_a = boite_a
    x1_b, y1_b, x2_b, y2_b = boite_b

    x_inter_min = max(
        x1_a,
        x1_b,
    )

    y_inter_min = max(
        y1_a,
        y1_b,
    )

    x_inter_max = min(
        x2_a,
        x2_b,
    )

    y_inter_max = min(
        y2_a,
        y2_b,
    )

    largeur_inter = max(
        0,
        x_inter_max - x_inter_min,
    )

    hauteur_inter = max(
        0,
        y_inter_max - y_inter_min,
    )

    aire_intersection = largeur_inter * hauteur_inter

    aire_a = (x2_a - x1_a) * (y2_a - y1_a)

    aire_b = (x2_b - x1_b) * (y2_b - y1_b)

    aire_union = aire_a + aire_b - aire_intersection

    if aire_union == 0:
        return 0

    return aire_intersection / aire_union


# ---------------------------------------------------------
# Lecture d'un label YOLO manuel
# ---------------------------------------------------------


def lire_label_manuel(
    chemin_label,
    largeur_image,
    hauteur_image,
):

    with open(
        chemin_label,
        "r",
        encoding="utf-8",
    ) as fichier:
        ligne = fichier.readline().strip()

    valeurs = ligne.split()

    centre_x = float(valeurs[1])

    centre_y = float(valeurs[2])

    largeur_boite = float(valeurs[3])

    hauteur_boite = float(valeurs[4])

    centre_x *= largeur_image
    centre_y *= hauteur_image

    largeur_boite *= largeur_image
    hauteur_boite *= hauteur_image

    x_min = centre_x - largeur_boite / 2

    y_min = centre_y - hauteur_boite / 2

    x_max = centre_x + largeur_boite / 2

    y_max = centre_y + hauteur_boite / 2

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


# ---------------------------------------------------------
# Chargement
# ---------------------------------------------------------

table = pd.read_csv(chemin_resume)

modele = YOLO(chemin_modele)


# ---------------------------------------------------------
# Compteurs
# ---------------------------------------------------------

tp = 0
fp = 0
fn = 0

nombre_ambigu = 0

details = []


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

for _, ligne in table.iterrows():
    nom_image = ligne["image"]

    statut = ligne["statut"]

    chemin_image = os.path.join(
        dossier_images,
        nom_image,
    )

    # -----------------------------------------------------
    # Images ambiguës
    # -----------------------------------------------------

    if statut == "ambigu":
        nombre_ambigu += 1

        details.append(
            {
                "image": nom_image,
                "statut": statut,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "iou_max": None,
            }
        )

        continue

    # -----------------------------------------------------
    # Prédiction YOLO
    # -----------------------------------------------------

    resultats = modele.predict(
        source=chemin_image,
        conf=seuil_confiance,
        verbose=False,
    )

    resultat = resultats[0]

    hauteur_image, largeur_image = resultat.orig_shape

    boites_predites = []

    if resultat.boxes is not None:
        for boite in resultat.boxes:
            coordonnees = boite.xyxy[0].cpu().tolist()

            boites_predites.append(tuple(coordonnees))

    # -----------------------------------------------------
    # Cas : non visible
    # -----------------------------------------------------

    if statut == "non_visible":
        nombre_predictions = len(boites_predites)

        fp += nombre_predictions

        details.append(
            {
                "image": nom_image,
                "statut": statut,
                "tp": 0,
                "fp": nombre_predictions,
                "fn": 0,
                "iou_max": None,
            }
        )

        continue

    # -----------------------------------------------------
    # Cas : annotation manuelle
    # -----------------------------------------------------

    if statut == "annote":
        nom_label = os.path.splitext(nom_image)[0] + ".txt"

        chemin_label = os.path.join(
            dossier_labels_manuels,
            nom_label,
        )

        boite_reelle = lire_label_manuel(
            chemin_label,
            largeur_image,
            hauteur_image,
        )

        if len(boites_predites) == 0:
            fn += 1

            details.append(
                {
                    "image": nom_image,
                    "statut": statut,
                    "tp": 0,
                    "fp": 0,
                    "fn": 1,
                    "iou_max": 0,
                }
            )

            continue

        ious = [
            calculer_iou(
                boite_reelle,
                boite_predite,
            )
            for boite_predite in boites_predites
        ]

        iou_max = max(ious)

        if iou_max >= seuil_iou:
            tp += 1

            fp += len(boites_predites) - 1

            details.append(
                {
                    "image": nom_image,
                    "statut": statut,
                    "tp": 1,
                    "fp": (len(boites_predites) - 1),
                    "fn": 0,
                    "iou_max": iou_max,
                }
            )

        else:
            fn += 1

            fp += len(boites_predites)

            details.append(
                {
                    "image": nom_image,
                    "statut": statut,
                    "tp": 0,
                    "fp": len(boites_predites),
                    "fn": 1,
                    "iou_max": iou_max,
                }
            )


# ---------------------------------------------------------
# Calcul des métriques
# ---------------------------------------------------------

if (tp + fp) > 0:
    precision = tp / (tp + fp)

else:
    precision = 0


if (tp + fn) > 0:
    rappel = tp / (tp + fn)

else:
    rappel = 0


# ---------------------------------------------------------
# Affichage
# ---------------------------------------------------------

print()
print("----------------------------------")
print("Evaluation validation manuelle")
print("----------------------------------")

print(
    "Seuil confiance :",
    seuil_confiance,
)

print(
    "Seuil IoU :",
    seuil_iou,
)

print()

print(
    "TP :",
    tp,
)

print(
    "FP :",
    fp,
)

print(
    "FN :",
    fn,
)

print()

print(
    "Precision :",
    round(
        precision,
        4,
    ),
)

print(
    "Rappel :",
    round(
        rappel,
        4,
    ),
)

print()

print(
    "Images ambiguës exclues :",
    nombre_ambigu,
)


# ---------------------------------------------------------
# Sauvegarde des détails
# ---------------------------------------------------------

table_details = pd.DataFrame(details)

chemin_details = "data/evaluation_validation_manuelle.csv"

table_details.to_csv(
    chemin_details,
    index=False,
)

print()
print(
    "Détails enregistrés :",
    chemin_details,
)

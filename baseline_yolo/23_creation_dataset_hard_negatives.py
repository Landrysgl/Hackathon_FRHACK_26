from pathlib import Path
import shutil
import csv


ROOT = Path(__file__).resolve().parent

CANDIDATES_DIR = (
    ROOT
    / "data"
    / "niveau2_error_analysis"
    / "hard_negative_train_candidates"
)

MAPPING_PATH = (
    CANDIDATES_DIR
    / "correspondance_numeros.txt"
)

CANDIDATES_CSV = (
    CANDIDATES_DIR
    / "candidats.csv"
)

BASE_DATASET = (
    ROOT
    / "data"
    / "dataset_niveau2_humain"
)

OUTPUT_DATASET = (
    ROOT
    / "data"
    / "dataset_niveau2_hardneg"
)

# Sélection manuelle validée
SELECTED_NUMBERS = {
    7,
    10,
    11,
    12,
    13,
    16,
    18,
    19,
    21,
}


# ============================================================
# LECTURE CORRESPONDANCE NUMERO -> CROP
# ============================================================

mapping = {}

for line in MAPPING_PATH.read_text(
    encoding="utf-8"
).splitlines():

    if not line.strip():
        continue

    number, filename = line.split(
        "\t",
        1
    )

    mapping[int(number)] = filename


missing_numbers = (
    SELECTED_NUMBERS
    - set(mapping)
)

if missing_numbers:
    raise RuntimeError(
        f"Numéros absents : {missing_numbers}"
    )


selected_crops = {
    mapping[n]
    for n in SELECTED_NUMBERS
}


# ============================================================
# RETROUVER LES IMAGES SOURCES
# ============================================================

with CANDIDATES_CSV.open(
    encoding="utf-8"
) as f:

    rows = list(
        csv.DictReader(f)
    )


selected_rows = [
    row
    for row in rows
    if row["crop"] in selected_crops
]


print(
    "Hard negatives sélectionnés :",
    len(selected_rows)
)


for row in selected_rows:

    print(
        " -",
        row["image_source"],
        "->",
        row["crop"]
    )


# ============================================================
# COPIE DU DATASET NIVEAU 2
# ============================================================

if OUTPUT_DATASET.exists():

    shutil.rmtree(
        OUTPUT_DATASET
    )


shutil.copytree(
    BASE_DATASET,
    OUTPUT_DATASET
)


train_images = (
    OUTPUT_DATASET
    / "images"
    / "train"
)

train_labels = (
    OUTPUT_DATASET
    / "labels"
    / "train"
)


# ============================================================
# IMPORTANT
#
# Les candidats viennent déjà d'images du train.
# On ne duplique PAS les crops comme nouvelles images.
#
# Pour le moment on crée un dataset expérimental
# identique et on sauvegarde la liste de hard negatives.
#
# L'étape suivante construira des crops négatifs propres
# si l'on décide de les injecter comme images indépendantes.
# ============================================================

selection_csv = (
    OUTPUT_DATASET
    / "hard_negatives_selection.csv"
)


with selection_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    if selected_rows:

        writer = csv.DictWriter(
            f,
            fieldnames=selected_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(
            selected_rows
        )


print("\n====================================")
print("SELECTION HARD NEGATIVES SAUVEGARDEE")
print("====================================")

print(
    "Nombre :",
    len(selected_rows)
)

print(
    "Dataset expérimental :",
    OUTPUT_DATASET
)

print(
    "Sélection :",
    selection_csv
)

print(
    "\n⚠️ Aucun crop n'a encore été ajouté "
    "à l'entraînement."
)

print(
    "✅ La validation reste strictement inchangée."
)
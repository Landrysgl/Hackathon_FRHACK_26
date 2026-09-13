from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parent

MANIFEST = (
    ROOT
    / "data"
    / "dataset_yolo"
    / "split_manifest.csv"
)

SUPPORTS = (
    ROOT
    / "data"
    / "supports_selectionnes_niveau1.csv"
)

LOT1_ANNOTATIONS = (
    ROOT
    / "data"
    / "niveau2_manual_train"
    / "annotations.csv"
)

SOURCE_IMAGES = (
    ROOT
    / "data"
    / "images_ign_niveau1_corrige"
)

OUT_DIR = (
    ROOT
    / "data"
    / "niveau2_manual_train_lot2"
)

OUT_IMAGES = OUT_DIR / "images"
OUT_CSV = OUT_DIR / "annotations.csv"


# ============================================================
# VERIFICATIONS
# ============================================================

for path in [
    MANIFEST,
    SUPPORTS,
    LOT1_ANNOTATIONS,
]:

    if not path.exists():
        raise FileNotFoundError(path)


if not SOURCE_IMAGES.exists():
    raise FileNotFoundError(SOURCE_IMAGES)


# ============================================================
# CHARGEMENT
# ============================================================

manifest = pd.read_csv(MANIFEST)
supports = pd.read_csv(SUPPORTS)
lot1 = pd.read_csv(LOT1_ANNOTATIONS)


print("===== DONNEES =====")

print(
    "Manifest total :",
    len(manifest)
)

print(
    "Lot 1 annoté :",
    len(lot1)
)


# ============================================================
# TRAIN UNIQUEMENT
# ============================================================

train = manifest[
    manifest["split"] == "train"
].copy()

# Le manifest Niveau 1 ne contient pas forcément
# le nom du fichier image.
# On le reconstruit à partir du SUP_ID.
if "image" not in train.columns:

    if "SUP_ID" not in train.columns:
        raise RuntimeError(
            "Ni 'image' ni 'SUP_ID' dans le manifest."
        )

    ids = pd.to_numeric(
        train["SUP_ID"],
        errors="raise"
    ).astype("int64")

    train["image"] = (
        "support_"
        + ids.astype(str)
        + ".jpg"
    )


print(
    "Images train Niveau 1 :",
    len(train)
)


# ============================================================
# EXCLUSION DU LOT 1
# ============================================================

already_annotated = set(
    lot1["image"].astype(str)
)

remaining = train[
    ~train["image"]
    .astype(str)
    .isin(already_annotated)
].copy()


print(
    "Déjà annotées dans train :",
    len(train) - len(remaining)
)

print(
    "Restantes à annoter :",
    len(remaining)
)


# On s'attend normalement à 130.
if len(remaining) != 130:

    print(
        "\n⚠️ ATTENTION : "
        "on attendait normalement 130 images."
    )

    print(
        "Le script continue, mais envoie-moi "
        "la sortie avant d'annoter."
    )


# ============================================================
# AJOUT DES METADONNEES ANFR
# ============================================================

# On garde le manifest comme référence de split,
# puis on récupère les informations détaillées
# depuis supports_selectionnes_niveau1.csv.

if "SUP_ID" not in remaining.columns:
    raise RuntimeError(
        "SUP_ID absent de split_manifest.csv"
    )

if "SUP_ID" not in supports.columns:
    raise RuntimeError(
        "SUP_ID absent de supports_selectionnes_niveau1.csv"
    )


# Evite les colonnes dupliquées après merge.
metadata_cols = [
    c
    for c in supports.columns
    if c not in remaining.columns
    or c == "SUP_ID"
]

remaining = remaining.merge(
    supports[metadata_cols],
    on="SUP_ID",
    how="left"
)


# ============================================================
# CLASSE NIVEAU 2
# ============================================================

def classify_support(nature):

    if pd.isna(nature):
        return "autre_support"

    s = str(nature).lower().strip()

    if (
        "pylône" in s
        or "pylone" in s
        or "tour hertzienne" in s
    ):
        return "pylone"

    if (
        "château d'eau" in s
        or "chateau d'eau" in s
        or "château eau" in s
        or "chateau eau" in s
    ):
        return "chateau_eau"

    if "mât" in s or "mat" in s:
        return "mat"

    if (
        "bâtiment" in s
        or "batiment" in s
        or "immeuble" in s
        or "local technique" in s
    ):
        return "batiment"

    if "silo" in s:
        return "silo"

    return "autre_support"


if "NAT_LB_NOM" not in remaining.columns:
    raise RuntimeError(
        "NAT_LB_NOM absent après fusion des métadonnées."
    )


remaining["classe_niveau2"] = (
    remaining["NAT_LB_NOM"]
    .apply(classify_support)
)


# ============================================================
# COLONNES D'ANNOTATION
# ============================================================

remaining["status"] = ""
remaining["x1"] = pd.NA
remaining["y1"] = pd.NA
remaining["x2"] = pd.NA
remaining["y2"] = pd.NA
remaining["commentaire"] = ""


# ============================================================
# MEME ORDRE DE COLONNES QUE LE LOT 1
# ============================================================

lot1_columns = list(lot1.columns)

for col in lot1_columns:

    if col not in remaining.columns:
        remaining[col] = pd.NA


remaining = remaining[
    lot1_columns
].copy()


# ============================================================
# PREPARATION DOSSIER
# ============================================================

if OUT_DIR.exists():

    shutil.rmtree(
        OUT_DIR
    )


OUT_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)


missing_images = []


for image_name in remaining[
    "image"
].astype(str):

    src = SOURCE_IMAGES / image_name
    dst = OUT_IMAGES / image_name

    if not src.exists():

        missing_images.append(
            image_name
        )

        continue

    shutil.copy2(
        src,
        dst
    )


if missing_images:

    print(
        "\n❌ Images sources manquantes :",
        len(missing_images)
    )

    for name in missing_images[:20]:
        print(" -", name)

    raise RuntimeError(
        "Des images sont manquantes. "
        "Ne commence pas l'annotation."
    )


# ============================================================
# SAUVEGARDE CSV
# ============================================================

remaining.to_csv(
    OUT_CSV,
    index=False
)


# ============================================================
# AUDIT
# ============================================================

copied = len(
    list(
        OUT_IMAGES.glob(
            "support_*.jpg"
        )
    )
)


print("\n====================================")
print("LOT 2 PREPARE")
print("====================================")

print(
    "Images attendues :",
    len(remaining)
)

print(
    "Images copiées :",
    copied
)

print(
    "Images manquantes :",
    len(missing_images)
)

print("\nRépartition ANFR indicative :")

print(
    remaining[
        "classe_niveau2"
    ].value_counts()
)

print(
    "\nCSV :",
    OUT_CSV
)

print(
    "Images :",
    OUT_IMAGES
)

print(
    "\n✅ Seulement des images du TRAIN Niveau 1."
)

print(
    "✅ Aucune des 80 images du lot 1."
)
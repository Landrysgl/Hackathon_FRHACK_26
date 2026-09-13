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

TEMPLATE_ANNOTATIONS = (
    ROOT
    / "data"
    / "niveau2_manual_train"
    / "annotations.csv"
)

LOT1_ANNOTATIONS = (
    ROOT
    / "data"
    / "niveau2_manual_train"
    / "annotations.csv"
)

LOT2_ANNOTATIONS = (
    ROOT
    / "data"
    / "niveau2_manual_train_lot2"
    / "annotations.csv"
)

MANUAL_TEST = (
    ROOT
    / "data"
    / "manual_validation"
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
    / "niveau2_manual_val"
)

OUT_IMAGES = OUT_DIR / "images"
OUT_CSV = OUT_DIR / "annotations.csv"


# ============================================================
# VERIFICATIONS
# ============================================================

for path in [
    MANIFEST,
    SUPPORTS,
    TEMPLATE_ANNOTATIONS,
    LOT1_ANNOTATIONS,
    LOT2_ANNOTATIONS,
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
template = pd.read_csv(TEMPLATE_ANNOTATIONS)
lot1 = pd.read_csv(LOT1_ANNOTATIONS)
lot2 = pd.read_csv(LOT2_ANNOTATIONS)


print("===== DONNEES =====")

print(
    "Manifest total :",
    len(manifest)
)

print(
    "Train annoté lot 1 :",
    len(lot1)
)

print(
    "Train annoté lot 2 :",
    len(lot2)
)


# ============================================================
# VAL UNIQUEMENT
# ============================================================

val = manifest[
    manifest["split"] == "val"
].copy()


# Le manifest ne contient pas forcément la colonne image.
if "image" not in val.columns:

    if "SUP_ID" not in val.columns:
        raise RuntimeError(
            "Ni 'image' ni 'SUP_ID' dans le manifest."
        )

    ids = pd.to_numeric(
        val["SUP_ID"],
        errors="raise"
    ).astype("int64")

    val["image"] = (
        "support_"
        + ids.astype(str)
        + ".jpg"
    )


print(
    "Images VAL Niveau 1 :",
    len(val)
)


if len(val) != 45:

    print(
        "\n⚠️ On attend normalement 45 images VAL."
    )


# ============================================================
# VERIFICATION ABSENCE DE FUITE TRAIN
# ============================================================

train_annotated = set(
    lot1["image"].astype(str)
) | set(
    lot2["image"].astype(str)
)


overlap_train = set(
    val["image"].astype(str)
) & train_annotated


print(
    "Chevauchement avec TRAIN annoté :",
    len(overlap_train)
)


if overlap_train:

    raise RuntimeError(
        "Fuite détectée entre train et val."
    )


# ============================================================
# VERIFICATION ABSENCE DE FUITE TEST HUMAIN
# ============================================================

if MANUAL_TEST.exists():

    manual_test = pd.read_csv(
        MANUAL_TEST
    )

    test_names = set(
        manual_test["image"].astype(str)
    )

    overlap_test = set(
        val["image"].astype(str)
    ) & test_names

    print(
        "Chevauchement avec TEST humain :",
        len(overlap_test)
    )

    if overlap_test:

        raise RuntimeError(
            "Fuite détectée entre val et test humain."
        )

else:

    print(
        "Test humain : annotations.csv absent, "
        "contrôle ignoré."
    )


# ============================================================
# AJOUT DES METADONNEES ANFR
# ============================================================

if "SUP_ID" not in supports.columns:
    raise RuntimeError(
        "SUP_ID absent de supports_selectionnes_niveau1.csv"
    )


metadata_cols = [
    c
    for c in supports.columns
    if c not in val.columns
    or c == "SUP_ID"
]


val = val.merge(
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


if "NAT_LB_NOM" not in val.columns:

    raise RuntimeError(
        "NAT_LB_NOM absent après fusion."
    )


val["classe_niveau2"] = (
    val["NAT_LB_NOM"]
    .apply(classify_support)
)


# ============================================================
# COLONNES D'ANNOTATION
# ============================================================

# Important : object pour éviter le bug pandas float64.
val["status"] = pd.Series(
    [""] * len(val),
    dtype="object"
)

val["x1"] = pd.NA
val["y1"] = pd.NA
val["x2"] = pd.NA
val["y2"] = pd.NA

val["commentaire"] = pd.Series(
    [""] * len(val),
    dtype="object"
)


# ============================================================
# MEME FORMAT QUE LES AUTRES ANNOTATIONS
# ============================================================

template_columns = list(
    template.columns
)

for col in template_columns:

    if col not in val.columns:
        val[col] = pd.NA


val = val[
    template_columns
].copy()


# ============================================================
# PREPARATION DU DOSSIER
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


for image_name in val[
    "image"
].astype(str):

    src = (
        SOURCE_IMAGES
        / image_name
    )

    dst = (
        OUT_IMAGES
        / image_name
    )

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
        "\n❌ Images manquantes :",
        len(missing_images)
    )

    for name in missing_images:
        print(" -", name)

    raise RuntimeError(
        "Ne commence pas l'annotation."
    )


# ============================================================
# SAUVEGARDE
# ============================================================

val.to_csv(
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
print("VALIDATION NIVEAU 2 PREPAREE")
print("====================================")

print(
    "Images attendues :",
    len(val)
)

print(
    "Images copiées :",
    copied
)

print(
    "Images manquantes :",
    len(missing_images)
)

print(
    "Chevauchement train :",
    len(overlap_train)
)

if MANUAL_TEST.exists():

    print(
        "Chevauchement test :",
        len(overlap_test)
    )


print("\nRépartition ANFR indicative :")

print(
    val[
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
    "\n✅ Seulement le split VAL du Niveau 1."
)

print(
    "✅ Aucun chevauchement avec le TRAIN annoté."
)

if MANUAL_TEST.exists():

    print(
        "✅ Aucun chevauchement avec le TEST humain."
    )
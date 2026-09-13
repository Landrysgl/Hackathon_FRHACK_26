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

MANUAL_VAL = (
    ROOT
    / "data"
    / "niveau2_manual_val"
    / "annotations.csv"
)

OLD_MANUAL_TEST = (
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
    / "niveau2_manual_holdout_final"
)

OUT_IMAGES = OUT_DIR / "images"
OUT_CSV = OUT_DIR / "annotations.csv"


# ============================================================
# VERIFICATIONS
# ============================================================

required = [
    MANIFEST,
    SUPPORTS,
    TEMPLATE_ANNOTATIONS,
    LOT1_ANNOTATIONS,
    LOT2_ANNOTATIONS,
    MANUAL_VAL,
    OLD_MANUAL_TEST,
]

for path in required:
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

manual_val = pd.read_csv(MANUAL_VAL)
old_test = pd.read_csv(OLD_MANUAL_TEST)


print("===== DONNEES =====")

print("Manifest total :", len(manifest))
print("Train lot 1 :", len(lot1))
print("Train lot 2 :", len(lot2))
print("Validation humaine :", len(manual_val))
print("Ancien test humain :", len(old_test))


# ============================================================
# SPLIT TEST NIVEAU 1
# ============================================================

test = manifest[
    manifest["split"] == "test"
].copy()


if "image" not in test.columns:

    if "SUP_ID" not in test.columns:
        raise RuntimeError(
            "Ni 'image' ni 'SUP_ID' dans le manifest."
        )

    ids = pd.to_numeric(
        test["SUP_ID"],
        errors="raise"
    ).astype("int64")

    test["image"] = (
        "support_"
        + ids.astype(str)
        + ".jpg"
    )


print(
    "Images TEST Niveau 1 :",
    len(test)
)


if len(test) != 45:
    raise RuntimeError(
        f"On attend 45 images TEST, trouvé {len(test)}."
    )


# ============================================================
# RETIRER LES 20 IMAGES DEJA UTILISEES
# ============================================================

test_names = set(
    test["image"].astype(str)
)

old_test_names = set(
    old_test["image"].astype(str)
)


outside_test = (
    old_test_names
    - test_names
)

if outside_test:

    raise RuntimeError(
        "Certaines images de l'ancien test humain "
        "ne sont pas dans le split TEST."
    )


holdout = test[
    ~test["image"]
    .astype(str)
    .isin(old_test_names)
].copy()


print(
    "Images ancien test retirées :",
    len(old_test_names)
)

print(
    "Images HOLDOUT restantes :",
    len(holdout)
)


if len(old_test_names) != 20:
    raise RuntimeError(
        f"Ancien test humain attendu : 20, trouvé {len(old_test_names)}."
    )


if len(holdout) != 25:
    raise RuntimeError(
        f"Holdout attendu : 25, trouvé {len(holdout)}."
    )


# ============================================================
# VERIFICATION DES CHEVAUCHEMENTS
# ============================================================

train_names = (
    set(lot1["image"].astype(str))
    | set(lot2["image"].astype(str))
)

val_names = set(
    manual_val["image"].astype(str)
)

holdout_names = set(
    holdout["image"].astype(str)
)


overlap_train = (
    holdout_names
    & train_names
)

overlap_val = (
    holdout_names
    & val_names
)

overlap_old_test = (
    holdout_names
    & old_test_names
)


print(
    "Chevauchement HOLDOUT / TRAIN :",
    len(overlap_train)
)

print(
    "Chevauchement HOLDOUT / VAL :",
    len(overlap_val)
)

print(
    "Chevauchement HOLDOUT / ancien TEST :",
    len(overlap_old_test)
)


if overlap_train:
    raise RuntimeError(
        "Fuite HOLDOUT / TRAIN."
    )

if overlap_val:
    raise RuntimeError(
        "Fuite HOLDOUT / VAL."
    )

if overlap_old_test:
    raise RuntimeError(
        "Fuite HOLDOUT / ancien TEST."
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
    if c not in holdout.columns
    or c == "SUP_ID"
]


holdout = holdout.merge(
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


if "NAT_LB_NOM" not in holdout.columns:
    raise RuntimeError(
        "NAT_LB_NOM absent après fusion."
    )


holdout["classe_niveau2"] = (
    holdout["NAT_LB_NOM"]
    .apply(classify_support)
)


# ============================================================
# COLONNES D'ANNOTATION
# ============================================================

holdout["status"] = pd.Series(
    [""] * len(holdout),
    dtype="object"
)

holdout["x1"] = pd.NA
holdout["y1"] = pd.NA
holdout["x2"] = pd.NA
holdout["y2"] = pd.NA

holdout["commentaire"] = pd.Series(
    [""] * len(holdout),
    dtype="object"
)


# ============================================================
# MEME FORMAT QUE LES AUTRES ANNOTATIONS
# ============================================================

template_columns = list(
    template.columns
)

for col in template_columns:

    if col not in holdout.columns:
        holdout[col] = pd.NA


holdout = holdout[
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


for image_name in holdout[
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

holdout.to_csv(
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
print("HOLDOUT FINAL NIVEAU 2 PREPARE")
print("====================================")

print(
    "Images TEST Niveau 1 :",
    len(test)
)

print(
    "Ancien test retiré :",
    len(old_test_names)
)

print(
    "Images attendues HOLDOUT :",
    len(holdout)
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

print(
    "Chevauchement val :",
    len(overlap_val)
)

print(
    "Chevauchement ancien test :",
    len(overlap_old_test)
)


print("\nRépartition ANFR indicative :")

print(
    holdout[
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
    "\n✅ 25 images jamais utilisées auparavant."
)

print(
    "✅ Aucun chevauchement avec TRAIN."
)

print(
    "✅ Aucun chevauchement avec VAL."
)

print(
    "✅ Aucun chevauchement avec l'ancien TEST humain."
)

print(
    "\n⚠️ Ne jamais utiliser ce HOLDOUT "
    "pour choisir ou régler le modèle."
)
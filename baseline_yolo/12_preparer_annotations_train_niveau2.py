from pathlib import Path
import shutil
import unicodedata

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MANIFEST_PATH = DATA_DIR / "dataset_yolo" / "split_manifest.csv"

# Dataset corrigé régénéré après la suppression accidentelle
SOURCE_IMAGES_DIR = DATA_DIR / "images_ign_niveau1_corrige"

OUT_DIR = DATA_DIR / "niveau2_manual_train"
OUT_IMAGES_DIR = OUT_DIR / "images"

SELECTION_PATH = OUT_DIR / "selection_train_niveau2.csv"
ANNOTATIONS_PATH = OUT_DIR / "annotations.csv"

TARGET_N = 80
SEED = 42

# Répartition destinée à obtenir un jeu varié.
QUOTAS = {
    "pylone": 26,
    "batiment": 26,
    "mat": 8,
    "chateau_eau": 8,
    "silo": 4,
    "autre_support": 8,
}


# ============================================================
# FONCTIONS
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    return "".join(
        c
        for c in unicodedata.normalize("NFD", value)
        if unicodedata.category(c) != "Mn"
    )


def classe_niveau2(nature):
    """
    Regroupe les nombreuses natures ANFR en catégories
    plus simples pour les expériences Niveau 2.
    """

    n = normalize_text(nature)

    if "chateau d'eau" in n:
        return "chateau_eau"

    if "silo" in n:
        return "silo"

    if (
        n == "mat"
        or n.startswith("mat ")
        or "mat beton" in n
        or "mat metallique" in n
    ):
        return "mat"

    if "pylone" in n or "tour hertzienne" in n:
        return "pylone"

    if (
        "batiment" in n
        or "immeuble" in n
        or "local technique" in n
    ):
        return "batiment"

    return "autre_support"


def image_path(sup_id):
    return SOURCE_IMAGES_DIR / f"support_{int(sup_id)}.jpg"


# ============================================================
# VERIFICATIONS
# ============================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest introuvable : {MANIFEST_PATH}"
    )

if not SOURCE_IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Dossier des 300 images corrigées introuvable : "
        f"{SOURCE_IMAGES_DIR}"
    )


df = pd.read_csv(MANIFEST_PATH)

required = {
    "SUP_ID",
    "NAT_LB_NOM",
    "latitude",
    "longitude",
    "split",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Colonnes manquantes : {sorted(missing)}"
    )


# ============================================================
# UNIQUEMENT LE TRAIN
# ============================================================

train = df[
    df["split"].astype(str).str.lower() == "train"
].copy()

print("===== SPLIT TRAIN =====")
print("Images train :", len(train))
print("SUP_ID uniques :", train["SUP_ID"].nunique())


if train["SUP_ID"].duplicated().any():
    raise RuntimeError(
        "Des SUP_ID sont dupliqués dans le train."
    )


# ============================================================
# CLASSES SIMPLIFIEES
# ============================================================

train["classe_niveau2"] = train["NAT_LB_NOM"].apply(
    classe_niveau2
)

print("\n===== REPARTITION DISPONIBLE =====")
print(
    train["classe_niveau2"]
    .value_counts()
    .reindex(QUOTAS.keys(), fill_value=0)
)


# ============================================================
# ECHANTILLONNAGE STRATIFIE
# ============================================================

selected_parts = []

print("\n===== SELECTION =====")

for classe, quota in QUOTAS.items():

    group = train[
        train["classe_niveau2"] == classe
    ].copy()

    n_take = min(quota, len(group))

    if n_take > 0:
        sample = group.sample(
            n=n_take,
            random_state=SEED
        )

        selected_parts.append(sample)

    print(
        f"{classe:15s} : "
        f"{n_take:2d} sélectionnées "
        f"/ {len(group):3d} disponibles"
    )


selected = pd.concat(
    selected_parts,
    ignore_index=False
)


# ============================================================
# COMPLEMENT SI UNE CLASSE N'AVAIT PAS ASSEZ D'IMAGES
# ============================================================

selected_ids = set(
    selected["SUP_ID"].astype(str)
)

missing_n = TARGET_N - len(selected)

if missing_n > 0:

    remaining = train[
        ~train["SUP_ID"].astype(str).isin(selected_ids)
    ]

    extra = remaining.sample(
        n=missing_n,
        random_state=SEED + 1
    )

    selected = pd.concat(
        [selected, extra],
        ignore_index=False
    )


if len(selected) > TARGET_N:
    selected = selected.sample(
        n=TARGET_N,
        random_state=SEED + 2
    )


# Ordre aléatoire mais reproductible pour l'annotation
selected = selected.sample(
    frac=1,
    random_state=SEED + 3
).reset_index(drop=True)


# ============================================================
# CONTROLES
# ============================================================

if len(selected) != TARGET_N:
    raise RuntimeError(
        f"Sélection incorrecte : {len(selected)} "
        f"au lieu de {TARGET_N}"
    )

if selected["SUP_ID"].duplicated().any():
    raise RuntimeError(
        "Doublons dans la sélection Niveau 2."
    )

if set(selected["split"].unique()) != {"train"}:
    raise RuntimeError(
        "ERREUR : une image val/test a été sélectionnée."
    )


# ============================================================
# VERIFICATION DES 80 IMAGES
# ============================================================

missing_images = []

for sup_id in selected["SUP_ID"]:
    if not image_path(sup_id).exists():
        missing_images.append(
            image_path(sup_id)
        )


if missing_images:

    print("\nImages manquantes :")

    for p in missing_images[:20]:
        print(" -", p)

    raise RuntimeError(
        f"{len(missing_images)} image(s) manquante(s)."
    )


selected["image"] = selected["SUP_ID"].apply(
    lambda x: f"support_{int(x)}.jpg"
)


# ============================================================
# CREATION DU DOSSIER
# ============================================================

OUT_IMAGES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Supprime uniquement les anciennes copies Niveau 2
expected = set(selected["image"])

for old_image in OUT_IMAGES_DIR.glob("*.jpg"):
    if old_image.name not in expected:
        old_image.unlink()


# Copie les images
for _, row in selected.iterrows():

    src = SOURCE_IMAGES_DIR / row["image"]
    dst = OUT_IMAGES_DIR / row["image"]

    if not dst.exists():
        shutil.copy2(src, dst)


# ============================================================
# SAUVEGARDE SELECTION
# ============================================================

selected.to_csv(
    SELECTION_PATH,
    index=False
)


# ============================================================
# CSV D'ANNOTATION
# ============================================================

annotations = selected.copy()

annotations["status"] = ""
annotations["x1"] = pd.NA
annotations["y1"] = pd.NA
annotations["x2"] = pd.NA
annotations["y2"] = pd.NA
annotations["commentaire"] = ""


# Si le script est relancé, on préserve le travail humain.
if ANNOTATIONS_PATH.exists():

    old = pd.read_csv(ANNOTATIONS_PATH)

    if "SUP_ID" in old.columns:

        old_cols = [
            c for c in [
                "SUP_ID",
                "status",
                "x1",
                "y1",
                "x2",
                "y2",
                "commentaire",
            ]
            if c in old.columns
        ]

        old = old[old_cols].copy()

        annotations = annotations.drop(
            columns=[
                "status",
                "x1",
                "y1",
                "x2",
                "y2",
                "commentaire",
            ]
        )

        annotations = annotations.merge(
            old,
            on="SUP_ID",
            how="left"
        )

        for col in [
            "status",
            "x1",
            "y1",
            "x2",
            "y2",
            "commentaire",
        ]:
            if col not in annotations.columns:
                annotations[col] = ""

        print(
            "\n✅ Les annotations existantes ont été préservées."
        )


annotations.to_csv(
    ANNOTATIONS_PATH,
    index=False
)


# ============================================================
# RESUME
# ============================================================

print("\n===================================")
print("NIVEAU 2 - DATASET A ANNOTER PRET")
print("===================================")

print("Images :", len(selected))
print(
    "SUP_ID uniques :",
    selected["SUP_ID"].nunique()
)

print("\nRépartition finale :")
print(
    selected["classe_niveau2"]
    .value_counts()
)

print("\nPrincipales natures ANFR :")
print(
    selected["NAT_LB_NOM"]
    .value_counts()
    .head(15)
)

print("\nImages :")
print(OUT_IMAGES_DIR)

print("\nAnnotations :")
print(ANNOTATIONS_PATH)

print(
    "\n✅ 100 % des images proviennent du train."
)

print(
    "✅ Les images val/test du Niveau 1 restent intactes."
)
from pathlib import Path
import random
import shutil

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

DATASET_DIR = DATA_DIR / "dataset_yolo"
MANIFEST_PATH = DATASET_DIR / "split_manifest.csv"
SOURCE_IMAGES_DIR = DATA_DIR / "images_ign_yvelines"

MANUAL_DIR = DATA_DIR / "manual_validation"
MANUAL_IMAGES_DIR = MANUAL_DIR / "images"
PREVIEWS_DIR = MANUAL_DIR / "previews"
ANNOTATIONS_PATH = MANUAL_DIR / "annotations.csv"

N_MANUAL = 20
RANDOM_SEED = 42
IMAGE_SIZE = 1024
WEAK_BOX_SIZE_PX = 160

for d in (MANUAL_IMAGES_DIR, PREVIEWS_DIR):
    d.mkdir(parents=True, exist_ok=True)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"{MANIFEST_PATH} introuvable. Lance d'abord le script 06."
    )

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"SUP_ID": "string"},
)

test = (
    manifest[manifest["split"] == "test"]
    .sort_values("SUP_ID")
    .reset_index(drop=True)
)

if len(test) < N_MANUAL:
    raise ValueError(
        f"Le split test ne contient que {len(test)} images, "
        f"impossible d'en choisir {N_MANUAL}."
    )

types = (
    test["NAT_LB_NOM"]
    .value_counts()
    .head(min(10, test["NAT_LB_NOM"].nunique()))
    .index
)

choisis = []
ids_choisis = set()

for i, nature in enumerate(types):
    groupe = test[test["NAT_LB_NOM"] == nature]
    row = groupe.sample(
        n=1,
        random_state=RANDOM_SEED + i,
    ).iloc[0]
    choisis.append(row)
    ids_choisis.add(row["SUP_ID"])

reste = test[~test["SUP_ID"].isin(ids_choisis)]

if len(choisis) < N_MANUAL:
    complement = reste.sample(
        n=N_MANUAL - len(choisis),
        random_state=RANDOM_SEED,
    )
    choisis.extend(
        row
        for _, row in complement.iterrows()
    )

selection = pd.DataFrame(choisis)
selection = (
    selection
    .drop_duplicates("SUP_ID")
    .head(N_MANUAL)
    .sort_values("SUP_ID")
    .reset_index(drop=True)
)

if len(selection) != N_MANUAL:
    raise RuntimeError(
        f"Seulement {len(selection)} images ont été sélectionnées."
    )

# Nettoyer seulement les images/previews qui ne font plus partie de la sélection.
noms_attendus = {
    f"support_{sid}.jpg"
    for sid in selection["SUP_ID"]
}

for d in (MANUAL_IMAGES_DIR, PREVIEWS_DIR):
    for p in d.glob("support_*.jpg"):
        if p.name not in noms_attendus:
            p.unlink()

for row in selection.itertuples(index=False):
    sup_id = str(row.SUP_ID)
    src = SOURCE_IMAGES_DIR / f"support_{sup_id}.jpg"
    dst = MANUAL_IMAGES_DIR / src.name

    if not src.exists():
        raise FileNotFoundError(src)

    shutil.copy2(src, dst)

    with Image.open(src) as im:
        preview = im.convert("RGB")

    draw = ImageDraw.Draw(preview)
    c = IMAGE_SIZE // 2
    h = WEAK_BOX_SIZE_PX // 2

    draw.rectangle(
        (c - h, c - h, c + h, c + h),
        outline="red",
        width=4,
    )
    draw.line((c - 20, c, c + 20, c), fill="yellow", width=3)
    draw.line((c, c - 20, c, c + 20), fill="yellow", width=3)

    preview.save(PREVIEWS_DIR / src.name, quality=95)

base = selection[
    [
        "SUP_ID",
        "NAT_LB_NOM",
        "latitude",
        "longitude",
    ]
].copy()

base["image"] = base["SUP_ID"].map(lambda x: f"support_{x}.jpg")
base["status"] = ""
base["x1"] = ""
base["y1"] = ""
base["x2"] = ""
base["y2"] = ""
base["commentaire"] = ""

# Préserver les annotations déjà réalisées si on relance le script.
if ANNOTATIONS_PATH.exists():
    ancien = pd.read_csv(
        ANNOTATIONS_PATH,
        dtype={"SUP_ID": "string"},
    )

    colonnes_annotation = [
        "SUP_ID",
        "status",
        "x1",
        "y1",
        "x2",
        "y2",
        "commentaire",
    ]

    ancien = ancien[
        [c for c in colonnes_annotation if c in ancien.columns]
    ]

    base = base.drop(
        columns=[
            "status",
            "x1",
            "y1",
            "x2",
            "y2",
            "commentaire",
        ]
    ).merge(
        ancien,
        on="SUP_ID",
        how="left",
    )

    for col in [
        "status",
        "x1",
        "y1",
        "x2",
        "y2",
        "commentaire",
    ]:
        if col not in base.columns:
            base[col] = ""

base.to_csv(ANNOTATIONS_PATH, index=False)

print("===== VALIDATION MANUELLE PREPAREE =====")
print("Images :", len(base))
print("Toutes proviennent du split test :", True)
print("Dossier images :", MANUAL_IMAGES_DIR)
print("Previews :", PREVIEWS_DIR)
print("CSV :", ANNOTATIONS_PATH)

print("\nRègles d'annotation :")
print("- visible     : dessiner une bbox humaine autour du support cible")
print("- non_visible : aucun objet GT ; toutes les bbox restent vides")
print("- ambigu      : image exclue des métriques finales")
print("\nOuvre maintenant 07_annotation_manuelle.ipynb.")

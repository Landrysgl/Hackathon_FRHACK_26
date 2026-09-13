from pathlib import Path
import shutil
import random

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

ANNOTATIONS = DATA / "niveau2_manual_train" / "annotations.csv"
SOURCE_IMAGES = DATA / "niveau2_manual_train" / "images"

OUT = DATA / "dataset_niveau2_humain"

SEED = 42
VAL_RATIO = 0.20


# ============================================================
# CHARGEMENT
# ============================================================

df = pd.read_csv(ANNOTATIONS)

df["status"] = (
    df["status"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

if (df["status"] == "").any():
    raise RuntimeError(
        f"{(df['status'] == '').sum()} images ne sont pas annotées."
    )

invalid = ~df["status"].isin(
    ["visible", "non_visible", "ambigu"]
)

if invalid.any():
    raise RuntimeError(
        f"Statuts invalides : {df.loc[invalid, 'status'].unique()}"
    )


# ============================================================
# ON EXCLUT LES AMBIGUS
# ============================================================

usable = df[
    df["status"].isin(["visible", "non_visible"])
].copy()

print("===== ANNOTATIONS =====")
print("Total :", len(df))
print("Utilisables :", len(usable))
print("Visible :", (usable["status"] == "visible").sum())
print("Non visible :", (usable["status"] == "non_visible").sum())
print("Ambigu exclus :", (df["status"] == "ambigu").sum())


# ============================================================
# VERIFICATION BBOX
# ============================================================

visible = usable["status"] == "visible"

bbox_cols = ["x1", "y1", "x2", "y2"]

if usable.loc[visible, bbox_cols].isna().any().any():
    raise RuntimeError(
        "Une image visible possède une bbox incomplète."
    )

for idx, row in usable.loc[visible].iterrows():

    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])

    if not (
        0 <= x1 < x2 <= 1024
        and 0 <= y1 < y2 <= 1024
    ):
        raise RuntimeError(
            f"Bbox invalide pour SUP_ID {row['SUP_ID']} : "
            f"{x1}, {y1}, {x2}, {y2}"
        )


# ============================================================
# SPLIT TRAIN / VAL
# ============================================================

random.seed(SEED)

positive = usable[
    usable["status"] == "visible"
].index.tolist()

negative = usable[
    usable["status"] == "non_visible"
].index.tolist()

random.shuffle(positive)
random.shuffle(negative)

n_val_pos = max(1, round(len(positive) * VAL_RATIO))

n_val_neg = (
    max(1, round(len(negative) * VAL_RATIO))
    if len(negative) > 1
    else 0
)

val_idx = set(
    positive[:n_val_pos]
    + negative[:n_val_neg]
)

usable["split_niveau2"] = usable.index.map(
    lambda i: "val" if i in val_idx else "train"
)


print("\n===== SPLIT NIVEAU 2 =====")
print(
    usable.groupby(
        ["split_niveau2", "status"]
    ).size()
)


# ============================================================
# CREATION DOSSIERS
# ============================================================

if OUT.exists():
    shutil.rmtree(OUT)

for split in ["train", "val"]:

    (OUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COPIE + LABELS
# ============================================================

for _, row in usable.iterrows():

    split = row["split_niveau2"]

    image_name = str(row["image"])

    src_img = SOURCE_IMAGES / image_name
    dst_img = OUT / "images" / split / image_name

    if not src_img.exists():
        raise FileNotFoundError(src_img)

    shutil.copy2(src_img, dst_img)

    label_path = (
        OUT
        / "labels"
        / split
        / (Path(image_name).stem + ".txt")
    )

    if row["status"] == "non_visible":

        # Vrai négatif YOLO
        label_path.write_text(
            "",
            encoding="utf-8"
        )

        continue

    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])

    xc = ((x1 + x2) / 2) / 1024
    yc = ((y1 + y2) / 2) / 1024

    w = (x2 - x1) / 1024
    h = (y2 - y1) / 1024

    label = (
        f"0 {xc:.6f} {yc:.6f} "
        f"{w:.6f} {h:.6f}\n"
    )

    label_path.write_text(
        label,
        encoding="utf-8"
    )


# ============================================================
# YAML
# ============================================================

yaml_text = f"""path: {OUT}
train: images/train
val: images/val

names:
  0: support
"""

(OUT / "dataset.yaml").write_text(
    yaml_text,
    encoding="utf-8"
)


# ============================================================
# MANIFEST
# ============================================================

usable.to_csv(
    OUT / "manifest_niveau2.csv",
    index=False
)


# ============================================================
# AUDIT
# ============================================================

print("\n===== DATASET CREE =====")

for split in ["train", "val"]:

    images = list(
        (OUT / "images" / split).glob("*.jpg")
    )

    labels = list(
        (OUT / "labels" / split).glob("*.txt")
    )

    print(
        f"{split}: "
        f"{len(images)} images / "
        f"{len(labels)} labels"
    )


print("\nDataset :", OUT)
print("YAML :", OUT / "dataset.yaml")

print(
    "\n✅ Labels faibles supprimés."
)

print(
    "✅ Les bbox utilisées sont uniquement "
    "tes annotations humaines."
)

print(
    "✅ Les non_visible deviennent de vrais négatifs."
)

print(
    "✅ Les ambigu sont exclus."
)
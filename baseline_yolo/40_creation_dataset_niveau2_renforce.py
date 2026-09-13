from pathlib import Path
import shutil

import pandas as pd
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parent

LOT1 = (
    ROOT
    / "data"
    / "niveau2_manual_train"
    / "annotations.csv"
)

LOT2 = (
    ROOT
    / "data"
    / "niveau2_manual_train_lot2"
    / "annotations.csv"
)

VAL = (
    ROOT
    / "data"
    / "niveau2_manual_val"
    / "annotations.csv"
)

TEST = (
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

OUT = (
    ROOT
    / "data"
    / "dataset_niveau2_renforce"
)

TRAIN_IMAGES = OUT / "images" / "train"
TRAIN_LABELS = OUT / "labels" / "train"

VAL_IMAGES = OUT / "images" / "val"
VAL_LABELS = OUT / "labels" / "val"

YAML_PATH = OUT / "dataset.yaml"

MERGED_TRAIN_CSV = (
    OUT
    / "annotations_train_fusionnees.csv"
)

VAL_CSV = (
    OUT
    / "annotations_val_humaines.csv"
)


# ============================================================
# CHARGEMENT
# ============================================================

for p in [LOT1, LOT2, VAL]:

    if not p.exists():
        raise FileNotFoundError(p)


lot1 = pd.read_csv(LOT1)
lot2 = pd.read_csv(LOT2)
val = pd.read_csv(VAL)


train = pd.concat(
    [lot1, lot2],
    ignore_index=True
)


print("====================================")
print("DATASET NIVEAU 2 RENFORCE")
print("====================================")

print(
    "Train brut :",
    len(train)
)

print(
    "Val brute :",
    len(val)
)


# ============================================================
# AUDIT DOUBLONS
# ============================================================

if train["image"].duplicated().any():

    dup = train.loc[
        train["image"].duplicated(
            keep=False
        ),
        "image"
    ].tolist()

    raise RuntimeError(
        f"Doublons dans train : {dup}"
    )


if val["image"].duplicated().any():

    raise RuntimeError(
        "Doublons dans validation."
    )


train_names = set(
    train["image"].astype(str)
)

val_names = set(
    val["image"].astype(str)
)


overlap_train_val = (
    train_names
    & val_names
)


print(
    "Chevauchement train/val :",
    len(overlap_train_val)
)


if overlap_train_val:

    raise RuntimeError(
        "Fuite train/val détectée."
    )


# ============================================================
# TEST GELE
# ============================================================

if TEST.exists():

    test = pd.read_csv(TEST)

    test_names = set(
        test["image"].astype(str)
    )

    overlap_train_test = (
        train_names
        & test_names
    )

    overlap_val_test = (
        val_names
        & test_names
    )

    print(
        "Chevauchement train/test :",
        len(overlap_train_test)
    )

    print(
        "Chevauchement val/test :",
        len(overlap_val_test)
    )

    if (
        overlap_train_test
        or overlap_val_test
    ):

        raise RuntimeError(
            "Fuite avec le test humain."
        )


# ============================================================
# STATUTS VALIDES
# ============================================================

train_usable = train[
    train["status"].isin(
        ["visible", "non_visible"]
    )
].copy()

val_usable = val[
    val["status"].isin(
        ["visible", "non_visible"]
    )
].copy()


print("\n===== TRAIN =====")

print(
    train["status"]
    .value_counts()
)

print(
    "Utilisables :",
    len(train_usable)
)


print("\n===== VAL =====")

print(
    val["status"]
    .value_counts()
)

print(
    "Utilisables :",
    len(val_usable)
)


# ============================================================
# VERIFICATION BBOX
# ============================================================

for name, df in [
    ("train", train_usable),
    ("val", val_usable),
]:

    visible = df[
        df["status"] == "visible"
    ]

    incomplete = visible[
        ["x1", "y1", "x2", "y2"]
    ].isna().any(axis=1)

    if incomplete.any():

        bad = visible.loc[
            incomplete,
            "image"
        ].tolist()

        raise RuntimeError(
            f"Bbox incomplètes {name}: {bad}"
        )


# ============================================================
# RECREATION DOSSIER
# ============================================================

if OUT.exists():

    shutil.rmtree(OUT)


for p in [
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VAL_IMAGES,
    VAL_LABELS,
]:

    p.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# ECRITURE YOLO
# ============================================================

def export_split(
    dataframe,
    images_dir,
    labels_dir,
):

    positives = 0
    negatives = 0

    for _, row in dataframe.iterrows():

        image_name = str(
            row["image"]
        )

        src = (
            SOURCE_IMAGES
            / image_name
        )

        if not src.exists():

            raise FileNotFoundError(
                src
            )


        dst = (
            images_dir
            / image_name
        )

        shutil.copy2(
            src,
            dst
        )


        label_path = (
            labels_dir
            / (
                Path(image_name).stem
                + ".txt"
            )
        )


        # ----------------------------------------
        # NEGATIF
        # ----------------------------------------

        if row["status"] == "non_visible":

            label_path.write_text(
                "",
                encoding="utf-8"
            )

            negatives += 1
            continue


        # ----------------------------------------
        # POSITIF
        # ----------------------------------------

        with Image.open(src) as im:

            width, height = im.size


        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])


        if not (
            0 <= x1 < x2 <= width
            and
            0 <= y1 < y2 <= height
        ):

            raise RuntimeError(
                f"Bbox invalide : "
                f"{image_name}"
            )


        xc = (
            (x1 + x2) / 2
        ) / width

        yc = (
            (y1 + y2) / 2
        ) / height

        bw = (
            x2 - x1
        ) / width

        bh = (
            y2 - y1
        ) / height


        label_path.write_text(
            (
                f"0 "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}\n"
            ),
            encoding="utf-8"
        )

        positives += 1


    return positives, negatives


train_pos, train_neg = export_split(
    train_usable,
    TRAIN_IMAGES,
    TRAIN_LABELS,
)

val_pos, val_neg = export_split(
    val_usable,
    VAL_IMAGES,
    VAL_LABELS,
)


# ============================================================
# YAML
# ============================================================

yaml_data = {
    "path": str(
        OUT.resolve()
    ),
    "train": "images/train",
    "val": "images/val",
    "names": {
        0: "support"
    },
}


YAML_PATH.write_text(
    yaml.safe_dump(
        yaml_data,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8"
)


# ============================================================
# CSV DE TRAÇABILITE
# ============================================================

train.to_csv(
    MERGED_TRAIN_CSV,
    index=False
)

val.to_csv(
    VAL_CSV,
    index=False
)


# ============================================================
# AUDIT FINAL
# ============================================================

print("\n====================================")
print("DATASET RENFORCE CREE")
print("====================================")

print("\nTRAIN")
print(
    "  positifs :",
    train_pos
)
print(
    "  négatifs :",
    train_neg
)
print(
    "  total    :",
    train_pos + train_neg
)

print("\nVAL")
print(
    "  positifs :",
    val_pos
)
print(
    "  négatifs :",
    val_neg
)
print(
    "  total    :",
    val_pos + val_neg
)

print(
    "\nTrain images :",
    len(
        list(
            TRAIN_IMAGES.glob("*.jpg")
        )
    )
)

print(
    "Train labels :",
    len(
        list(
            TRAIN_LABELS.glob("*.txt")
        )
    )
)

print(
    "Val images :",
    len(
        list(
            VAL_IMAGES.glob("*.jpg")
        )
    )
)

print(
    "Val labels :",
    len(
        list(
            VAL_LABELS.glob("*.txt")
        )
    )
)

print(
    "\nYAML :",
    YAML_PATH
)

print(
    "\n✅ Ambiguës exclues."
)

print(
    "✅ Validation humaine indépendante."
)

print(
    "✅ Test humain gelé totalement séparé."
)
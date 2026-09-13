from pathlib import Path
import shutil

import pandas as pd
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parent

ANNOTATIONS = (
    ROOT
    / "data"
    / "niveau2_manual_train"
    / "annotations.csv"
)

BASE_DATASET = (
    ROOT
    / "data"
    / "dataset_niveau2_humain"
)

OUT_DATASET = (
    ROOT
    / "data"
    / "dataset_niveau2_multiclasse"
)


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = {
    0: "batiment",
    1: "pylone_mat",
    2: "chateau_eau",
    3: "autre_support",
}

CLASS_MAPPING = {
    "batiment": 0,
    "pylone": 1,
    "mat": 1,
    "chateau_eau": 2,
    "autre_support": 3,
}


# ============================================================
# CHARGEMENT ANNOTATIONS
# ============================================================

df = pd.read_csv(ANNOTATIONS)

required = {
    "image",
    "status",
    "classe_niveau2",
    "x1",
    "y1",
    "x2",
    "y2",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Colonnes manquantes : {missing}"
    )


# Une seule ligne par image
if df["image"].duplicated().any():
    duplicates = df.loc[
        df["image"].duplicated(),
        "image"
    ].tolist()

    raise RuntimeError(
        f"Images dupliquées : {duplicates}"
    )


annotations = (
    df.set_index("image")
    .to_dict(orient="index")
)


# ============================================================
# CREATION DOSSIERS
# ============================================================

if OUT_DATASET.exists():
    shutil.rmtree(OUT_DATASET)


for split in ["train", "val"]:

    (
        OUT_DATASET
        / "images"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUT_DATASET
        / "labels"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CONVERSION
# ============================================================

stats = {
    "train": {
        "negative": 0,
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    },
    "val": {
        "negative": 0,
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    },
}


for split in ["train", "val"]:

    src_images = (
        BASE_DATASET
        / "images"
        / split
    )

    images = sorted(
        src_images.glob("*.jpg")
    )

    print(
        f"\n===== {split.upper()} ====="
    )

    for image_path in images:

        image_name = image_path.name

        if image_name not in annotations:
            raise RuntimeError(
                f"Annotation absente : {image_name}"
            )

        row = annotations[image_name]

        status = str(
            row["status"]
        ).strip()

        dst_image = (
            OUT_DATASET
            / "images"
            / split
            / image_name
        )

        dst_label = (
            OUT_DATASET
            / "labels"
            / split
            / f"{image_path.stem}.txt"
        )

        shutil.copy2(
            image_path,
            dst_image
        )


        # ====================================================
        # NEGATIF
        # ====================================================

        if status == "non_visible":

            dst_label.write_text(
                "",
                encoding="utf-8"
            )

            stats[split]["negative"] += 1

            continue


        # Le dataset de base ne doit jamais
        # contenir les "ambigu".
        if status != "visible":

            raise RuntimeError(
                f"Statut inattendu pour "
                f"{image_name} : {status}"
            )


        # ====================================================
        # CLASSE
        # ====================================================

        original_class = str(
            row["classe_niveau2"]
        ).strip()

        if original_class not in CLASS_MAPPING:

            raise RuntimeError(
                f"Classe inconnue : "
                f"{original_class}"
            )

        class_id = CLASS_MAPPING[
            original_class
        ]


        # ====================================================
        # BBOX
        # ====================================================

        coords = [
            row["x1"],
            row["y1"],
            row["x2"],
            row["y2"],
        ]

        if any(
            pd.isna(x)
            for x in coords
        ):
            raise RuntimeError(
                f"Bbox absente : {image_name}"
            )


        x1, y1, x2, y2 = map(
            float,
            coords
        )


        with Image.open(
            image_path
        ) as im:

            width, height = im.size


        if not (
            0 <= x1 < x2 <= width
            and
            0 <= y1 < y2 <= height
        ):

            raise RuntimeError(
                f"Bbox invalide "
                f"{image_name}: "
                f"{x1, y1, x2, y2}"
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


        dst_label.write_text(
            (
                f"{class_id} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}\n"
            ),
            encoding="utf-8"
        )

        stats[split][class_id] += 1


# ============================================================
# YAML
# ============================================================

yaml_data = {
    "path": str(
        OUT_DATASET.resolve()
    ),
    "train": "images/train",
    "val": "images/val",
    "names": CLASS_NAMES,
}

yaml_path = (
    OUT_DATASET
    / "dataset.yaml"
)

yaml_path.write_text(
    yaml.safe_dump(
        yaml_data,
        sort_keys=False,
        allow_unicode=True
    ),
    encoding="utf-8"
)


# ============================================================
# AUDIT FINAL
# ============================================================

print("\n====================================")
print("DATASET MULTI-CLASSE CREE")
print("====================================")

for split in ["train", "val"]:

    n_images = len(
        list(
            (
                OUT_DATASET
                / "images"
                / split
            ).glob("*.jpg")
        )
    )

    n_labels = len(
        list(
            (
                OUT_DATASET
                / "labels"
                / split
            ).glob("*.txt")
        )
    )

    print(
        f"\n{split.upper()} : "
        f"{n_images} images / "
        f"{n_labels} labels"
    )

    print(
        "  negatives :",
        stats[split]["negative"]
    )

    for class_id, name in CLASS_NAMES.items():

        print(
            f"  {name:<15} : "
            f"{stats[split][class_id]}"
        )


print(
    "\nYAML :",
    yaml_path
)

print(
    "\n✅ Même split train/val "
    "que le dataset mono-classe."
)
from pathlib import Path
import csv

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT
    / "data"
    / "dataset_niveau2_hardneg"
)

SOURCE_IMAGES = (
    ROOT
    / "data"
    / "dataset_niveau2_humain"
    / "images"
    / "train"
)

SELECTION_CSV = (
    DATASET
    / "hard_negatives_selection.csv"
)

TRAIN_IMAGES = (
    DATASET
    / "images"
    / "train"
)

TRAIN_LABELS = (
    DATASET
    / "labels"
    / "train"
)

YAML_PATH = (
    DATASET
    / "dataset.yaml"
)

CROP_SIZE = 320


def make_clean_crop(image, bbox, size=320):

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    half = size / 2

    left = int(round(cx - half))
    top = int(round(cy - half))
    right = int(round(cx + half))
    bottom = int(round(cy + half))

    if left < 0:
        right -= left
        left = 0

    if top < 0:
        bottom -= top
        top = 0

    if right > image.width:
        shift = right - image.width
        left -= shift
        right = image.width

    if bottom > image.height:
        shift = bottom - image.height
        top -= shift
        bottom = image.height

    left = max(0, left)
    top = max(0, top)

    return image.crop(
        (left, top, right, bottom)
    )


# ============================================================
# LECTURE SELECTION
# ============================================================

with SELECTION_CSV.open(
    encoding="utf-8"
) as f:

    rows = list(
        csv.DictReader(f)
    )


print("===== INJECTION HARD NEGATIVES =====")
print("Sélection :", len(rows))


created = []


for i, row in enumerate(
    rows,
    start=1
):

    source_name = row["image_source"]

    source_path = (
        SOURCE_IMAGES
        / source_name
    )

    if not source_path.exists():
        raise FileNotFoundError(
            source_path
        )


    bbox = [
        float(row["x1"]),
        float(row["y1"]),
        float(row["x2"]),
        float(row["y2"]),
    ]


    image = Image.open(
        source_path
    ).convert("RGB")


    crop = make_clean_crop(
        image,
        bbox,
        CROP_SIZE
    )


    stem = Path(
        source_name
    ).stem

    new_name = (
        f"hardneg_{i:02d}_{stem}.jpg"
    )

    image_out = (
        TRAIN_IMAGES
        / new_name
    )

    label_out = (
        TRAIN_LABELS
        / f"{Path(new_name).stem}.txt"
    )


    # Crop propre, sans rectangle rouge
    crop.save(
        image_out,
        quality=95
    )


    # Label vide = image négative YOLO
    label_out.write_text(
        "",
        encoding="utf-8"
    )


    created.append(
        new_name
    )

    print(
        f"{i:02d} -> {new_name}"
    )


# ============================================================
# YAML
# ============================================================

dataset_yaml = {
    "path": str(
        DATASET.resolve()
    ),
    "train": "images/train",
    "val": "images/val",
    "names": {
        0: "support"
    },
}


YAML_PATH.write_text(
    yaml.safe_dump(
        dataset_yaml,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


# ============================================================
# AUDIT
# ============================================================

train_images_count = len(
    list(
        TRAIN_IMAGES.glob("*.jpg")
    )
)

train_labels_count = len(
    list(
        TRAIN_LABELS.glob("*.txt")
    )
)

empty_labels = 0

for p in TRAIN_LABELS.glob(
    "*.txt"
):

    if not p.read_text(
        encoding="utf-8"
    ).strip():

        empty_labels += 1


print("\n====================================")
print("HARD NEGATIVES INJECTES")
print("====================================")

print(
    "Nouveaux hard negatives :",
    len(created)
)

print(
    "Images train total :",
    train_images_count
)

print(
    "Labels train total :",
    train_labels_count
)

print(
    "Labels vides train :",
    empty_labels
)

print(
    "YAML :",
    YAML_PATH
)

print(
    "\n✅ Validation inchangée."
)

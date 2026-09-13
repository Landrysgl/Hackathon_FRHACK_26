from pathlib import Path
import csv
import math

from PIL import Image, ImageDraw
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "runs"
    / "niveau2_finetune_baseline"
    / "weights"
    / "best.pt"
)

TRAIN_IMAGES = (
    ROOT
    / "data"
    / "dataset_niveau2_humain"
    / "images"
    / "train"
)

TRAIN_LABELS = (
    ROOT
    / "data"
    / "dataset_niveau2_humain"
    / "labels"
    / "train"
)

OUT_DIR = (
    ROOT
    / "data"
    / "niveau2_error_analysis"
    / "hard_negative_train_candidates"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = OUT_DIR / "candidats.csv"
MOSAIC_PATH = OUT_DIR / "mosaique_numerotee.jpg"
MAP_PATH = OUT_DIR / "correspondance_numeros.txt"

CONF = 0.03

# Seulement la fausse prédiction la plus confiante
# de chaque image -> maximum 49 candidats.
TOP_K_PER_IMAGE = 1

MAX_IOU_WITH_GT = 0.10

CROP_SIZE = 320

MOSAIC_COLS = 4
CELL_W = 360
CELL_H = 390


def iou_xyxy(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    inter = iw * ih

    area_a = (
        max(0.0, a[2] - a[0])
        * max(0.0, a[3] - a[1])
    )

    area_b = (
        max(0.0, b[2] - b[0])
        * max(0.0, b[3] - b[1])
    )

    union = area_a + area_b - inter

    return (
        inter / union
        if union > 0
        else 0.0
    )


def read_gt(label_path, image_path):

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return None

    parts = text.split()

    if len(parts) != 5:
        raise RuntimeError(
            f"Label invalide : {label_path}"
        )

    _, xc, yc, bw, bh = map(float, parts)

    with Image.open(image_path) as im:
        w, h = im.size

    xc *= w
    yc *= h
    bw *= w
    bh *= h

    return [
        xc - bw / 2,
        yc - bh / 2,
        xc + bw / 2,
        yc + bh / 2,
    ]


def make_crop(image, bbox, size):

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    half = size / 2

    left = int(cx - half)
    top = int(cy - half)
    right = int(cx + half)
    bottom = int(cy + half)

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

    crop = image.crop(
        (left, top, right, bottom)
    )

    return crop, left, top


if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)


model = YOLO(str(MODEL_PATH))


print("===== MINAGE HARD NEGATIVES TRAIN =====")
print("Modèle :", MODEL_PATH)
print("Images train :", TRAIN_IMAGES)
print("Seuil :", CONF)


results = model.predict(
    source=str(TRAIN_IMAGES),
    imgsz=1024,
    conf=CONF,
    save=False,
    verbose=False,
)


records = []
crop_paths = []


for result in results:

    image_path = Path(result.path)

    label_path = (
        TRAIN_LABELS
        / f"{image_path.stem}.txt"
    )

    gt = read_gt(
        label_path,
        image_path
    )

    predictions = []

    if result.boxes is not None:

        for box in result.boxes:

            confidence = float(
                box.conf.item()
            )

            bbox = (
                box.xyxy
                .squeeze()
                .tolist()
            )

            overlap = (
                iou_xyxy(bbox, gt)
                if gt is not None
                else 0.0
            )

            if overlap <= MAX_IOU_WITH_GT:

                predictions.append(
                    {
                        "confidence": confidence,
                        "bbox": bbox,
                        "iou_gt": overlap,
                    }
                )


    predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    predictions = predictions[
        :TOP_K_PER_IMAGE
    ]


    if not predictions:
        continue


    image = Image.open(
        image_path
    ).convert("RGB")


    for p in predictions:

        crop, ox, oy = make_crop(
            image,
            p["bbox"],
            CROP_SIZE
        )

        draw = ImageDraw.Draw(crop)

        x1, y1, x2, y2 = p["bbox"]

        local_box = [
            x1 - ox,
            y1 - oy,
            x2 - ox,
            y2 - oy,
        ]

        draw.rectangle(
            local_box,
            outline="red",
            width=4
        )

        draw.text(
            (5, 5),
            f"conf={p['confidence']:.3f}",
            fill="yellow"
        )

        crop_name = (
            f"{image_path.stem}"
            f"__conf{p['confidence']:.3f}.jpg"
        )

        crop_path = OUT_DIR / crop_name

        crop.save(
            crop_path,
            quality=95
        )

        crop_paths.append(crop_path)

        records.append(
            {
                "image_source": image_path.name,
                "crop": crop_name,
                "confidence": p["confidence"],
                "iou_gt": p["iou_gt"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )


# Numérotation stable
crop_paths = sorted(crop_paths)

mapping = []


rows_count = math.ceil(
    len(crop_paths) / MOSAIC_COLS
)

canvas = Image.new(
    "RGB",
    (
        MOSAIC_COLS * CELL_W,
        rows_count * CELL_H
    ),
    "white"
)


for number, path in enumerate(
    crop_paths,
    start=1
):

    img = Image.open(
        path
    ).convert("RGB")

    img.thumbnail(
        (340, 340)
    )

    cell = Image.new(
        "RGB",
        (CELL_W, CELL_H),
        "white"
    )

    x = (
        CELL_W - img.width
    ) // 2

    cell.paste(
        img,
        (x, 35)
    )

    draw = ImageDraw.Draw(cell)

    draw.text(
        (10, 5),
        f"CANDIDAT {number}",
        fill="black"
    )

    col = (
        number - 1
    ) % MOSAIC_COLS

    row = (
        number - 1
    ) // MOSAIC_COLS

    canvas.paste(
        cell,
        (
            col * CELL_W,
            row * CELL_H
        )
    )

    mapping.append(
        f"{number}\t{path.name}"
    )


canvas.save(
    MOSAIC_PATH,
    quality=95
)

MAP_PATH.write_text(
    "\n".join(mapping),
    encoding="utf-8"
)


if records:

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)


print("\n====================================")
print("CANDIDATS TRAIN PREPARES")
print("====================================")

print("Nombre :", len(crop_paths))
print("Mosaïque :", MOSAIC_PATH)
print("Correspondance :", MAP_PATH)
print("CSV :", CSV_PATH)

print(
    "\n✅ Aucun fichier de validation "
    "n'est utilisé pour l'entraînement."
)
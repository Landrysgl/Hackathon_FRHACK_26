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

VAL_IMAGES = (
    ROOT
    / "data"
    / "dataset_niveau2_humain"
    / "images"
    / "val"
)

VAL_LABELS = (
    ROOT
    / "data"
    / "dataset_niveau2_humain"
    / "labels"
    / "val"
)

OUT_DIR = (
    ROOT
    / "data"
    / "niveau2_error_analysis"
    / "hard_negative_candidates"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = OUT_DIR / "candidats_hard_negatives.csv"
MOSAIC_PATH = OUT_DIR / "mosaique_hard_negatives.jpg"

CONF = 0.03

# Maximum 5 candidats par image
TOP_K_PER_IMAGE = 5

# On évite les prédictions trop proches de la vraie bbox.
# Cela limite le risque de transformer un vrai support en négatif.
MAX_IOU_WITH_GT = 0.10

CROP_SIZE = 300
MOSAIC_COLS = 5


# ============================================================
# OUTILS
# ============================================================

def iou_xyxy(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    inter = iw * ih

    area_a = max(0.0, a[2] - a[0]) * max(
        0.0, a[3] - a[1]
    )

    area_b = max(0.0, b[2] - b[0]) * max(
        0.0, b[3] - b[1]
    )

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def read_gt(label_path):

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

    xc *= 1024
    yc *= 1024
    bw *= 1024
    bh *= 1024

    return [
        xc - bw / 2,
        yc - bh / 2,
        xc + bw / 2,
        yc + bh / 2,
    ]


def make_context_crop(image, bbox, size=300):

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    half = size / 2

    left = int(round(cx - half))
    top = int(round(cy - half))
    right = int(round(cx + half))
    bottom = int(round(cy + half))

    # Limites image
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


# ============================================================
# MODELE
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

model = YOLO(str(MODEL_PATH))


print("===== EXTRACTION HARD NEGATIVES =====")
print("Modèle :", MODEL_PATH)
print("Seuil :", CONF)
print("Maximum par image :", TOP_K_PER_IMAGE)


# ============================================================
# INFERENCE
# ============================================================

results = model.predict(
    source=str(VAL_IMAGES),
    imgsz=1024,
    conf=CONF,
    save=False,
    verbose=False,
)


rows = []
crop_files = []


for result in results:

    image_path = Path(result.path)

    label_path = (
        VAL_LABELS
        / f"{image_path.stem}.txt"
    )

    gt = read_gt(label_path)

    candidates = []

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

            if gt is None:
                overlap = 0.0
            else:
                overlap = iou_xyxy(
                    bbox,
                    gt
                )

            # Pour les hard negatives, on écarte
            # les prédictions proches du vrai objet.
            if overlap <= MAX_IOU_WITH_GT:

                candidates.append(
                    {
                        "confidence": confidence,
                        "bbox": bbox,
                        "iou_gt": overlap,
                    }
                )

    candidates.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    candidates = candidates[
        :TOP_K_PER_IMAGE
    ]

    image = Image.open(
        image_path
    ).convert("RGB")


    for rank, candidate in enumerate(
        candidates,
        start=1
    ):

        bbox = candidate["bbox"]

        crop, offset_x, offset_y = make_context_crop(
            image,
            bbox,
            CROP_SIZE
        )

        # Dessine la prédiction dans le crop
        draw = ImageDraw.Draw(crop)

        local_bbox = [
            bbox[0] - offset_x,
            bbox[1] - offset_y,
            bbox[2] - offset_x,
            bbox[3] - offset_y,
        ]

        draw.rectangle(
            local_bbox,
            outline="red",
            width=4
        )

        draw.text(
            (5, 5),
            f"conf={candidate['confidence']:.3f}",
            fill="yellow"
        )

        crop_name = (
            f"{image_path.stem}"
            f"__fp{rank:02d}"
            f"__conf{candidate['confidence']:.3f}.jpg"
        )

        crop_path = (
            OUT_DIR
            / crop_name
        )

        crop.save(
            crop_path,
            quality=92
        )

        crop_files.append(
            crop_path
        )

        rows.append(
            {
                "image_source": image_path.name,
                "rang": rank,
                "confidence": candidate["confidence"],
                "iou_avec_gt": candidate["iou_gt"],
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "crop": crop_name,

                # Colonnes à remplir ensuite manuellement
                "type_erreur": "",
                "garder_comme_hard_negative": "",
                "commentaire": "",
            }
        )


# ============================================================
# CSV
# ============================================================

if rows:

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MOSAIQUE
# ============================================================

if crop_files:

    thumb_w = 300
    thumb_h = 330

    cols = MOSAIC_COLS
    rows_count = math.ceil(
        len(crop_files) / cols
    )

    canvas = Image.new(
        "RGB",
        (
            cols * thumb_w,
            rows_count * thumb_h
        ),
        "white"
    )

    for i, path in enumerate(crop_files):

        img = Image.open(
            path
        ).convert("RGB")

        img.thumbnail(
            (thumb_w, 300)
        )

        cell = Image.new(
            "RGB",
            (thumb_w, thumb_h),
            "white"
        )

        cell.paste(
            img,
            (
                (thumb_w - img.width) // 2,
                0
            )
        )

        draw = ImageDraw.Draw(cell)

        draw.text(
            (5, 305),
            path.stem[:45],
            fill="black"
        )

        col = i % cols
        row = i // cols

        canvas.paste(
            cell,
            (
                col * thumb_w,
                row * thumb_h
            )
        )

    canvas.save(
        MOSAIC_PATH,
        quality=92
    )


# ============================================================
# RESUME
# ============================================================

print("\n===================================")
print("CANDIDATS HARD NEGATIVES PREPARES")
print("===================================")

print("Nombre de crops :", len(crop_files))
print("Dossier :", OUT_DIR)
print("CSV :", CSV_PATH)
print("Mosaïque :", MOSAIC_PATH)

print(
    "\n⚠️ Rien n'est encore ajouté à l'entraînement."
)

print(
    "➡️ Il faut d'abord vérifier manuellement "
    "que ces crops ne contiennent pas le vrai support."
)
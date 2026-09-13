from pathlib import Path
import csv

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
    / "val_conf003"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = OUT_DIR / "resume_faux_positifs.csv"

CONF = 0.03
IMGSZ = 1024

# Pour ne pas rendre les images totalement illisibles.
MAX_BOXES_DRAWN = 20


def iou(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)

    inter = iw * ih

    area_a = (
        max(0, a[2] - a[0])
        * max(0, a[3] - a[1])
    )

    area_b = (
        max(0, b[2] - b[0])
        * max(0, b[3] - b[1])
    )

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


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


if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)


model = YOLO(str(MODEL_PATH))


print("===== ANALYSE FAUX POSITIFS =====")
print("Modèle :", MODEL_PATH)
print("Seuil :", CONF)


results = model.predict(
    source=str(VAL_IMAGES),
    imgsz=IMGSZ,
    conf=CONF,
    save=False,
    verbose=False,
)


rows = []


for result in results:

    image_path = Path(result.path)

    label_path = (
        VAL_LABELS
        / f"{image_path.stem}.txt"
    )

    gt = read_gt(label_path)

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

            predictions.append(
                {
                    "conf": confidence,
                    "bbox": bbox,
                }
            )


    # Les plus fortes confiances en premier
    predictions.sort(
        key=lambda x: x["conf"],
        reverse=True
    )


    best_iou = 0.0
    best_iou_conf = None

    if gt is not None:

        for p in predictions:

            score = iou(
                p["bbox"],
                gt
            )

            if score > best_iou:

                best_iou = score
                best_iou_conf = p["conf"]


    # --------------------------------------------------------
    # VISUALISATION
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)


    # Vérité terrain humaine = vert
    if gt is not None:

        draw.rectangle(
            gt,
            outline="lime",
            width=5
        )

        draw.text(
            (gt[0], max(0, gt[1] - 18)),
            "GT humain",
            fill="lime"
        )


    # Prédictions = rouge
    for p in predictions[:MAX_BOXES_DRAWN]:

        x1, y1, x2, y2 = p["bbox"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 15)),
            f"{p['conf']:.3f}",
            fill="red"
        )


    # Infos générales
    title = (
        f"pred={len(predictions)}"
        f" | GT={'oui' if gt else 'non'}"
        f" | bestIoU={best_iou:.3f}"
    )

    draw.text(
        (10, 10),
        title,
        fill="yellow"
    )


    out_path = (
        OUT_DIR
        / f"analyse_{image_path.name}"
    )

    image.save(
        out_path,
        quality=92
    )


    rows.append(
        {
            "image": image_path.name,
            "gt_present": gt is not None,
            "nombre_predictions": len(predictions),
            "confidence_max": (
                predictions[0]["conf"]
                if predictions
                else 0.0
            ),
            "best_iou_avec_gt": best_iou,
            "confidence_best_iou": best_iou_conf,
        }
    )


# ============================================================
# CSV
# ============================================================

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


print("\n===================================")
print("ANALYSE PREPAREE")
print("===================================")

print("Images analysées :", len(rows))
print("Dossier :", OUT_DIR)
print("Résumé :", CSV_PATH)

print(
    "\nVert = vérité terrain humaine"
)

print(
    "Rouge = prédictions du modèle"
)
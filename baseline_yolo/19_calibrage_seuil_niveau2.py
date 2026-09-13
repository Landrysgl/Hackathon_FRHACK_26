from pathlib import Path
import json

from PIL import Image
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

OUT_PATH = (
    ROOT
    / "runs"
    / "niveau2_finetune_baseline"
    / "threshold_calibration.json"
)

IMGSZ = 1024
IOU_MATCH = 0.50

THRESHOLDS = [
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.030,
    0.050,
    0.075,
    0.100,
    0.150,
    0.200,
    0.300,
]


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


def read_gt(label_path, image_path):

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    # Image négative
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


if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

model = YOLO(str(MODEL_PATH))


# ============================================================
# INFERENCE UNE SEULE FOIS
# ============================================================

print("===== INFERENCE VALIDATION NIVEAU 2 =====")

results = model.predict(
    source=str(VAL_IMAGES),
    imgsz=IMGSZ,
    conf=0.001,
    save=False,
    verbose=False,
)


images_data = []

for result in results:

    image_path = Path(result.path)

    label_path = (
        VAL_LABELS
        / f"{image_path.stem}.txt"
    )

    gt = read_gt(
        label_path,
        image_path
    )

    predictions = []

    if result.boxes is not None:

        for box in result.boxes:

            conf = float(box.conf.item())

            xyxy = (
                box.xyxy
                .squeeze()
                .tolist()
            )

            predictions.append(
                {
                    "confidence": conf,
                    "bbox": xyxy,
                }
            )

    images_data.append(
        {
            "image": image_path.name,
            "gt": gt,
            "predictions": predictions,
        }
    )


# ============================================================
# TEST DES SEUILS
# ============================================================

all_metrics = []

print("\n===== CALIBRAGE =====")

for threshold in THRESHOLDS:

    TP = 0
    FP = 0
    FN = 0

    for item in images_data:

        gt = item["gt"]

        preds = [
            p
            for p in item["predictions"]
            if p["confidence"] >= threshold
        ]

        # Image négative
        if gt is None:

            FP += len(preds)
            continue

        # Image positive sans prédiction
        if len(preds) == 0:

            FN += 1
            continue

        ious = [
            iou_xyxy(
                p["bbox"],
                gt
            )
            for p in preds
        ]

        best_iou = max(ious)

        if best_iou >= IOU_MATCH:

            TP += 1

            # Une prédiction appariée,
            # les autres sont des faux positifs.
            FP += len(preds) - 1

        else:

            FN += 1
            FP += len(preds)

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0.0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    row = {
        "threshold": threshold,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    all_metrics.append(row)

    print(
        f"conf={threshold:>5.3f} | "
        f"TP={TP:2d} "
        f"FP={FP:3d} "
        f"FN={FN:2d} | "
        f"P={precision:.4f} "
        f"R={recall:.4f} "
        f"F1={f1:.4f}"
    )


# ============================================================
# MEILLEUR SEUIL
# ============================================================

best = max(
    all_metrics,
    key=lambda x: (
        x["f1"],
        x["precision"],
        x["recall"],
    )
)

print("\n===================================")
print("MEILLEUR SEUIL SUR VALIDATION")
print("===================================")

print(
    f"Seuil      : {best['threshold']}"
)

print(
    f"TP / FP / FN : "
    f"{best['TP']} / "
    f"{best['FP']} / "
    f"{best['FN']}"
)

print(
    f"Précision  : {best['precision']:.4f}"
)

print(
    f"Rappel     : {best['recall']:.4f}"
)

print(
    f"F1         : {best['f1']:.4f}"
)


payload = {
    "model": str(MODEL_PATH),
    "validation_images": len(images_data),
    "iou_match": IOU_MATCH,
    "best_threshold": best["threshold"],
    "best_metrics": best,
    "all_thresholds": all_metrics,
}

OUT_PATH.write_text(
    json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("\nRésultats :", OUT_PATH)
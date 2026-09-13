from pathlib import Path
import json
import math
import shutil

import pandas as pd
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MANUAL_DIR = DATA_DIR / "manual_validation"
ANNOTATIONS_PATH = MANUAL_DIR / "annotations.csv"
MANUAL_IMAGES_DIR = MANUAL_DIR / "images"
EVAL_DIR = MANUAL_DIR / "evaluation"

YOLO_EVAL_DIR = EVAL_DIR / "yolo_dataset"
YOLO_IMAGES_DIR = YOLO_EVAL_DIR / "images" / "val"
YOLO_LABELS_DIR = YOLO_EVAL_DIR / "labels" / "val"
YOLO_YAML = YOLO_EVAL_DIR / "dataset.yaml"

WEIGHTS = ROOT / "runs" / "baseline_weak" / "weights" / "best.pt"

IMAGE_SIZE = 1024
CONF_THRESHOLD = 0.05
MATCH_IOU_THRESHOLD = 0.50
IMGSZ = 1024

VALID_STATUSES = {"visible", "non_visible", "ambigu"}

if not ANNOTATIONS_PATH.exists():
    raise FileNotFoundError(
        f"{ANNOTATIONS_PATH} introuvable. "
        "Prépare et annote d'abord la validation manuelle."
    )

if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"{WEIGHTS} introuvable. Lance d'abord l'entraînement."
    )

annotations = pd.read_csv(
    ANNOTATIONS_PATH,
    dtype={"SUP_ID": "string"},
)

annotations["status"] = (
    annotations["status"]
    .fillna("")
    .astype(str)
    .str.strip()
)

non_annotes = annotations[annotations["status"] == ""]
if len(non_annotes) > 0:
    raise ValueError(
        f"{len(non_annotes)} images ne sont pas encore annotées."
    )

invalides = annotations[
    ~annotations["status"].isin(VALID_STATUSES)
]
if len(invalides) > 0:
    raise ValueError(
        "Statuts invalides : "
        f"{invalides[['SUP_ID', 'status']].to_dict('records')}"
    )

utilisables = annotations[
    annotations["status"] != "ambigu"
].copy()

if len(utilisables) == 0:
    raise ValueError("Aucune annotation utilisable.")

visibles = utilisables[
    utilisables["status"] == "visible"
].copy()

for col in ["x1", "y1", "x2", "y2"]:
    visibles[col] = pd.to_numeric(
        visibles[col],
        errors="coerce",
    )

if visibles[["x1", "y1", "x2", "y2"]].isna().any().any():
    raise ValueError(
        "Une image 'visible' ne possède pas une bbox complète."
    )

for row in visibles.itertuples(index=False):
    if not (
        0 <= row.x1 < row.x2 <= IMAGE_SIZE
        and 0 <= row.y1 < row.y2 <= IMAGE_SIZE
    ):
        raise ValueError(
            f"BBox invalide pour SUP_ID {row.SUP_ID}: "
            f"{row.x1, row.y1, row.x2, row.y2}"
        )

print("===== VALIDATION MANUELLE =====")
print("Total annoté :", len(annotations))
print("Utilisables :", len(utilisables))
print("Visible :", int((utilisables["status"] == "visible").sum()))
print("Non visible :", int((utilisables["status"] == "non_visible").sum()))
print("Ambigu exclus :", int((annotations["status"] == "ambigu").sum()))

if YOLO_EVAL_DIR.exists():
    shutil.rmtree(YOLO_EVAL_DIR)

YOLO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
YOLO_LABELS_DIR.mkdir(parents=True, exist_ok=True)

for row in utilisables.itertuples(index=False):
    src = MANUAL_IMAGES_DIR / row.image
    dst = YOLO_IMAGES_DIR / row.image
    shutil.copy2(src, dst)

    label_path = YOLO_LABELS_DIR / f"{Path(row.image).stem}.txt"

    if row.status == "non_visible":
        label_path.write_text("", encoding="utf-8")
        continue

    x1 = float(row.x1)
    y1 = float(row.y1)
    x2 = float(row.x2)
    y2 = float(row.y2)

    xc = ((x1 + x2) / 2.0) / IMAGE_SIZE
    yc = ((y1 + y2) / 2.0) / IMAGE_SIZE
    w = (x2 - x1) / IMAGE_SIZE
    h = (y2 - y1) / IMAGE_SIZE

    label_path.write_text(
        f"0 {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}\n",
        encoding="utf-8",
    )

YOLO_YAML.write_text(
    "\n".join([
        f"path: {YOLO_EVAL_DIR.resolve()}",
        "train: images/val",
        "val: images/val",
        "nc: 1",
        "names:",
        "  0: antenne",
        "",
    ]),
    encoding="utf-8",
)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(str(WEIGHTS))

TP = 0
FP = 0
FN = 0
details = []

for row in utilisables.itertuples(index=False):
    image_path = MANUAL_IMAGES_DIR / row.image

    predictions = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        imgsz=IMGSZ,
        device=device,
        verbose=False,
    )[0]

    pred_boxes = []

    if predictions.boxes is not None:
        for box in predictions.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf.item())
            pred_boxes.append({
                "bbox": [float(v) for v in xyxy],
                "conf": conf,
            })

    if row.status == "non_visible":
        tp_i = 0
        fp_i = len(pred_boxes)
        fn_i = 0
        best_iou = None

    else:
        gt = [
            float(row.x1),
            float(row.y1),
            float(row.x2),
            float(row.y2),
        ]

        ious = [
            iou_xyxy(p["bbox"], gt)
            for p in pred_boxes
        ]

        best_iou = max(ious) if ious else 0.0

        if best_iou >= MATCH_IOU_THRESHOLD:
            tp_i = 1
            fn_i = 0
            fp_i = max(0, len(pred_boxes) - 1)
        else:
            tp_i = 0
            fn_i = 1
            fp_i = len(pred_boxes)

    TP += tp_i
    FP += fp_i
    FN += fn_i

    details.append({
        "SUP_ID": row.SUP_ID,
        "image": row.image,
        "status": row.status,
        "nb_predictions": len(pred_boxes),
        "best_iou": best_iou,
        "TP": tp_i,
        "FP": fp_i,
        "FN": fn_i,
    })

precision = TP / (TP + FP) if (TP + FP) else 0.0
recall = TP / (TP + FN) if (TP + FN) else 0.0

print("\n===== TP / FP / FN A CONF =", CONF_THRESHOLD, "=====")
print("TP :", TP)
print("FP :", FP)
print("FN :", FN)
print(f"Précision : {precision:.4f}")
print(f"Rappel    : {recall:.4f}")

print("\n===== mAP SUR VERITE TERRAIN HUMAINE =====")

metrics = model.val(
    data=str(YOLO_YAML),
    split="val",
    imgsz=IMGSZ,
    device=device,
    plots=True,
    verbose=True,
)

map50 = float(metrics.box.map50)
map50_95 = float(metrics.box.map)
map_precision = float(metrics.box.mp)
map_recall = float(metrics.box.mr)

print(f"mAP50      : {map50:.4f}")
print(f"mAP50-95   : {map50_95:.4f}")
print(f"Precision* : {map_precision:.4f}")
print(f"Recall*    : {map_recall:.4f}")
print("* métriques Ultralytics intégrées sur le jeu manuel")

EVAL_DIR.mkdir(parents=True, exist_ok=True)

details_path = EVAL_DIR / "details.csv"
pd.DataFrame(details).to_csv(details_path, index=False)

summary = {
    "weights": str(WEIGHTS.resolve()),
    "manual_images_total": int(len(annotations)),
    "manual_images_used": int(len(utilisables)),
    "visible": int((utilisables["status"] == "visible").sum()),
    "non_visible": int((utilisables["status"] == "non_visible").sum()),
    "ambigu_excluded": int((annotations["status"] == "ambigu").sum()),
    "conf_threshold_for_tp_fp_fn": CONF_THRESHOLD,
    "match_iou_threshold": MATCH_IOU_THRESHOLD,
    "TP": TP,
    "FP": FP,
    "FN": FN,
    "precision": precision,
    "recall": recall,
    "map50_manual": map50,
    "map50_95_manual": map50_95,
    "ultralytics_precision_manual": map_precision,
    "ultralytics_recall_manual": map_recall,
}

summary_path = EVAL_DIR / "metrics_manual.json"
summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n✅ Evaluation humaine terminée.")
print("Détails :", details_path)
print("Résumé :", summary_path)

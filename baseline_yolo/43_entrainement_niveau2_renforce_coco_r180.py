from pathlib import Path
import json

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

DATASET_DIR = (
    ROOT
    / "data"
    / "dataset_niveau2_renforce"
)

YAML_PATH = (
    DATASET_DIR
    / "dataset.yaml"
)

BASELINE_WEIGHTS = ROOT / "yolov8n.pt"

RUNS_DIR = ROOT / "runs"

RUN_NAME = "niveau2_renforce_coco_rotation180"

EPOCHS = 50
IMGSZ = 1024
BATCH = 8
PATIENCE = 15
SEED = 42

# Seule modification expérimentale
DEGREES = 180.0


# ============================================================
# VERIFICATIONS
# ============================================================

if not YAML_PATH.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {YAML_PATH}"
    )

if not BASELINE_WEIGHTS.exists():
    raise FileNotFoundError(
        f"Poids Niveau 1 introuvables : {BASELINE_WEIGHTS}"
    )


device = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


print("====================================")
print("NIVEAU 2 - ROTATION AERIENNE")
print("====================================")

print("Dataset :", YAML_PATH)
print("Poids initiaux :", BASELINE_WEIGHTS)
print("Epochs max :", EPOCHS)
print("Image size :", IMGSZ)
print("Batch :", BATCH)
print("Patience :", PATIENCE)
print("Seed :", SEED)
print("Rotation :", f"±{DEGREES}°")
print("Device :", device)

if torch.cuda.is_available():
    print(
        "GPU :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# MODELE
# ============================================================

model = YOLO(
    str(BASELINE_WEIGHTS)
)


# ============================================================
# ENTRAINEMENT
# ============================================================

model.train(
    data=str(YAML_PATH),

    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,

    seed=SEED,
    deterministic=True,

    device=device,

    project=str(RUNS_DIR),
    name=RUN_NAME,
    exist_ok=True,

    plots=True,

    # Expérience contrôlée :
    # seule cette augmentation est modifiée.
    degrees=DEGREES,
)


# ============================================================
# RESULTATS
# ============================================================

BEST = (
    RUNS_DIR
    / RUN_NAME
    / "weights"
    / "best.pt"
)

LAST = (
    RUNS_DIR
    / RUN_NAME
    / "weights"
    / "last.pt"
)


print("\n====================================")
print("ENTRAINEMENT TERMINE")
print("====================================")

print("best.pt :", BEST)
print("last.pt :", LAST)

if BEST.exists():

    print(
        "Taille best.pt : "
        f"{BEST.stat().st_size / 1024**2:.1f} Mo"
    )

    print("✅ Meilleur modèle sauvegardé.")

else:

    print("❌ best.pt introuvable.")


summary = {
    "experiment": RUN_NAME,
    "initial_weights": str(BASELINE_WEIGHTS),
    "dataset": str(YAML_PATH),
    "epochs_max": EPOCHS,
    "imgsz": IMGSZ,
    "batch": BATCH,
    "patience": PATIENCE,
    "seed": SEED,
    "degrees": DEGREES,
    "best_weights": str(BEST),
}

summary_path = (
    RUNS_DIR
    / RUN_NAME
    / "training_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print(
    "Résumé :",
    summary_path
)
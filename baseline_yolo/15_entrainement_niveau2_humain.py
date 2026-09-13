from pathlib import Path
import json

import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "data" / "dataset_niveau2_humain"
YAML_PATH = DATASET_DIR / "dataset.yaml"

RUNS_DIR = ROOT / "runs"
RUN_NAME = "niveau2_humain"

LOCAL_MODEL = ROOT / "yolov8n.pt"

EPOCHS = 50
IMGSZ = 1024
BATCH = 8
PATIENCE = 15
SEED = 42


# ============================================================
# VERIFICATIONS
# ============================================================

if not YAML_PATH.exists():
    raise FileNotFoundError(
        f"dataset.yaml introuvable : {YAML_PATH}"
    )

device = 0 if torch.cuda.is_available() else "cpu"

print("===== EXPERIENCE NIVEAU 2 - HUMAIN UNIQUEMENT =====")
print("Dataset :", YAML_PATH)
print("Epochs :", EPOCHS)
print("Image size :", IMGSZ)
print("Batch :", BATCH)
print("Device :", device)

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA indisponible.")


# ============================================================
# MODELE INITIAL
# ============================================================

if LOCAL_MODEL.exists():
    model_source = str(LOCAL_MODEL)
else:
    # Ultralytics le téléchargera si nécessaire.
    model_source = "yolov8n.pt"

print("Poids initiaux :", model_source)

model = YOLO(model_source)


# ============================================================
# ENTRAINEMENT
# ============================================================

results = model.train(
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
)


# ============================================================
# RESULTATS
# ============================================================

BEST = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
LAST = RUNS_DIR / RUN_NAME / "weights" / "last.pt"
RESULTS_CSV = RUNS_DIR / RUN_NAME / "results.csv"

print("\n====================================")
print("ENTRAINEMENT NIVEAU 2 TERMINE")
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


# ============================================================
# PETIT RESUME JSON
# ============================================================

summary = {
    "experiment": "niveau2_humain",
    "initial_weights": model_source,
    "dataset": str(YAML_PATH),
    "epochs_max": EPOCHS,
    "imgsz": IMGSZ,
    "batch": BATCH,
    "patience": PATIENCE,
    "seed": SEED,
    "device": str(device),
    "best_weights": str(BEST),
}

summary_path = RUNS_DIR / RUN_NAME / "training_summary.json"

summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("Résumé :", summary_path)
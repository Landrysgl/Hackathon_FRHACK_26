from pathlib import Path
import json

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "dataset_yolo"
YAML_PATH = DATASET_DIR / "dataset.yaml"

RUNS_DIR = ROOT / "runs"
RUN_NAME = "baseline_weak"

MODEL = "yolov8n.pt"
EPOCHS = 50
IMGSZ = 1024
BATCH = 8
PATIENCE = 15
SEED = 42

if not YAML_PATH.exists():
    raise FileNotFoundError(
        f"{YAML_PATH} introuvable. Lance d'abord 06_creation_dataset_yolo.py."
    )

device = 0 if torch.cuda.is_available() else "cpu"

print("===== ENTRAINEMENT BASELINE =====")
print("Dataset :", YAML_PATH)
print("Modèle :", MODEL)
print("Device :", device)
print("Image size :", IMGSZ)
print("Batch :", BATCH)
print("Epochs :", EPOCHS)
print(
    "⚠️ Entraînement sur annotations faibles centrales, "
    "pas sur vérité terrain humaine."
)

model = YOLO(MODEL)

result = model.train(
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
    workers=4,
    cache=False,
    verbose=True,
)

best_path = RUNS_DIR / RUN_NAME / "weights" / "best.pt"

if not best_path.exists():
    raise FileNotFoundError(
        f"L'entraînement est terminé mais best.pt est introuvable : {best_path}"
    )

summary = {
    "model_initial": MODEL,
    "epochs_max": EPOCHS,
    "imgsz": IMGSZ,
    "batch": BATCH,
    "patience": PATIENCE,
    "seed": SEED,
    "device": str(device),
    "best_weights": str(best_path.resolve()),
    "training_labels": "weak",
}

summary_path = RUNS_DIR / RUN_NAME / "training_summary.json"
summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n✅ Entraînement terminé.")
print("Poids :", best_path)
print("Résumé :", summary_path)

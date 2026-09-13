from pathlib import Path
import json

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "dataset_yolo"
YAML_PATH = DATASET_DIR / "dataset.yaml"
WEIGHTS = ROOT / "runs" / "baseline_weak" / "weights" / "best.pt"
OUTPUT_PATH = ROOT / "runs" / "baseline_weak" / "weak_metrics.json"

IMGSZ = 1024

if not YAML_PATH.exists():
    raise FileNotFoundError(YAML_PATH)
if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"{WEIGHTS} introuvable. Lance d'abord 08_entrainement_baseline.py."
    )

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(str(WEIGHTS))

resultats = {}

for split in ("val", "test"):
    print(f"\n===== EVALUATION {split.upper()} - LABELS FAIBLES =====")

    metrics = model.val(
        data=str(YAML_PATH),
        split=split,
        imgsz=IMGSZ,
        device=device,
        plots=True,
        verbose=True,
    )

    split_metrics = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision_moyenne": float(metrics.box.mp),
        "rappel_moyen": float(metrics.box.mr),
    }

    resultats[split] = split_metrics

    for cle, valeur in split_metrics.items():
        print(f"{cle}: {valeur:.4f}")

OUTPUT_PATH.write_text(
    json.dumps(
        {
            "warning": (
                "Ces métriques comparent le modèle aux annotations faibles "
                "centrales. Elles ne constituent pas une validation humaine."
            ),
            "splits": resultats,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("\n⚠️ Ne présente pas ces mAP comme une vérité terrain réelle.")
print("Métriques enregistrées :", OUTPUT_PATH)

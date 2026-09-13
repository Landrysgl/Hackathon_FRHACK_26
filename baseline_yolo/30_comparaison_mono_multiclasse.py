from pathlib import Path
import json

import pandas as pd
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

RUNS_DIR = ROOT / "runs"

MONO_YAML = (
    ROOT
    / "data"
    / "dataset_niveau2_compare_mono"
    / "dataset.yaml"
)

MULTI_YAML = (
    ROOT
    / "data"
    / "dataset_niveau2_compare_multi"
    / "dataset.yaml"
)


# ============================================================
# PARAMETRES COMMUNS
# ============================================================

INITIAL_WEIGHTS = "yolov8n.pt"

EPOCHS = 50
IMGSZ = 1024
BATCH = 8
PATIENCE = 15
SEED = 42
DEGREES = 180.0


EXPERIMENTS = [
    {
        "name": "niveau2_compare_mono_r180",
        "data": MONO_YAML,
        "type": "mono",
    },
    {
        "name": "niveau2_compare_multi_r180",
        "data": MULTI_YAML,
        "type": "multi",
    },
]


# ============================================================
# VERIFICATIONS
# ============================================================

for exp in EXPERIMENTS:

    if not exp["data"].exists():
        raise FileNotFoundError(
            exp["data"]
        )


device = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


print("==========================================")
print("COMPARAISON MONO-CLASSE / MULTI-CLASSE")
print("==========================================")

print("Initialisation :", INITIAL_WEIGHTS)
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


summaries = []


# ============================================================
# ENTRAINEMENTS
# ============================================================

for exp in EXPERIMENTS:

    print("\n\n")
    print("=" * 60)
    print(
        f"ENTRAINEMENT : "
        f"{exp['type'].upper()}"
    )
    print("=" * 60)

    print(
        "Dataset :",
        exp["data"]
    )

    print(
        "Run :",
        exp["name"]
    )


    run_dir = (
        RUNS_DIR
        / exp["name"]
    )

    if run_dir.exists():

        raise RuntimeError(
            f"\nLe dossier existe déjà : {run_dir}\n"
            "Je préfère arrêter plutôt que mélanger "
            "deux expériences."
        )


    # Modèle neuf pour chaque expérience :
    # même initialisation.
    model = YOLO(
        INITIAL_WEIGHTS
    )


    model.train(
        data=str(exp["data"]),

        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,

        seed=SEED,
        deterministic=True,

        device=device,

        project=str(RUNS_DIR),
        name=exp["name"],
        exist_ok=False,

        plots=True,

        degrees=DEGREES,
    )


    # ========================================================
    # ANALYSE RESULTS.CSV
    # ========================================================

    results_csv = (
        run_dir
        / "results.csv"
    )

    if not results_csv.exists():

        raise FileNotFoundError(
            results_csv
        )


    df = pd.read_csv(
        results_csv
    )


    best_index = (
        df["metrics/mAP50-95(B)"]
        .idxmax()
    )

    best = df.loc[
        best_index
    ]


    best_weights = (
        run_dir
        / "weights"
        / "best.pt"
    )


    print("\n===== RESULTAT =====")

    print(
        "Époques exécutées :",
        len(df)
    )

    print(
        "Meilleure époque :",
        int(best["epoch"])
    )

    print(
        "Precision :",
        float(
            best["metrics/precision(B)"]
        )
    )

    print(
        "Recall :",
        float(
            best["metrics/recall(B)"]
        )
    )

    print(
        "mAP50 :",
        float(
            best["metrics/mAP50(B)"]
        )
    )

    print(
        "mAP50-95 :",
        float(
            best["metrics/mAP50-95(B)"]
        )
    )

    print(
        "best.pt :",
        best_weights
    )


    summaries.append(
        {
            "type": exp["type"],
            "run": exp["name"],
            "epochs_executed": len(df),
            "best_epoch": int(
                best["epoch"]
            ),
            "precision": float(
                best[
                    "metrics/precision(B)"
                ]
            ),
            "recall": float(
                best[
                    "metrics/recall(B)"
                ]
            ),
            "map50": float(
                best[
                    "metrics/mAP50(B)"
                ]
            ),
            "map50_95": float(
                best[
                    "metrics/mAP50-95(B)"
                ]
            ),
            "weights": str(
                best_weights
            ),
        }
    )


# ============================================================
# COMPARAISON
# ============================================================

print("\n\n")
print("==========================================")
print("COMPARAISON FINALE")
print("==========================================")

for s in summaries:

    print(
        f"\n{s['type'].upper()}"
    )

    print(
        "  best epoch :",
        s["best_epoch"]
    )

    print(
        "  precision  :",
        f"{s['precision']:.6f}"
    )

    print(
        "  recall     :",
        f"{s['recall']:.6f}"
    )

    print(
        "  mAP50      :",
        f"{s['map50']:.6f}"
    )

    print(
        "  mAP50-95   :",
        f"{s['map50_95']:.6f}"
    )


summary_path = (
    RUNS_DIR
    / "comparaison_mono_multi_r180.json"
)

summary_path.write_text(
    json.dumps(
        {
            "initial_weights": INITIAL_WEIGHTS,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "patience": PATIENCE,
            "seed": SEED,
            "degrees": DEGREES,
            "experiments": summaries,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print(
    "\nRésumé :",
    summary_path
)
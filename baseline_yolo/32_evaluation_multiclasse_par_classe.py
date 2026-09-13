from pathlib import Path
import json

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

WEIGHTS = (
    ROOT
    / "runs"
    / "niveau2_compare_multi_r180"
    / "weights"
    / "best.pt"
)

DATA = (
    ROOT
    / "data"
    / "dataset_niveau2_compare_multi"
    / "dataset.yaml"
)

OUT_DIR = (
    ROOT
    / "runs"
    / "niveau2_compare_multi_r180"
    / "evaluation_par_classe"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


if not WEIGHTS.exists():
    raise FileNotFoundError(WEIGHTS)

if not DATA.exists():
    raise FileNotFoundError(DATA)


model = YOLO(str(WEIGHTS))


print("====================================")
print("EVALUATION MULTI-CLASSE")
print("====================================")

metrics = model.val(
    data=str(DATA),
    imgsz=1024,
    split="val",
    plots=True,
    project=str(OUT_DIR.parent),
    name=OUT_DIR.name,
    exist_ok=True,
    verbose=False,
)


names = model.names

print("\n===== RESULTATS GLOBAUX =====")

print(
    "Precision :",
    float(metrics.box.mp)
)

print(
    "Recall :",
    float(metrics.box.mr)
)

print(
    "mAP50 :",
    float(metrics.box.map50)
)

print(
    "mAP50-95 :",
    float(metrics.box.map)
)


print("\n===== RESULTATS PAR CLASSE =====")

rows = []

maps = metrics.box.maps

for class_id, class_name in names.items():

    class_id = int(class_id)

    map50_95 = (
        float(maps[class_id])
        if class_id < len(maps)
        else None
    )

    # Ultralytics expose également
    # les résultats détaillés via class_result().
    try:

        p, r, map50, map5095 = (
            metrics.box.class_result(
                class_id
            )
        )

        p = float(p)
        r = float(r)
        map50 = float(map50)
        map5095 = float(map5095)

    except Exception:

        p = None
        r = None
        map50 = None
        map5095 = map50_95


    print(
        f"\nClasse {class_id} - "
        f"{class_name}"
    )

    print(
        "  Precision :",
        p
    )

    print(
        "  Recall :",
        r
    )

    print(
        "  mAP50 :",
        map50
    )

    print(
        "  mAP50-95 :",
        map5095
    )


    rows.append(
        {
            "class_id": class_id,
            "class_name": class_name,
            "precision": p,
            "recall": r,
            "map50": map50,
            "map50_95": map5095,
        }
    )


summary = {
    "weights": str(
        WEIGHTS.resolve()
    ),
    "data": str(
        DATA.resolve()
    ),
    "global": {
        "precision": float(
            metrics.box.mp
        ),
        "recall": float(
            metrics.box.mr
        ),
        "map50": float(
            metrics.box.map50
        ),
        "map50_95": float(
            metrics.box.map
        ),
    },
    "classes": rows,
}


out_json = (
    OUT_DIR
    / "metrics_par_classe.json"
)

out_json.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print(
    "\nRésultats :",
    out_json
)

print(
    "Matrices/figures :",
    OUT_DIR
)
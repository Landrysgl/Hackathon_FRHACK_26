from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent

OUT_JSON = ROOT / "niveau2_audit_final.json"
OUT_MD = ROOT / "NIVEAU2_RESULTATS.md"


# ============================================================
# CHEMINS PRINCIPAUX
# ============================================================

ANNOTATIONS = (
    ROOT
    / "data"
    / "niveau2_manual_train"
    / "annotations.csv"
)

BEST_MODEL = (
    ROOT
    / "runs"
    / "niveau2_finetune_rotation180"
    / "weights"
    / "best.pt"
)


EVALUATIONS = {
    "niveau1_labels_faibles": (
        ROOT
        / "data"
        / "manual_validation"
        / "evaluation"
        / "metrics_manual.json"
    ),

    "niveau2_humain_seul": (
        ROOT
        / "data"
        / "manual_validation"
        / "evaluation_niveau2_humain"
        / "metrics_manual.json"
    ),

    "niveau2_finetune": (
        ROOT
        / "data"
        / "manual_validation"
        / "evaluation_niveau2_finetune"
        / "metrics_manual.json"
    ),

    "niveau2_hardneg": (
        ROOT
        / "data"
        / "manual_validation"
        / "evaluation_niveau2_hardneg"
        / "metrics_manual.json"
    ),

    "niveau2_rotation180": (
        ROOT
        / "data"
        / "manual_validation"
        / "evaluation_niveau2_rotation180"
        / "metrics_manual.json"
    ),

    "niveau2_compare_mono": (
        ROOT
        / "data"
        / "manual_validation"
        / "evaluation_niveau2_compare_mono"
        / "metrics_manual.json"
    ),

    "niveau2_final_conf003": (
        ROOT
        / "data"
        / "manual_validation"
        / "evaluation_niveau2_final_conf003"
        / "metrics_manual.json"
    ),
}


REQUIRED_PATHS = {
    "annotations_niveau2": ANNOTATIONS,

    "dataset_humain": (
        ROOT
        / "data"
        / "dataset_niveau2_humain"
    ),

    "dataset_hardneg": (
        ROOT
        / "data"
        / "dataset_niveau2_hardneg"
    ),

    "dataset_compare_mono": (
        ROOT
        / "data"
        / "dataset_niveau2_compare_mono"
    ),

    "dataset_compare_multi": (
        ROOT
        / "data"
        / "dataset_niveau2_compare_multi"
    ),

    "split_compare": (
        ROOT
        / "data"
        / "split_niveau2_compare_equilibre.csv"
    ),

    "best_model": BEST_MODEL,

    "multiclass_metrics": (
        ROOT
        / "runs"
        / "niveau2_compare_multi_r180"
        / "evaluation_par_classe"
        / "metrics_par_classe.json"
    ),

    "threshold_calibration": (
        ROOT
        / "runs"
        / "niveau2_finetune_rotation180"
        / "threshold_calibration.json"
    ),
}


# ============================================================
# AUDIT FICHIERS
# ============================================================

print("====================================")
print("AUDIT FINAL NIVEAU 2")
print("====================================")

missing = []

print("\n===== FICHIERS PRINCIPAUX =====")

for name, path in REQUIRED_PATHS.items():

    ok = path.exists()

    print(
        f"{name:<24} : "
        f"{'OK' if ok else 'MANQUANT'}"
    )

    if not ok:
        missing.append(str(path))


# ============================================================
# ANNOTATIONS HUMAINES
# ============================================================

if not ANNOTATIONS.exists():
    raise FileNotFoundError(ANNOTATIONS)

df = pd.read_csv(ANNOTATIONS)


status_counts = (
    df["status"]
    .value_counts()
    .to_dict()
)

visible = df[
    df["status"] == "visible"
].copy()


class_counts = (
    visible["classe_niveau2"]
    .value_counts()
    .to_dict()
)


bbox_complete = (
    visible[
        ["x1", "y1", "x2", "y2"]
    ]
    .notna()
    .all(axis=1)
    .sum()
)


print("\n===== ANNOTATIONS HUMAINES =====")

print(
    "Total :",
    len(df)
)

print(
    "Visible :",
    status_counts.get(
        "visible",
        0
    )
)

print(
    "Non visible :",
    status_counts.get(
        "non_visible",
        0
    )
)

print(
    "Ambigu :",
    status_counts.get(
        "ambigu",
        0
    )
)

print(
    "Bbox complètes visibles :",
    bbox_complete,
    "/",
    len(visible)
)

print("\nClasses visibles :")

for cls, n in class_counts.items():

    print(
        f"  {cls:<15} : {n}"
    )


# ============================================================
# EVALUATIONS SUR TEST HUMAIN
# ============================================================

eval_results = {}


print("\n===== TEST HUMAIN GELÉ =====")

for name, path in EVALUATIONS.items():

    if not path.exists():

        print(
            f"{name:<28} : ABSENT"
        )

        continue


    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


    row = {
        "map50": data.get(
            "map50_manual"
        ),

        "map50_95": data.get(
            "map50_95_manual"
        ),

        "precision_ultralytics": data.get(
            "ultralytics_precision_manual"
        ),

        "recall_ultralytics": data.get(
            "ultralytics_recall_manual"
        ),

        "TP": data.get("TP"),
        "FP": data.get("FP"),
        "FN": data.get("FN"),

        "conf": data.get(
            "conf_threshold_for_tp_fp_fn"
        ),
    }


    eval_results[name] = row


    print(
        f"\n{name}"
    )

    print(
        "  mAP50      :",
        row["map50"]
    )

    print(
        "  mAP50-95   :",
        row["map50_95"]
    )

    print(
        "  recall     :",
        row["recall_ultralytics"]
    )

    print(
        "  conf TP/FP :",
        row["conf"]
    )

    print(
        "  TP/FP/FN   :",
        row["TP"],
        row["FP"],
        row["FN"]
    )


# ============================================================
# MEILLEUR MODELE
# ============================================================

best_name = None
best_map50 = -1


for name, row in eval_results.items():

    # Exclusion du doublon final conf=0.03
    if name == "niveau2_final_conf003":
        continue

    # Ici on veut comparer les expériences Niveau 2.
    if not name.startswith("niveau2_"):
        continue

    value = row["map50"]

    if (
        value is not None
        and value > best_map50
    ):

        best_map50 = value
        best_name = name


print("\n====================================")
print("MODELE RETENU")
print("====================================")

print(
    "Expérience :",
    best_name
)

print(
    "mAP50 test humain :",
    best_map50
)

print(
    "Poids :",
    BEST_MODEL
)

print(
    "Poids présents :",
    BEST_MODEL.exists()
)


# ============================================================
# MULTI-CLASSE
# ============================================================

multi_path = (
    REQUIRED_PATHS[
        "multiclass_metrics"
    ]
)

multi_results = None


if multi_path.exists():

    multi_results = json.loads(
        multi_path.read_text(
            encoding="utf-8"
        )
    )


    print("\n===== MULTI-CLASSE =====")

    for row in multi_results[
        "classes"
    ]:

        print(
            f"{row['class_name']:<15} "
            f"mAP50={row['map50']} "
            f"R={row['recall']}"
        )


# ============================================================
# CALIBRAGE
# ============================================================

threshold_path = (
    REQUIRED_PATHS[
        "threshold_calibration"
    ]
)

threshold_data = None


if threshold_path.exists():

    threshold_data = json.loads(
        threshold_path.read_text(
            encoding="utf-8"
        )
    )


    print("\n===== SEUIL =====")

    print(
        "Meilleur seuil validation :",
        threshold_data.get(
            "best_threshold"
        )
    )

    print(
        "Métriques :",
        threshold_data.get(
            "best_metrics"
        )
    )


# ============================================================
# COMPARAISON NIVEAU 1 / NIVEAU 2
# ============================================================

n1 = eval_results.get(
    "niveau1_labels_faibles"
)

n2 = eval_results.get(
    "niveau2_rotation180"
)


improvement_map50 = None
improvement_map5095 = None


if n1 and n2:

    if (
        n1["map50"]
        and n1["map50"] != 0
    ):

        improvement_map50 = (
            (
                n2["map50"]
                - n1["map50"]
            )
            / n1["map50"]
            * 100
        )


    if (
        n1["map50_95"]
        and n1["map50_95"] != 0
    ):

        improvement_map5095 = (
            (
                n2["map50_95"]
                - n1["map50_95"]
            )
            / n1["map50_95"]
            * 100
        )


print("\n===== GAIN VS NIVEAU 1 =====")

print(
    "Gain mAP50 (%) :",
    improvement_map50
)

print(
    "Gain mAP50-95 (%) :",
    improvement_map5095
)


# ============================================================
# JSON FINAL
# ============================================================

audit = {
    "annotations": {
        "total": len(df),
        "status": status_counts,
        "visible_classes": class_counts,
        "visible_bbox_complete": int(
            bbox_complete
        ),
    },

    "evaluations": eval_results,

    "best_model": {
        "experiment": best_name,
        "weights": str(
            BEST_MODEL.resolve()
        ),
        "map50": best_map50,
    },

    "improvement_vs_level1": {
        "map50_percent": improvement_map50,
        "map50_95_percent": improvement_map5095,
    },

    "threshold_calibration": (
        threshold_data
    ),

    "multiclass": (
        multi_results
    ),

    "missing_paths": missing,
}


OUT_JSON.write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# MARKDOWN RESUME
# ============================================================

lines = []

lines.append(
    "# Niveau 2 — Résultats finaux"
)

lines.append("")

lines.append(
    "## Données humaines"
)

lines.append("")

lines.append(
    f"- 80 images annotées manuellement"
)

lines.append(
    f"- {status_counts.get('visible', 0)} visibles"
)

lines.append(
    f"- {status_counts.get('non_visible', 0)} non visibles"
)

lines.append(
    f"- {status_counts.get('ambigu', 0)} ambiguës"
)

lines.append("")

lines.append(
    "## Meilleur modèle"
)

lines.append("")

lines.append(
    "Fine-tuning du modèle Niveau 1 "
    "sur annotations humaines avec rotation ±180°."
)

lines.append("")

if n2:

    lines.append(
        f"- mAP50 test humain : "
        f"{n2['map50']:.6f}"
    )

    lines.append(
        f"- mAP50-95 test humain : "
        f"{n2['map50_95']:.6f}"
    )

    lines.append(
        f"- Recall Ultralytics : "
        f"{n2['recall_ultralytics']:.3f}"
    )


lines.append("")

lines.append(
    "## Conclusions expérimentales"
)

lines.append("")

lines.append(
    "- Les annotations humaines seules "
    "sont insuffisantes avec le volume actuel."
)

lines.append(
    "- Le fine-tuning depuis le Niveau 1 "
    "est préférable à un entraînement humain seul."
)

lines.append(
    "- Les hard negatives testés n'améliorent "
    "pas la mAP sur le test humain."
)

lines.append(
    "- La rotation ±180° améliore le meilleur "
    "modèle mono-classe."
)

lines.append(
    "- Le multi-classe n'est pas exploitable "
    "avec le nombre actuel d'exemples par classe."
)

lines.append(
    "- Le modèle reste très sous-confiant : "
    "aucune détection exploitable à conf=0.03 "
    "sur le test humain."
)

lines.append("")

lines.append(
    "## Modèle retenu"
)

lines.append("")

lines.append(
    "`runs/niveau2_finetune_rotation180/"
    "weights/best.pt`"
)


OUT_MD.write_text(
    "\n".join(lines),
    encoding="utf-8"
)


print("\n====================================")
print("AUDIT TERMINE")
print("====================================")

print(
    "JSON :",
    OUT_JSON
)

print(
    "Résumé Markdown :",
    OUT_MD
)

print(
    "Fichiers manquants :",
    len(missing)
)

if missing:

    for p in missing:

        print(
            " -",
            p
        )

else:

    print(
        "✅ Aucun artefact principal manquant."
    )
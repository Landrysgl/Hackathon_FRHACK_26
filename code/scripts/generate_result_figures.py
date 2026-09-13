#!/usr/bin/env python3
"""Génère les figures synthétiques livrées avec Pyl-Poil à partir des CSV de référence.

Les figures sont volontairement dérivées de données versionnées dans ``data/`` et
``results/`` afin qu'elles puissent être régénérées sans relancer l'entraînement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def plot_weak_annotation_method(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.set_xlim(0, 1024)
    ax.set_ylim(1024, 0)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((432, 432), 160, 160, fill=False, linewidth=2.0))
    ax.scatter([512], [512], marker="+")
    ax.annotate("Coordonnée ANFR / centre image", (512, 512), xytext=(570, 395), arrowprops={"arrowstyle": "->"})
    ax.text(512, 615, "bbox faible 160 × 160 px", ha="center")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("Niveau 1 — principe de l'annotation faible centrale")
    _save(fig, out / "00_level1_weak_annotation_method.png")

def plot_support_natures(out: Path) -> None:
    df = pd.read_csv(ROOT / "data/reference/supports_yvelines.csv")
    counts = df["NAT_LB_NOM"].fillna("Non renseigné").value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh(counts.index, counts.values)
    ax.set_title("Yvelines — principales natures de supports ANFR")
    ax.set_xlabel("Nombre de supports")
    ax.set_ylabel("Nature ANFR")
    for i, value in enumerate(counts.values):
        ax.text(value, i, f" {int(value)}", va="center", fontsize=8)
    _save(fig, out / "01_yvelines_support_natures.png")


def plot_annotation_coverage(out: Path) -> None:
    df = pd.read_csv(ROOT / "results/metrics/annotation_coverage.csv")
    labels = [
        "N1 test manuel",
        "N2 train lot 1",
        "N2 train lot 2",
        "N2 validation",
        "N2 holdout final",
    ]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    bottom = np.zeros(len(df))
    for col, label in [("visible", "Visible"), ("non_visible", "Non visible"), ("ambigu", "Ambigu")]:
        vals = df[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=label)
        bottom += vals
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Nombre d'images revues")
    ax.set_title("Couverture des annotations humaines")
    ax.legend()
    for i, total in enumerate(df["total"]):
        ax.text(i, float(total) + 2, str(int(total)), ha="center", fontsize=8)
    _save(fig, out / "02_annotation_coverage.png")


def plot_visible_support_types(out: Path) -> None:
    frames = [
        pd.read_csv(ROOT / "data/annotations/level2_train_batch1.csv"),
        pd.read_csv(ROOT / "data/annotations/level2_train_batch2.csv"),
    ]
    df = pd.concat(frames, ignore_index=True)
    visible = df[df["status"].fillna("").astype(str).eq("visible")]
    counts = visible["classe_niveau2"].fillna("non_renseigne").value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 5.1))
    ax.barh(counts.index, counts.values)
    ax.set_title("Niveau 2 — types visuels des 144 supports visibles du train")
    ax.set_xlabel("Nombre d'annotations visibles")
    for i, value in enumerate(counts.values):
        ax.text(value, i, f" {int(value)}", va="center", fontsize=8)
    _save(fig, out / "03_visible_support_types_train.png")


def plot_initial_experiments(out: Path) -> None:
    df = pd.read_csv(ROOT / "results/metrics/experiment_comparison.csv")
    df = df[df["evaluation_split"].eq("initial_manual_test_20")].copy()
    labels_map = {
        "niveau1_baseline": "N1 baseline faible",
        "niveau2_compare_mono": "N2 comparaison mono",
        "niveau2_final_conf003": "N2 rotation 180° (conf .03)",
        "niveau2_finetune": "N2 fine-tune N1",
        "niveau2_hardneg": "N2 hard negatives",
        "niveau2_humain": "N2 humain seul",
        "niveau2_renforce": "N2 renforcé init. N1",
        "niveau2_rotation180": "N2 rotation 180°",
    }
    df["label"] = df["experiment"].map(labels_map).fillna(df["experiment"])
    df = df.sort_values("map50", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    ax.barh(df["label"], df["map50"])
    ax.set_title("Comparaison d'expériences sur le même test manuel initial")
    ax.set_xlabel("mAP@50")
    for i, value in enumerate(df["map50"]):
        ax.text(value, i, f" {value:.4f}", va="center", fontsize=8)
    ax.text(
        0.01,
        -0.15,
        "Comparaison historique sur 13 images utilisables (10 visibles + 3 non visibles).",
        transform=ax.transAxes,
        fontsize=8,
    )
    _save(fig, out / "04_level2_experiments_initial_test_map50.png")


def plot_final_holdout(out: Path) -> None:
    metrics = json.loads((ROOT / "results/metrics/level2_holdout_final.json").read_text(encoding="utf-8"))
    values = [
        metrics["map50_manual"],
        metrics["map50_95_manual"],
        metrics["ultralytics_precision_manual"],
        metrics["ultralytics_recall_manual"],
    ]
    labels = ["mAP@50", "mAP@50-95", "Précision", "Rappel"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("Modèle final — holdout indépendant (23 images utilisables)")
    ax.set_ylabel("Score")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}", ha="center", va="bottom")
    _save(fig, out / "05_level2_final_holdout_metrics.png")


def plot_training_curve(out: Path) -> None:
    df = pd.read_csv(ROOT / "results/training_history/niveau2_renforce_coco_sans_rotation/results.csv")
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50 validation")
    ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@50-95 validation")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Score")
    ax.set_title("Entraînement du modèle final — YOLOv8n COCO, sans rotation forcée")
    ax.legend()
    _save(fig, out / "06_final_training_curve.png")


def plot_distance_calibration(out: Path) -> None:
    df = pd.read_csv(ROOT / "data/level3/anfr_visual_offset_calibration.csv")
    distances = pd.to_numeric(df["distance_to_nearest_anfr_m"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    ax.hist(distances, bins=24)
    ax.axvline(150.0, linestyle="--", linewidth=1.8, label="Seuil candidats = 150 m")
    ax.axvline(float(distances.max()), linestyle=":", linewidth=1.5, label=f"Maximum calibration = {distances.max():.1f} m")
    ax.set_xlabel("Distance centre visuel → support ANFR le plus proche (m)")
    ax.set_ylabel("Nombre de supports visibles")
    ax.set_title("Calibration du filtre géographique sur 175 supports visibles")
    ax.legend()
    _save(fig, out / "07_anfr_distance_calibration.png")


def plot_level3_funnel(out: Path) -> None:
    summary = json.loads((ROOT / "results/metrics/level3_summary.json").read_text(encoding="utf-8"))
    labels = ["Détections\nbrutes", "≥ 150 m", "Après\ndéduplication", "Plausibles\n(revue humaine)"]
    values = [
        summary["detections_brutes"],
        summary["detections_ge150m"],
        summary["candidats_uniques"],
        summary["plausibles"],
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Nombre")
    ax.set_title("Niveau 3 — réduction progressive des candidats")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value, str(int(value)), ha="center", va="bottom")
    _save(fig, out / "08_level3_candidate_funnel.png")


def plot_level3_by_zone(out: Path) -> None:
    df = pd.read_csv(ROOT / "results/level3/summary_by_zone.csv")
    labels = ["Dense", "Intermédiaire", "Peu dense"]
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.5, 5.1))
    ax.bar(x - width, df["plausibles"], width, label="Plausibles")
    ax.bar(x, df["faux_positifs"], width, label="Faux positifs")
    ax.bar(x + width, df["incertains"], width, label="Incertains")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Candidats uniques")
    ax.set_title("Niveau 3 — revue humaine selon le contexte géographique")
    ax.legend()
    _save(fig, out / "09_level3_review_by_zone.png")


def _fit_image(img: Image.Image, width: int, height: int) -> Image.Image:
    copy = img.convert("RGB")
    copy.thumbnail((width, height))
    canvas = Image.new("RGB", (width, height), "white")
    x = (width - copy.width) // 2
    y = (height - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def make_review_examples(out: Path) -> None:
    source_dir = ROOT / "results/figures/level3/review_examples"
    examples = [
        (source_dir / "plausible_D003.jpg", "D003 - plausible - mat"),
        (source_dir / "false_positive_I001.jpg", "I001 - faux positif"),
        (source_dir / "uncertain_P004.jpg", "P004 - incertain"),
    ]
    w, h, header = 520, 520, 52
    montage = Image.new("RGB", (w * len(examples), h + header), "white")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    for i, (path, label) in enumerate(examples):
        img = _fit_image(Image.open(path), w, h)
        montage.paste(img, (i*w, header))
        draw.text((i*w + 12, 18), label, fill="black", font=font)
    output = out / "10_level3_review_examples.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère toutes les figures de résultats Pyl-Poil.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/figures/overview",
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    plot_weak_annotation_method(out)
    plot_support_natures(out)
    plot_annotation_coverage(out)
    plot_visible_support_types(out)
    plot_initial_experiments(out)
    plot_final_holdout(out)
    plot_training_curve(out)
    plot_distance_calibration(out)
    plot_level3_funnel(out)
    plot_level3_by_zone(out)
    make_review_examples(out)

    print(f"11 figures générées dans {out}")


if __name__ == "__main__":
    main()

from pathlib import Path
import math
import json

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SELECTION_PATH = DATA_DIR / "supports_selectionnes_niveau1.csv"
IMAGES_DIR = DATA_DIR / "images_ign_yvelines"
DATASET_DIR = DATA_DIR / "dataset_yolo"
MANIFEST_PATH = DATASET_DIR / "split_manifest.csv"
MANUAL_ANNOTATIONS = DATA_DIR / "manual_validation" / "annotations.csv"
MANUAL_METRICS = (
    DATA_DIR
    / "manual_validation"
    / "evaluation"
    / "metrics_manual.json"
)
WEIGHTS = ROOT / "runs" / "baseline_weak" / "weights" / "best.pt"

IMAGE_SIZE = 1024
SPATIAL_WARNING_M = 200.0


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


erreurs = []
warnings = []

print("===== AUDIT FINAL NIVEAU 1 =====")

if not SELECTION_PATH.exists():
    erreurs.append("supports_selectionnes_niveau1.csv absent")
else:
    selection = pd.read_csv(
        SELECTION_PATH,
        dtype={"SUP_ID": "string"},
    )

    print("\n[Selection]")
    print("Lignes :", len(selection))
    print("SUP_ID uniques :", selection["SUP_ID"].nunique())

    if len(selection) != 300:
        erreurs.append("La sélection ne contient pas 300 lignes.")
    if selection["SUP_ID"].nunique() != 300:
        erreurs.append("La sélection ne contient pas 300 SUP_ID uniques.")

if IMAGES_DIR.exists():
    images = sorted(IMAGES_DIR.glob("support_*.jpg"))
    print("\n[Images source]")
    print("JPEG :", len(images))

    if len(images) != 300:
        erreurs.append(
            f"Le dossier source contient {len(images)} images au lieu de 300."
        )

    for p in images:
        try:
            with Image.open(p) as im:
                im.load()
                if im.size != (IMAGE_SIZE, IMAGE_SIZE):
                    erreurs.append(f"Dimensions invalides : {p.name} {im.size}")
        except Exception as exc:
            erreurs.append(f"Image illisible : {p.name}: {exc}")
else:
    erreurs.append("Dossier images_ign_yvelines absent.")

if not MANIFEST_PATH.exists():
    erreurs.append("split_manifest.csv absent.")
else:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype={"SUP_ID": "string"},
    )

    print("\n[Splits]")
    print(manifest["split"].value_counts())

    if len(manifest) != 300:
        erreurs.append("Le manifest ne contient pas 300 lignes.")

    noms_par_split = {}

    for split in ("train", "val", "test"):
        image_dir = DATASET_DIR / "images" / split
        label_dir = DATASET_DIR / "labels" / split

        images_split = sorted(image_dir.glob("*.jpg"))
        labels_split = sorted(label_dir.glob("*.txt"))

        noms_images = {p.stem for p in images_split}
        noms_labels = {p.stem for p in labels_split}

        noms_par_split[split] = noms_images

        print(
            f"{split}: {len(images_split)} images / "
            f"{len(labels_split)} labels"
        )

        if noms_images != noms_labels:
            erreurs.append(
                f"Mismatch images/labels dans {split}."
            )

        for label in labels_split:
            lignes = [
                l.strip()
                for l in label.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]

            if len(lignes) != 1:
                erreurs.append(
                    f"{label.name}: attendu 1 label faible, trouvé {len(lignes)}."
                )
                continue

            parts = lignes[0].split()
            if len(parts) != 5:
                erreurs.append(
                    f"{label.name}: format YOLO invalide."
                )
                continue

            try:
                cls = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except Exception:
                erreurs.append(
                    f"{label.name}: valeurs non numériques."
                )
                continue

            if cls != 0 or not all(0 <= v <= 1 for v in vals):
                erreurs.append(
                    f"{label.name}: classe/coordonnées invalides."
                )

    if (
        noms_par_split["train"] & noms_par_split["val"]
        or noms_par_split["train"] & noms_par_split["test"]
        or noms_par_split["val"] & noms_par_split["test"]
    ):
        erreurs.append("Chevauchement de fichiers entre splits.")

    manifest = manifest.reset_index(drop=True)
    min_dist = float("inf")
    min_pair = None

    for i in range(len(manifest)):
        for j in range(i + 1, len(manifest)):
            if manifest.loc[i, "split"] == manifest.loc[j, "split"]:
                continue

            d = haversine_m(
                float(manifest.loc[i, "latitude"]),
                float(manifest.loc[i, "longitude"]),
                float(manifest.loc[j, "latitude"]),
                float(manifest.loc[j, "longitude"]),
            )

            if d < min_dist:
                min_dist = d
                min_pair = (
                    manifest.loc[i, "SUP_ID"],
                    manifest.loc[j, "SUP_ID"],
                    manifest.loc[i, "split"],
                    manifest.loc[j, "split"],
                )

    print(f"Distance inter-split minimale : {min_dist:.1f} m")
    print("Paire :", min_pair)

    if min_dist < SPATIAL_WARNING_M:
        warnings.append(
            f"Deux supports de splits différents sont à {min_dist:.1f} m."
        )

if MANUAL_ANNOTATIONS.exists():
    ann = pd.read_csv(
        MANUAL_ANNOTATIONS,
        dtype={"SUP_ID": "string"},
    )
    statuses = ann["status"].fillna("").astype(str).str.strip()

    print("\n[Validation manuelle]")
    print(statuses.replace("", "NON_ANNOTE").value_counts())

    if (statuses == "").any():
        warnings.append(
            f"{int((statuses == '').sum())} annotations manuelles restent vides."
        )
else:
    warnings.append("annotations.csv de validation manuelle absent.")

print("\n[Modèle]")
print("best.pt :", "OK" if WEIGHTS.exists() else "ABSENT")
if not WEIGHTS.exists():
    warnings.append("Le modèle baseline best.pt n'existe pas encore.")

if MANUAL_METRICS.exists():
    metrics = json.loads(
        MANUAL_METRICS.read_text(encoding="utf-8")
    )

    print("\n[Métriques humaines]")
    for key in [
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "map50_manual",
        "map50_95_manual",
    ]:
        print(f"{key}: {metrics.get(key)}")
else:
    warnings.append(
        "Les métriques humaines n'ont pas encore été générées."
    )

print("\n==================================")
print("RESULTAT AUDIT")
print("==================================")

if erreurs:
    print(f"❌ {len(erreurs)} ERREUR(S)")
    for e in erreurs:
        print(" -", e)
else:
    print("✅ Aucune erreur structurelle détectée.")

if warnings:
    print(f"\n⚠️ {len(warnings)} AVERTISSEMENT(S)")
    for w in warnings:
        print(" -", w)
else:
    print("✅ Aucun avertissement.")

if erreurs:
    raise SystemExit(1)

print("\n✅ Niveau 1 structurellement cohérent.")

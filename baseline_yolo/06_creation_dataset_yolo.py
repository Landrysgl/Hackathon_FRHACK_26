from pathlib import Path
import math
import random
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SELECTION_PATH = DATA_DIR / "supports_selectionnes_niveau1.csv"
SOURCE_IMAGES_DIR = DATA_DIR / "images_ign_yvelines"
DATASET_DIR = DATA_DIR / "dataset_yolo"

IMAGE_SIZE = 1024
WEAK_BOX_SIZE_PX = 160
SPATIAL_GROUP_DISTANCE_M = 200.0
RANDOM_SEED = 42

TARGET_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


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


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


if not SELECTION_PATH.exists():
    raise FileNotFoundError(
        f"{SELECTION_PATH} introuvable. Lance d'abord le script 04."
    )

supports = pd.read_csv(
    SELECTION_PATH,
    dtype={"SUP_ID": "string"},
).sort_values("SUP_ID").reset_index(drop=True)

if len(supports) != 300 or supports["SUP_ID"].nunique() != 300:
    raise ValueError(
        "La sélection doit contenir exactement 300 SUP_ID distincts."
    )

for col in ("latitude", "longitude"):
    supports[col] = pd.to_numeric(supports[col], errors="coerce")

if supports[["latitude", "longitude"]].isna().any().any():
    raise ValueError("Coordonnées manquantes dans la sélection.")

images_manquantes = [
    sup_id
    for sup_id in supports["SUP_ID"]
    if not (SOURCE_IMAGES_DIR / f"support_{sup_id}.jpg").exists()
]

if images_manquantes:
    raise FileNotFoundError(
        f"{len(images_manquantes)} images manquent. "
        f"Exemples : {images_manquantes[:10]}"
    )

print("===== CONSTRUCTION DES GROUPES SPATIAUX =====")

n = len(supports)
uf = UnionFind(n)

coords = list(
    zip(
        supports["latitude"].astype(float),
        supports["longitude"].astype(float),
    )
)

for i in range(n):
    lat1, lon1 = coords[i]
    for j in range(i + 1, n):
        lat2, lon2 = coords[j]
        if haversine_m(lat1, lon1, lat2, lon2) < SPATIAL_GROUP_DISTANCE_M:
            uf.union(i, j)

components = {}
for idx in range(n):
    racine = uf.find(idx)
    components.setdefault(racine, []).append(idx)

groupes = list(components.values())

rng = random.Random(RANDOM_SEED)
rng.shuffle(groupes)
groupes.sort(key=len, reverse=True)

print("Nombre de groupes spatiaux :", len(groupes))
print("Taille du plus grand groupe :", max(map(len, groupes)))

targets = {
    split: round(ratio * n)
    for split, ratio in TARGET_RATIOS.items()
}
targets["train"] += n - sum(targets.values())

assignments = {}
counts = {split: 0 for split in TARGET_RATIOS}

for groupe in groupes:
    remaining = {
        split: targets[split] - counts[split]
        for split in TARGET_RATIOS
    }

    candidats = [
        split
        for split, reste in remaining.items()
        if reste > 0
    ]

    if candidats:
        split_choisi = max(
            candidats,
            key=lambda s: (
                remaining[s],
                TARGET_RATIOS[s],
            ),
        )
    else:
        split_choisi = min(
            TARGET_RATIOS,
            key=lambda s: counts[s] / max(targets[s], 1),
        )

    for idx in groupe:
        assignments[idx] = split_choisi

    counts[split_choisi] += len(groupe)

supports["split"] = [
    assignments[i]
    for i in range(n)
]

print("\n===== SPLIT =====")
print(supports["split"].value_counts())

for split in TARGET_RATIOS:
    if (supports["split"] == split).sum() == 0:
        raise ValueError(f"Split vide : {split}")

if DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)

for split in TARGET_RATIOS:
    (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

weak_norm = WEAK_BOX_SIZE_PX / IMAGE_SIZE
label_line = f"0 0.5 0.5 {weak_norm:.8f} {weak_norm:.8f}\n"

for row in supports.itertuples(index=False):
    sup_id = str(row.SUP_ID)
    split = row.split

    src = SOURCE_IMAGES_DIR / f"support_{sup_id}.jpg"
    dst = DATASET_DIR / "images" / split / src.name
    shutil.copy2(src, dst)

    label_path = DATASET_DIR / "labels" / split / f"support_{sup_id}.txt"
    label_path.write_text(label_line, encoding="utf-8")

manifest_path = DATASET_DIR / "split_manifest.csv"
supports.to_csv(manifest_path, index=False)

yaml_path = DATASET_DIR / "dataset.yaml"
yaml_path.write_text(
    "\n".join([
        f"path: {DATASET_DIR.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "nc: 1",
        "names:",
        "  0: antenne",
        "",
    ]),
    encoding="utf-8",
)

print("\n===== CONTROLE DES FUITES SPATIALES =====")

min_inter_split = float("inf")
paire_min = None

for i in range(n):
    for j in range(i + 1, n):
        if supports.loc[i, "split"] == supports.loc[j, "split"]:
            continue

        d = haversine_m(
            supports.loc[i, "latitude"],
            supports.loc[i, "longitude"],
            supports.loc[j, "latitude"],
            supports.loc[j, "longitude"],
        )

        if d < min_inter_split:
            min_inter_split = d
            paire_min = (
                supports.loc[i, "SUP_ID"],
                supports.loc[j, "SUP_ID"],
                supports.loc[i, "split"],
                supports.loc[j, "split"],
            )

print(f"Distance minimale entre deux splits : {min_inter_split:.1f} m")
print("Paire la plus proche :", paire_min)

if min_inter_split < SPATIAL_GROUP_DISTANCE_M:
    raise RuntimeError(
        "Une fuite spatiale a été détectée malgré le regroupement."
    )

print("\n===== DATASET YOLO CREE =====")
print("Dossier :", DATASET_DIR)
print("Manifest :", manifest_path)
print("YAML :", yaml_path)
print(
    "Label faible : boîte centrale "
    f"{WEAK_BOX_SIZE_PX}x{WEAK_BOX_SIZE_PX}px "
    f"({weak_norm:.5f} x {weak_norm:.5f} normalisé)"
)
print(
    "⚠️ Ces boîtes sont des annotations faibles, "
    "pas une vérité terrain humaine."
)

from pathlib import Path
import math
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

ANNOTATIONS = DATA / "niveau2_manual_train" / "annotations.csv"
SOURCE_IMAGES = DATA / "niveau2_manual_train" / "images"

OUT = DATA / "dataset_niveau2_humain"

DISTANCE_GROUP_M = 200.0
TARGET_VAL_RATIO = 0.20


# ============================================================
# DISTANCE HAVERSINE
# ============================================================

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.parent[rb] = ra


# ============================================================
# CHARGEMENT
# ============================================================

df = pd.read_csv(ANNOTATIONS)

df["status"] = (
    df["status"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

usable = df[
    df["status"].isin(["visible", "non_visible"])
].copy()

print("===== ANNOTATIONS UTILISABLES =====")
print("Total :", len(usable))
print("Visible :", (usable["status"] == "visible").sum())
print("Non visible :", (usable["status"] == "non_visible").sum())

if len(usable) == 0:
    raise RuntimeError("Aucune annotation exploitable.")


# ============================================================
# GROUPES SPATIAUX
# ============================================================

usable = usable.reset_index(drop=True)

uf = UnionFind(len(usable))

for i in range(len(usable)):
    for j in range(i + 1, len(usable)):

        d = haversine_m(
            float(usable.loc[i, "latitude"]),
            float(usable.loc[i, "longitude"]),
            float(usable.loc[j, "latitude"]),
            float(usable.loc[j, "longitude"]),
        )

        if d < DISTANCE_GROUP_M:
            uf.union(i, j)


usable["spatial_group"] = [
    uf.find(i)
    for i in range(len(usable))
]

# Réindexe proprement les groupes
group_map = {
    old: new
    for new, old in enumerate(
        sorted(usable["spatial_group"].unique())
    )
}

usable["spatial_group"] = (
    usable["spatial_group"]
    .map(group_map)
)


print("\n===== GROUPES SPATIAUX =====")
print(
    "Nombre de groupes :",
    usable["spatial_group"].nunique()
)

print(
    "Plus grand groupe :",
    usable.groupby("spatial_group").size().max()
)


# ============================================================
# CHOIX DES GROUPES DE VALIDATION
# ============================================================

groups = []

for gid, g in usable.groupby("spatial_group"):

    groups.append(
        {
            "group": gid,
            "n": len(g),
            "visible": (g["status"] == "visible").sum(),
            "negative": (g["status"] == "non_visible").sum(),
        }
    )

groups_df = pd.DataFrame(groups)

target_val = round(len(usable) * TARGET_VAL_RATIO)
target_visible = round(
    (usable["status"] == "visible").sum()
    * TARGET_VAL_RATIO
)
target_negative = round(
    (usable["status"] == "non_visible").sum()
    * TARGET_VAL_RATIO
)


# Tri : groupes les plus utiles d'abord
groups_df = groups_df.sort_values(
    ["n", "visible", "negative"],
    ascending=False
).reset_index(drop=True)


val_groups = set()

val_n = 0
val_visible = 0
val_negative = 0


def score_after(row):
    new_n = val_n + row["n"]
    new_v = val_visible + row["visible"]
    new_neg = val_negative + row["negative"]

    return (
        abs(new_n - target_val)
        + 0.5 * abs(new_v - target_visible)
        + 0.5 * abs(new_neg - target_negative)
    )


remaining = groups_df.copy()

while len(remaining) > 0:

    current_score = (
        abs(val_n - target_val)
        + 0.5 * abs(val_visible - target_visible)
        + 0.5 * abs(val_negative - target_negative)
    )

    remaining = remaining.copy()
    remaining["score"] = remaining.apply(
        score_after,
        axis=1
    )

    best_idx = remaining["score"].idxmin()
    best = remaining.loc[best_idx]

    # Si ajouter un groupe n'améliore plus rien,
    # on s'arrête sauf si la validation est encore trop petite.
    if (
        best["score"] >= current_score
        and val_n >= max(1, target_val - 2)
    ):
        break

    gid = int(best["group"])

    val_groups.add(gid)

    val_n += int(best["n"])
    val_visible += int(best["visible"])
    val_negative += int(best["negative"])

    remaining = remaining.drop(best_idx)


usable["split_niveau2"] = usable["spatial_group"].apply(
    lambda x: "val" if x in val_groups else "train"
)


# ============================================================
# CONTROLE SPLIT
# ============================================================

print("\n===== SPLIT SPATIAL =====")

print(
    usable.groupby(
        ["split_niveau2", "status"]
    ).size()
)

print("\nTotal par split :")
print(
    usable["split_niveau2"]
    .value_counts()
)


# ============================================================
# DISTANCE MINIMALE TRAIN / VAL
# ============================================================

train = usable[
    usable["split_niveau2"] == "train"
]

val = usable[
    usable["split_niveau2"] == "val"
]

min_dist = float("inf")
closest = None

for _, a in train.iterrows():
    for _, b in val.iterrows():

        d = haversine_m(
            float(a["latitude"]),
            float(a["longitude"]),
            float(b["latitude"]),
            float(b["longitude"]),
        )

        if d < min_dist:
            min_dist = d
            closest = (
                str(a["SUP_ID"]),
                str(b["SUP_ID"]),
            )


print("\n===== CONTROLE FUITE SPATIALE =====")
print(
    f"Distance minimale train/val : "
    f"{min_dist:.1f} m"
)
print("Paire la plus proche :", closest)

if min_dist < DISTANCE_GROUP_M:
    raise RuntimeError(
        "Fuite spatiale détectée."
    )


# ============================================================
# RECONSTRUCTION DU DATASET
# ============================================================

if OUT.exists():
    shutil.rmtree(OUT)

for split in ["train", "val"]:

    (OUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# IMAGES + LABELS
# ============================================================

for _, row in usable.iterrows():

    split = row["split_niveau2"]
    image_name = str(row["image"])

    src = SOURCE_IMAGES / image_name
    dst = OUT / "images" / split / image_name

    if not src.exists():
        raise FileNotFoundError(src)

    shutil.copy2(src, dst)

    label_path = (
        OUT
        / "labels"
        / split
        / f"{Path(image_name).stem}.txt"
    )

    # Négatif réel
    if row["status"] == "non_visible":

        label_path.write_text(
            "",
            encoding="utf-8"
        )

        continue

    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])

    xc = ((x1 + x2) / 2) / 1024
    yc = ((y1 + y2) / 2) / 1024

    w = (x2 - x1) / 1024
    h = (y2 - y1) / 1024

    label_path.write_text(
        f"0 {xc:.6f} {yc:.6f} "
        f"{w:.6f} {h:.6f}\n",
        encoding="utf-8"
    )


# ============================================================
# YAML
# ============================================================

yaml_text = f"""path: {OUT}
train: images/train
val: images/val

names:
  0: support
"""

(OUT / "dataset.yaml").write_text(
    yaml_text,
    encoding="utf-8"
)


# ============================================================
# MANIFEST
# ============================================================

usable.to_csv(
    OUT / "manifest_niveau2.csv",
    index=False
)


# ============================================================
# AUDIT FINAL
# ============================================================

print("\n===== DATASET NIVEAU 2 SPATIAL =====")

for split in ["train", "val"]:

    images = list(
        (OUT / "images" / split)
        .glob("*.jpg")
    )

    labels = list(
        (OUT / "labels" / split)
        .glob("*.txt")
    )

    print(
        f"{split}: "
        f"{len(images)} images / "
        f"{len(labels)} labels"
    )


print("\nDataset :", OUT)
print(
    "YAML :",
    OUT / "dataset.yaml"
)

print(
    "\n✅ Split spatial sans fuite < 200 m."
)

print(
    "✅ Uniquement annotations humaines."
)

print(
    "✅ Images ambiguës exclues."
)

print(
    "✅ Non-visible conservés comme négatifs."
)
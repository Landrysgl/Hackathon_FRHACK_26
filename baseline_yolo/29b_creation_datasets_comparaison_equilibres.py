from pathlib import Path
import math
import shutil

import pandas as pd
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parent

ANNOTATIONS = (
    ROOT
    / "data"
    / "niveau2_manual_train"
    / "annotations.csv"
)

SOURCE_IMAGES = (
    ROOT
    / "data"
    / "niveau2_manual_train"
    / "images"
)

CURRENT_DATASET = (
    ROOT
    / "data"
    / "dataset_niveau2_humain"
)

OUT_MONO = (
    ROOT
    / "data"
    / "dataset_niveau2_compare_mono"
)

OUT_MULTI = (
    ROOT
    / "data"
    / "dataset_niveau2_compare_multi"
)

MANIFEST_OUT = (
    ROOT
    / "data"
    / "split_niveau2_compare_equilibre.csv"
)

DISTANCE_M = 200.0


MULTI_MAPPING = {
    "batiment": 0,
    "pylone": 1,
    "mat": 1,
    "chateau_eau": 2,
    "autre_support": 3,
}

MULTI_NAMES = {
    0: "batiment",
    1: "pylone_mat",
    2: "chateau_eau",
    3: "autre_support",
}


# ============================================================
# DISTANCE
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
# CHARGEMENT
# ============================================================

df = pd.read_csv(ANNOTATIONS)

df = df[
    df["status"].isin(
        ["visible", "non_visible"]
    )
].copy()

df = df.reset_index(drop=True)

print("Images utilisables :", len(df))


# ============================================================
# SPLIT ACTUEL
# ============================================================

train_names = {
    p.name
    for p in (
        CURRENT_DATASET
        / "images"
        / "train"
    ).glob("*.jpg")
}

val_names = {
    p.name
    for p in (
        CURRENT_DATASET
        / "images"
        / "val"
    ).glob("*.jpg")
}


def current_split(name):

    if name in train_names:
        return "train"

    if name in val_names:
        return "val"

    raise RuntimeError(
        f"Image absente du split actuel : {name}"
    )


df["split_actuel"] = (
    df["image"]
    .map(current_split)
)


# ============================================================
# CLASSE MULTI REGROUPEE
# ============================================================

def mapped_class(row):

    if row["status"] != "visible":
        return "negative"

    c = row["classe_niveau2"]

    if c == "pylone" or c == "mat":
        return "pylone_mat"

    return c


df["classe_multi"] = df.apply(
    mapped_class,
    axis=1
)


# ============================================================
# UNION-FIND POUR GROUPES SPATIAUX
# ============================================================

n = len(df)

parent = list(range(n))


def find(x):

    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]

    return x


def union(a, b):

    ra = find(a)
    rb = find(b)

    if ra != rb:
        parent[rb] = ra


for i in range(n):

    for j in range(i + 1, n):

        d = haversine_m(
            float(df.loc[i, "latitude"]),
            float(df.loc[i, "longitude"]),
            float(df.loc[j, "latitude"]),
            float(df.loc[j, "longitude"]),
        )

        if d <= DISTANCE_M:
            union(i, j)


roots = [find(i) for i in range(n)]

root_to_gid = {}

for root in roots:

    if root not in root_to_gid:
        root_to_gid[root] = len(root_to_gid)


df["group_id"] = [
    root_to_gid[r]
    for r in roots
]


# ============================================================
# VERIFIER QUE LE SPLIT ACTUEL RESPECTE LES GROUPES
# ============================================================

for gid, group in df.groupby("group_id"):

    splits = set(
        group["split_actuel"]
    )

    if len(splits) != 1:

        raise RuntimeError(
            f"Groupe spatial {gid} "
            f"réparti entre train/val : {splits}"
        )


group_split = {
    gid: group["split_actuel"].iloc[0]
    for gid, group
    in df.groupby("group_id")
}


# ============================================================
# STATS
# ============================================================

def print_val_stats(data, title):

    val = data[
        data["split_compare"] == "val"
    ]

    print(f"\n===== {title} =====")

    print(
        "Images val :",
        len(val)
    )

    print(
        "Négatives :",
        (val["status"] == "non_visible").sum()
    )

    print("\nClasses visibles :")

    print(
        val.loc[
            val["status"] == "visible",
            "classe_multi"
        ].value_counts()
    )


# ============================================================
# CHERCHER UN SWAP DE GROUPES
# ============================================================

train_groups = [
    gid
    for gid, split
    in group_split.items()
    if split == "train"
]

val_groups = [
    gid
    for gid, split
    in group_split.items()
    if split == "val"
]


# Groupe train contenant au moins un château d'eau visible
chateau_train_groups = []

for gid in train_groups:

    g = df[
        df["group_id"] == gid
    ]

    condition = (
        (g["status"] == "visible")
        & (
            g["classe_multi"]
            == "chateau_eau"
        )
    )

    if condition.any():
        chateau_train_groups.append(gid)


candidates = []


for train_gid in chateau_train_groups:

    train_group = df[
        df["group_id"] == train_gid
    ]

    for val_gid in val_groups:

        val_group = df[
            df["group_id"] == val_gid
        ]

        # On conserve exactement 12 images val
        if len(train_group) != len(val_group):
            continue

        temp = df.copy()

        temp["split_compare"] = (
            temp["split_actuel"]
        )

        temp.loc[
            temp["group_id"] == train_gid,
            "split_compare"
        ] = "val"

        temp.loc[
            temp["group_id"] == val_gid,
            "split_compare"
        ] = "train"

        val = temp[
            temp["split_compare"] == "val"
        ]


        # Même nombre de négatives
        if (
            val["status"]
            == "non_visible"
        ).sum() != 2:
            continue


        visible_counts = (
            val.loc[
                val["status"] == "visible",
                "classe_multi"
            ]
            .value_counts()
            .to_dict()
        )


        required = [
            "batiment",
            "pylone_mat",
            "chateau_eau",
            "autre_support",
        ]

        if any(
            visible_counts.get(c, 0) < 1
            for c in required
        ):
            continue


        # Score :
        # 1. favoriser le minimum par classe
        # 2. favoriser un split équilibré
        counts = [
            visible_counts.get(c, 0)
            for c in required
        ]

        score = (
            min(counts),
            -(
                max(counts)
                - min(counts)
            ),
            visible_counts.get(
                "chateau_eau",
                0
            ),
        )


        candidates.append(
            (
                score,
                train_gid,
                val_gid,
                temp,
                visible_counts,
            )
        )


if not candidates:

    raise RuntimeError(
        "Aucun échange spatial valide trouvé."
    )


candidates.sort(
    key=lambda x: x[0],
    reverse=True
)

(
    best_score,
    best_train_gid,
    best_val_gid,
    df_final,
    best_counts,
) = candidates[0]


print("\n====================================")
print("REEQUILIBRAGE CHOISI")
print("====================================")

print(
    "Groupe TRAIN -> VAL :",
    best_train_gid
)

print(
    df_final.loc[
        df_final["group_id"]
        == best_train_gid,
        [
            "SUP_ID",
            "image",
            "status",
            "classe_multi",
        ]
    ].to_string(index=False)
)

print(
    "\nGroupe VAL -> TRAIN :",
    best_val_gid
)

print(
    df_final.loc[
        df_final["group_id"]
        == best_val_gid,
        [
            "SUP_ID",
            "image",
            "status",
            "classe_multi",
        ]
    ].to_string(index=False)
)


# ============================================================
# DISTANCE MIN TRAIN / VAL
# ============================================================

train_final = df_final[
    df_final["split_compare"]
    == "train"
]

val_final = df_final[
    df_final["split_compare"]
    == "val"
]


min_distance = float("inf")
closest_pair = None


for _, a in train_final.iterrows():

    for _, b in val_final.iterrows():

        d = haversine_m(
            float(a["latitude"]),
            float(a["longitude"]),
            float(b["latitude"]),
            float(b["longitude"]),
        )

        if d < min_distance:

            min_distance = d

            closest_pair = (
                a["SUP_ID"],
                b["SUP_ID"],
            )


print(
    "\nDistance minimale train/val :",
    f"{min_distance:.1f} m"
)

print(
    "Paire la plus proche :",
    closest_pair
)

if min_distance <= DISTANCE_M:

    raise RuntimeError(
        "Séparation spatiale insuffisante."
    )


print_val_stats(
    df_final,
    "NOUVELLE VALIDATION"
)


# ============================================================
# MANIFEST
# ============================================================

df_final[
    [
        "SUP_ID",
        "image",
        "status",
        "classe_niveau2",
        "classe_multi",
        "group_id",
        "split_compare",
        "latitude",
        "longitude",
    ]
].to_csv(
    MANIFEST_OUT,
    index=False
)

print(
    "\nManifest :",
    MANIFEST_OUT
)


# ============================================================
# CREATION DATASETS
# ============================================================

def create_dataset(
    output_dir,
    multiclass=False
):

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:

        (
            output_dir
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            output_dir
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )


    for _, row in df_final.iterrows():

        split = row["split_compare"]

        source_image = (
            SOURCE_IMAGES
            / row["image"]
        )

        if not source_image.exists():
            raise FileNotFoundError(
                source_image
            )

        dest_image = (
            output_dir
            / "images"
            / split
            / row["image"]
        )

        dest_label = (
            output_dir
            / "labels"
            / split
            / (
                Path(row["image"]).stem
                + ".txt"
            )
        )

        shutil.copy2(
            source_image,
            dest_image
        )


        # Image négative
        if row["status"] == "non_visible":

            dest_label.write_text(
                "",
                encoding="utf-8"
            )

            continue


        with Image.open(
            source_image
        ) as im:

            width, height = im.size


        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])


        xc = (
            (x1 + x2) / 2
        ) / width

        yc = (
            (y1 + y2) / 2
        ) / height

        bw = (
            x2 - x1
        ) / width

        bh = (
            y2 - y1
        ) / height


        if multiclass:

            original = row[
                "classe_niveau2"
            ]

            class_id = (
                MULTI_MAPPING[
                    original
                ]
            )

        else:

            class_id = 0


        dest_label.write_text(
            (
                f"{class_id} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}\n"
            ),
            encoding="utf-8"
        )


    if multiclass:

        names = MULTI_NAMES

    else:

        names = {
            0: "support"
        }


    yaml_data = {
        "path": str(
            output_dir.resolve()
        ),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }


    (
        output_dir
        / "dataset.yaml"
    ).write_text(
        yaml.safe_dump(
            yaml_data,
            sort_keys=False,
            allow_unicode=True
        ),
        encoding="utf-8"
    )


create_dataset(
    OUT_MONO,
    multiclass=False
)

create_dataset(
    OUT_MULTI,
    multiclass=True
)


print("\n====================================")
print("DATASETS DE COMPARAISON CREES")
print("====================================")

print(
    "Mono :",
    OUT_MONO
)

print(
    "Multi :",
    OUT_MULTI
)

print(
    "\n✅ Même split spatial pour "
    "mono-classe et multi-classe."
)
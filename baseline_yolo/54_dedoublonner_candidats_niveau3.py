from pathlib import Path
import math

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PILOT_DIR = (
    DATA_DIR
    / "niveau3_pilote"
)

INPUT_CSV = (
    PILOT_DIR
    / "candidats"
    / "candidats.csv"
)

OUT_CSV = (
    PILOT_DIR
    / "candidats"
    / "candidats_uniques.csv"
)

EARTH_RADIUS_M = 6371008.8

# Deux détections à moins de 30 m
# sont considérées comme appartenant
# au même candidat géographique.
CLUSTER_DISTANCE_M = 30.0


# ============================================================
# DISTANCE
# ============================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2,
):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return EARTH_RADIUS_M * c


# ============================================================
# CHARGEMENT
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("====================================")
print("DEDUPLICATION CANDIDATS NIVEAU 3")
print("====================================")

print(
    "Détections candidates initiales :",
    len(df)
)

print(
    "Distance de regroupement :",
    CLUSTER_DISTANCE_M,
    "m"
)


# ============================================================
# UNION-FIND
# ============================================================

parent = list(range(len(df)))


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


for i in range(len(df)):
    for j in range(i + 1, len(df)):

        d = haversine(
            float(df.iloc[i]["latitude"]),
            float(df.iloc[i]["longitude"]),
            float(df.iloc[j]["latitude"]),
            float(df.iloc[j]["longitude"]),
        )

        if d <= CLUSTER_DISTANCE_M:
            union(i, j)


# ============================================================
# IDENTIFICATION DES GROUPES
# ============================================================

groups = {}

for i in range(len(df)):
    root = find(i)
    groups.setdefault(root, []).append(i)


rows = []

for cluster_number, indices in enumerate(
    groups.values(),
    start=1,
):

    cluster = df.iloc[indices].copy()

    # Représentant :
    # priorité à la confiance la plus élevée.
    representative_index = (
        cluster["confidence"].idxmax()
    )

    representative = df.loc[
        representative_index
    ].copy()

    representative[
        "unique_candidate_id"
    ] = f"U{cluster_number:03d}"

    representative[
        "detections_in_cluster"
    ] = len(cluster)

    representative[
        "cluster_candidate_ids"
    ] = ",".join(
        cluster["candidate_id"]
        .astype(str)
        .tolist()
    )

    representative[
        "cluster_max_confidence"
    ] = cluster[
        "confidence"
    ].max()

    representative[
        "cluster_max_distance_anfr_m"
    ] = cluster[
        "nearest_ANFR_distance_m"
    ].max()

    representative[
        "review_status"
    ] = ""

    representative[
        "review_comment"
    ] = ""

    rows.append(
        representative
    )


unique = pd.DataFrame(
    rows
)


# Meilleurs candidats en premier :
# confiance élevée, puis distance ANFR.
unique = unique.sort_values(
    [
        "confidence",
        "nearest_ANFR_distance_m",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(
    drop=True
)


unique.to_csv(
    OUT_CSV,
    index=False
)


# ============================================================
# RESULTATS
# ============================================================

print("\n====================================")
print("CANDIDATS UNIQUES")
print("====================================")

print(
    "Détections initiales :",
    len(df)
)

print(
    "Candidats géographiques uniques :",
    len(unique)
)

print(
    "Détections fusionnées :",
    len(df) - len(unique)
)


print("\n===== CLUSTERS MULTIPLES =====")

multi = unique[
    unique[
        "detections_in_cluster"
    ] > 1
]

if len(multi) == 0:
    print("Aucun")
else:
    print(
        multi[
            [
                "unique_candidate_id",
                "cluster_candidate_ids",
                "detections_in_cluster",
                "confidence",
                "nearest_ANFR_distance_m",
            ]
        ].to_string(
            index=False
        )
    )


print("\n===== CANDIDATS A INSPECTER =====")

print(
    unique[
        [
            "unique_candidate_id",
            "candidate_id",
            "confidence",
            "nearest_ANFR_distance_m",
            "latitude",
            "longitude",
            "cluster_candidate_ids",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nCSV :",
    OUT_CSV
)

print(
    "\n✅ Candidats proches regroupés."
)

print(
    "✅ Une détection représentative conservée par site."
)
from pathlib import Path
import math

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

DETECTIONS_PATH = (
    DATA_DIR
    / "niveau3_multizone"
    / "detections"
    / "detections_geolocalisees.csv"
)

OLD_REVIEW_PATH = (
    DATA_DIR
    / "niveau3_pilote"
    / "candidats"
    / "candidats_uniques.csv"
)

OUT_DIR = (
    DATA_DIR
    / "niveau3_multizone"
    / "candidats"
)

OUT_CSV = (
    OUT_DIR
    / "candidats_uniques.csv"
)

MATCHES_CSV = (
    OUT_DIR
    / "transferts_revue_dense.csv"
)


DISTANCE_MIN_ANFR_M = 150.0
CLUSTER_DISTANCE_M = 30.0

# Un peu plus large que le clustering car le représentant
# d'un même cluster peut changer légèrement entre deux JPEG.
TRANSFER_MAX_DISTANCE_M = 40.0

EARTH_RADIUS_M = 6371008.8


# ============================================================
# DISTANCE
# ============================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2,
):

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2.0) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * math.atan2(
        math.sqrt(a),
        math.sqrt(1.0 - a),
    )

    return EARTH_RADIUS_M * c


# ============================================================
# CHARGEMENT
# ============================================================

for path in [
    DETECTIONS_PATH,
    OLD_REVIEW_PATH,
]:

    if not path.exists():
        raise FileNotFoundError(path)


detections = pd.read_csv(
    DETECTIONS_PATH
)

old_review = pd.read_csv(
    OLD_REVIEW_PATH
)


far = detections[
    detections[
        "nearest_ANFR_distance_m"
    ] >= DISTANCE_MIN_ANFR_M
].copy().reset_index(drop=True)


print("====================================")
print("DEDUPLICATION MULTIZONE NIVEAU 3")
print("====================================")

print(
    "Détections totales :",
    len(detections)
)

print(
    "Détections >= 150 m :",
    len(far)
)

print("\nPar zone :")

print(
    far["zone_id"].value_counts()
)


# ============================================================
# UNION-FIND
# ============================================================

parent = list(
    range(len(far))
)


def find(x):

    while parent[x] != x:

        parent[x] = parent[
            parent[x]
        ]

        x = parent[x]

    return x


def union(a, b):

    ra = find(a)
    rb = find(b)

    if ra != rb:
        parent[rb] = ra


# On ne fusionne que dans une même zone.
for i in range(len(far)):

    for j in range(
        i + 1,
        len(far)
    ):

        if (
            far.iloc[i]["zone_id"]
            != far.iloc[j]["zone_id"]
        ):
            continue

        d = haversine(
            far.iloc[i]["latitude"],
            far.iloc[i]["longitude"],
            far.iloc[j]["latitude"],
            far.iloc[j]["longitude"],
        )

        if d <= CLUSTER_DISTANCE_M:
            union(i, j)


# ============================================================
# CLUSTERS
# ============================================================

groups = {}

for i in range(len(far)):

    root = find(i)

    groups.setdefault(
        root,
        []
    ).append(i)


records = []
member_indices = []


for indices in groups.values():

    cluster = far.iloc[
        indices
    ].copy()

    # Représentant = confiance maximale.
    representative_index = (
        cluster[
            "confidence"
        ].idxmax()
    )

    rep = far.loc[
        representative_index
    ].copy()


    record = rep.to_dict()

    record[
        "detections_in_cluster"
    ] = len(cluster)

    record[
        "cluster_detection_ids"
    ] = ",".join(
        cluster[
            "detection_id"
        ]
        .astype(str)
        .tolist()
    )

    record[
        "cluster_max_confidence"
    ] = float(
        cluster[
            "confidence"
        ].max()
    )

    record[
        "cluster_max_distance_anfr_m"
    ] = float(
        cluster[
            "nearest_ANFR_distance_m"
        ].max()
    )

    record[
        "review_status"
    ] = ""

    record[
        "review_comment"
    ] = ""

    record[
        "review_origin"
    ] = ""

    record[
        "old_unique_candidate_id"
    ] = ""

    records.append(
        record
    )

    member_indices.append(
        indices
    )


unique = pd.DataFrame(
    records
)


# ============================================================
# IDENTIFIANTS PROPRES PAR ZONE
# ============================================================

zone_prefix = {
    "dense": "D",
    "intermediaire": "I",
    "peu_dense": "P",
}


unique_ids = []


for zone_id in [
    "dense",
    "intermediaire",
    "peu_dense",
]:

    indices = unique[
        unique[
            "zone_id"
        ] == zone_id
    ].index.tolist()


    indices = sorted(
        indices,
        key=lambda idx: (
            -float(
                unique.loc[
                    idx,
                    "confidence"
                ]
            ),
            -float(
                unique.loc[
                    idx,
                    "nearest_ANFR_distance_m"
                ]
            ),
        ),
    )


    for number, idx in enumerate(
        indices,
        start=1,
    ):

        unique_ids.append(
            (
                idx,
                f"{zone_prefix[zone_id]}"
                f"{number:03d}",
            )
        )


unique[
    "unique_candidate_id"
] = ""


for idx, uid in unique_ids:

    unique.at[
        idx,
        "unique_candidate_id"
    ] = uid


# ============================================================
# TRANSFERT DES REVUES DE L'ANCIEN PILOTE DENSE
# ============================================================

old_review[
    "review_status"
] = (
    old_review[
        "review_status"
    ]
    .fillna("")
    .astype(str)
)


old_review[
    "review_comment"
] = (
    old_review[
        "review_comment"
    ]
    .fillna("")
    .astype(str)
)


old_review = old_review[
    old_review[
        "review_status"
    ]
    .str.strip()
    .ne("")
].copy()


dense_new_indices = unique[
    unique[
        "zone_id"
    ] == "dense"
].index.tolist()


# Toutes les paires possibles old/new.
pairs = []


for new_idx in dense_new_indices:

    members = far.iloc[
        member_indices[new_idx]
    ]

    for old_idx, old in old_review.iterrows():

        # On utilise la distance minimale entre
        # l'ancien candidat et toutes les détections
        # du nouveau cluster.
        distances = []

        for _, member in members.iterrows():

            distances.append(
                haversine(
                    old["latitude"],
                    old["longitude"],
                    member["latitude"],
                    member["longitude"],
                )
            )


        min_distance = min(
            distances
        )


        if (
            min_distance
            <= TRANSFER_MAX_DISTANCE_M
        ):

            pairs.append(
                (
                    min_distance,
                    new_idx,
                    old_idx,
                )
            )


# Matching glouton 1 -> 1 du plus proche au plus éloigné.
pairs.sort(
    key=lambda x: x[0]
)


used_new = set()
used_old = set()

transfers = []


for distance_m, new_idx, old_idx in pairs:

    if new_idx in used_new:
        continue

    if old_idx in used_old:
        continue


    old = old_review.loc[
        old_idx
    ]


    unique.at[
        new_idx,
        "review_status"
    ] = old[
        "review_status"
    ]

    unique.at[
        new_idx,
        "review_comment"
    ] = old[
        "review_comment"
    ]

    unique.at[
        new_idx,
        "review_origin"
    ] = "transferred_from_pilot"

    unique.at[
        new_idx,
        "old_unique_candidate_id"
    ] = old[
        "unique_candidate_id"
    ]


    transfers.append(
        {
            "new_unique_candidate_id":
                unique.at[
                    new_idx,
                    "unique_candidate_id"
                ],

            "old_unique_candidate_id":
                old[
                    "unique_candidate_id"
                ],

            "distance_match_m":
                distance_m,

            "review_status":
                old[
                    "review_status"
                ],
        }
    )


    used_new.add(
        new_idx
    )

    used_old.add(
        old_idx
    )


# ============================================================
# TRI FINAL
# ============================================================

zone_order = {
    "dense": 0,
    "intermediaire": 1,
    "peu_dense": 2,
}


unique[
    "_zone_order"
] = unique[
    "zone_id"
].map(
    zone_order
)


unique = unique.sort_values(
    [
        "_zone_order",
        "confidence",
    ],
    ascending=[
        True,
        False,
    ],
).drop(
    columns=[
        "_zone_order"
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAUVEGARDE
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


unique.to_csv(
    OUT_CSV,
    index=False
)


pd.DataFrame(
    transfers
).to_csv(
    MATCHES_CSV,
    index=False
)


# ============================================================
# AUDIT
# ============================================================

print("\n====================================")
print("CANDIDATS UNIQUES MULTIZONE")
print("====================================")

print(
    "Détections >=150 m :",
    len(far)
)

print(
    "Candidats uniques :",
    len(unique)
)

print(
    "Détections fusionnées :",
    len(far) - len(unique)
)


print("\n===== CANDIDATS UNIQUES PAR ZONE =====")

print(
    unique[
        "zone_id"
    ].value_counts()
)


transferred = (
    unique[
        "review_origin"
    ]
    == "transferred_from_pilot"
).sum()


print(
    "\nRevues dense transférées :",
    int(transferred)
)


print(
    "Dense restant à examiner :",
    int(
        (
            (unique["zone_id"] == "dense")
            &
            (
                unique["review_status"]
                .astype(str)
                .str.strip()
                .eq("")
            )
        ).sum()
    )
)


print(
    "\n===== REVUES DEJA DISPONIBLES ====="
)

print(
    unique[
        "review_status"
    ].replace(
        "",
        "non_examine"
    ).value_counts()
)


print(
    "\n===== A EXAMINER PAR ZONE ====="
)

to_review = unique[
    unique[
        "review_status"
    ]
    .astype(str)
    .str.strip()
    .eq("")
]


print(
    to_review[
        "zone_id"
    ].value_counts()
)


print(
    "\nCSV final :",
    OUT_CSV
)

print(
    "Correspondances dense :",
    MATCHES_CSV
)


print(
    "\n✅ Déduplication multizone terminée."
)

print(
    "✅ Anciennes revues dense transférées spatialement."
)

print(
    "✅ Seuls les candidats réellement nouveaux restent à examiner."
)
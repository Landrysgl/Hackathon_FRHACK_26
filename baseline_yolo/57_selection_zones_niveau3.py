from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SUPPORTS_PATH = (
    DATA_DIR
    / "supports_yvelines.csv"
)

OUT_DIR = (
    DATA_DIR
    / "niveau3_multizone"
)

OUT_PATH = (
    OUT_DIR
    / "zones_selectionnees.csv"
)


EARTH_RADIUS_M = 6378137.0

DENSITY_RADIUS_M = 500.0

# On veut des zones vraiment différentes spatialement.
MIN_DISTANCE_BETWEEN_ZONES_M = 5000.0


# ============================================================
# DISTANCE HAVERSINE
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
        math.sin(dlat / 2.0) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2.0) ** 2
    )

    c = (
        2.0
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1.0 - a),
        )
    )

    return EARTH_RADIUS_M * c


# ============================================================
# CHARGEMENT
# ============================================================

if not SUPPORTS_PATH.exists():
    raise FileNotFoundError(
        SUPPORTS_PATH
    )


supports = pd.read_csv(
    SUPPORTS_PATH
)


supports["latitude"] = pd.to_numeric(
    supports["latitude"],
    errors="coerce",
)

supports["longitude"] = pd.to_numeric(
    supports["longitude"],
    errors="coerce",
)


supports = supports.dropna(
    subset=[
        "latitude",
        "longitude",
    ]
).reset_index(drop=True)


print("====================================")
print("SELECTION ZONES NIVEAU 3")
print("====================================")

print(
    "Supports ANFR :",
    len(supports)
)


# ============================================================
# DENSITE LOCALE
# ============================================================

lat_rad = np.radians(
    supports["latitude"].to_numpy()
)

lon_rad = np.radians(
    supports["longitude"].to_numpy()
)

lat_ref = float(
    lat_rad.mean()
)


x_m = (
    EARTH_RADIUS_M
    * lon_rad
    * math.cos(lat_ref)
)

y_m = (
    EARTH_RADIUS_M
    * lat_rad
)


radius2 = (
    DENSITY_RADIUS_M
    ** 2
)


counts = np.zeros(
    len(supports),
    dtype=int,
)


for i in range(
    len(supports)
):

    dx = x_m - x_m[i]
    dy = y_m - y_m[i]

    counts[i] = int(
        np.sum(
            dx * dx
            + dy * dy
            <= radius2
        )
    )


supports["density_500m"] = counts


print("\n===== DENSITE =====")

print(
    supports["density_500m"]
    .describe()
)


# ============================================================
# ZONE DENSE
# ============================================================

dense_idx = int(
    supports[
        "density_500m"
    ].idxmax()
)

dense = supports.loc[
    dense_idx
]


# ============================================================
# FILTRE DISTANCE
# ============================================================

def far_enough_from_selected(
    row,
    selected_rows,
):

    for selected in selected_rows:

        d = haversine(
            float(row["latitude"]),
            float(row["longitude"]),
            float(selected["latitude"]),
            float(selected["longitude"]),
        )

        if d < MIN_DISTANCE_BETWEEN_ZONES_M:
            return False

    return True


# ============================================================
# ZONE INTERMEDIAIRE
# ============================================================

eligible_medium = supports[
    supports.apply(
        lambda row:
        far_enough_from_selected(
            row,
            [dense],
        ),
        axis=1,
    )
].copy()


target_median = float(
    eligible_medium[
        "density_500m"
    ].median()
)


eligible_medium[
    "distance_to_target_density"
] = (
    eligible_medium[
        "density_500m"
    ]
    - target_median
).abs()


medium_idx = (
    eligible_medium[
        "distance_to_target_density"
    ]
    .idxmin()
)

medium = supports.loc[
    medium_idx
]


# ============================================================
# ZONE PEU DENSE
# ============================================================

eligible_sparse = supports[
    supports.apply(
        lambda row:
        far_enough_from_selected(
            row,
            [
                dense,
                medium,
            ],
        ),
        axis=1,
    )
].copy()


min_density = int(
    eligible_sparse[
        "density_500m"
    ].min()
)


sparse_candidates = eligible_sparse[
    eligible_sparse[
        "density_500m"
    ] == min_density
].copy()


# Parmi les zones les moins denses,
# prendre celle la plus éloignée des deux autres.
def min_distance_to_selected(row):

    distances = []

    for selected in [
        dense,
        medium,
    ]:

        distances.append(
            haversine(
                float(row["latitude"]),
                float(row["longitude"]),
                float(selected["latitude"]),
                float(selected["longitude"]),
            )
        )

    return min(distances)


sparse_candidates[
    "distance_selected"
] = sparse_candidates.apply(
    min_distance_to_selected,
    axis=1,
)


sparse_idx = (
    sparse_candidates[
        "distance_selected"
    ]
    .idxmax()
)

sparse = supports.loc[
    sparse_idx
]


# ============================================================
# RESULTAT
# ============================================================

selected = pd.DataFrame(
    [
        {
            "zone_id": "dense",
            "SUP_ID": dense["SUP_ID"],
            "latitude": dense["latitude"],
            "longitude": dense["longitude"],
            "supports_500m": int(
                dense["density_500m"]
            ),
        },
        {
            "zone_id": "intermediaire",
            "SUP_ID": medium["SUP_ID"],
            "latitude": medium["latitude"],
            "longitude": medium["longitude"],
            "supports_500m": int(
                medium["density_500m"]
            ),
        },
        {
            "zone_id": "peu_dense",
            "SUP_ID": sparse["SUP_ID"],
            "latitude": sparse["latitude"],
            "longitude": sparse["longitude"],
            "supports_500m": int(
                sparse["density_500m"]
            ),
        },
    ]
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


selected.to_csv(
    OUT_PATH,
    index=False,
)


print("\n====================================")
print("ZONES SELECTIONNEES")
print("====================================")

print(
    selected.to_string(
        index=False
    )
)


print("\n===== DISTANCES ENTRE ZONES =====")

for i in range(len(selected)):

    for j in range(
        i + 1,
        len(selected)
    ):

        a = selected.iloc[i]
        b = selected.iloc[j]

        d = haversine(
            a["latitude"],
            a["longitude"],
            b["latitude"],
            b["longitude"],
        )

        print(
            f"{a['zone_id']} <-> "
            f"{b['zone_id']} : "
            f"{d / 1000:.2f} km"
        )


print(
    "\nCSV :",
    OUT_PATH
)

print(
    "\n✅ 3 zones de densité différente."
)

print(
    "✅ Zones spatialement séparées."
)

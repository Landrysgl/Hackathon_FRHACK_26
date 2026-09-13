from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MULTI_DIR = DATA_DIR / "niveau3_multizone"

CANDIDATES_PATH = (
    MULTI_DIR
    / "candidats"
    / "candidats_uniques.csv"
)

OUT_DIR = (
    MULTI_DIR
    / "resultats_finaux"
)

OUT_CSV = (
    OUT_DIR
    / "candidats_finaux.csv"
)

OUT_JSON = (
    OUT_DIR
    / "candidats_finaux.json"
)


FINAL_IDS = [
    "D003",
    "D006",
    "D007",
]


# ============================================================
# CHARGEMENT
# ============================================================

df = pd.read_csv(
    CANDIDATES_PATH
)


for col in [
    "review_status",
    "support_type",
]:

    df[col] = (
        df[col]
        .fillna("")
        .astype(str)
    )


# ============================================================
# VERIFICATIONS
# ============================================================

selected = df[
    df[
        "unique_candidate_id"
    ].isin(
        FINAL_IDS
    )
].copy()


if len(selected) != 3:

    found = selected[
        "unique_candidate_id"
    ].tolist()

    raise RuntimeError(
        f"3 candidats attendus. "
        f"Trouvés : {found}"
    )


for uid in FINAL_IDS:

    row = selected[
        selected[
            "unique_candidate_id"
        ] == uid
    ].iloc[0]


    if (
        row[
            "review_status"
        ] != "plausible"
    ):

        raise RuntimeError(
            f"{uid} n'est pas plausible."
        )


    if (
        not row[
            "support_type"
        ].strip()
        or
        row[
            "support_type"
        ] == "indetermine"
    ):

        raise RuntimeError(
            f"{uid} n'a pas de "
            "type de support valide."
        )


# ============================================================
# MARQUAGE DANS LE CSV PRINCIPAL
# ============================================================

df[
    "selection_finale"
] = False

df[
    "rang_final"
] = pd.NA


for rank, uid in enumerate(
    FINAL_IDS,
    start=1,
):

    mask = (
        df[
            "unique_candidate_id"
        ] == uid
    )

    df.loc[
        mask,
        "selection_finale"
    ] = True

    df.loc[
        mask,
        "rang_final"
    ] = rank


df.to_csv(
    CANDIDATES_PATH,
    index=False,
)


# ============================================================
# EXPORT DEDIE
# ============================================================

selected = df[
    df[
        "selection_finale"
    ] == True
].copy()


selected = selected.sort_values(
    "rang_final"
)


selected.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# JSON LEGER POUR RAPPORT / PRESENTATION
# ============================================================

result = []


for _, row in selected.iterrows():

    result.append(
        {
            "rang":
                int(
                    row[
                        "rang_final"
                    ]
                ),

            "id":
                str(
                    row[
                        "unique_candidate_id"
                    ]
                ),

            "zone":
                str(
                    row[
                        "zone_id"
                    ]
                ),

            "support_type":
                str(
                    row[
                        "support_type"
                    ]
                ),

            "confidence":
                float(
                    row[
                        "confidence"
                    ]
                ),

            "distance_anfr_m":
                float(
                    row[
                        "nearest_ANFR_distance_m"
                    ]
                ),

            "latitude":
                float(
                    row[
                        "latitude"
                    ]
                ),

            "longitude":
                float(
                    row[
                        "longitude"
                    ]
                ),
        }
    )


OUT_JSON.write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# SORTIE
# ============================================================

print("====================================")
print("3 CANDIDATS FINAUX NIVEAU 3")
print("====================================")

print(
    selected[
        [
            "rang_final",
            "unique_candidate_id",
            "zone_id",
            "support_type",
            "confidence",
            "nearest_ANFR_distance_m",
            "latitude",
            "longitude",
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
    "JSON :",
    OUT_JSON
)

print(
    "\n✅ D003, D006 et D007 "
    "sont désormais la sélection finale."
)
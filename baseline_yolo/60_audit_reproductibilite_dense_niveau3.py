from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PILOT_DIR = (
    DATA_DIR
    / "niveau3_pilote"
)

MULTI_DIR = (
    DATA_DIR
    / "niveau3_multizone"
    / "zones"
    / "dense"
)

PILOT_IMAGES = (
    PILOT_DIR
    / "images"
)

DENSE_IMAGES = (
    MULTI_DIR
    / "images"
)

PILOT_META = (
    PILOT_DIR
    / "metadata_patches.csv"
)

DENSE_META = (
    MULTI_DIR
    / "metadata_patches.csv"
)


# ============================================================
# VERIFICATIONS
# ============================================================

for path in [
    PILOT_IMAGES,
    DENSE_IMAGES,
    PILOT_META,
    DENSE_META,
]:
    if not path.exists():
        raise FileNotFoundError(path)


print("====================================")
print("AUDIT REPRODUCTIBILITE ZONE DENSE")
print("====================================")


# ============================================================
# METADONNEES
# ============================================================

pilot_meta = pd.read_csv(
    PILOT_META
)

dense_meta = pd.read_csv(
    DENSE_META
)


merged = pilot_meta.merge(
    dense_meta,
    on=[
        "row",
        "col",
    ],
    suffixes=(
        "_pilot",
        "_dense",
    ),
)


print(
    "Patches comparés :",
    len(merged)
)


if len(merged) != 25:
    raise RuntimeError(
        "25 patches attendus."
    )


print("\n===== GEOREFERENCEMENT =====")


geo_columns = [
    "world_left",
    "world_top",
    "world_right",
    "world_bottom",
    "north",
    "south",
    "west",
    "east",
]


for col in geo_columns:

    a = pd.to_numeric(
        merged[
            f"{col}_pilot"
        ]
    )

    b = pd.to_numeric(
        merged[
            f"{col}_dense"
        ]
    )

    diff = (
        a - b
    ).abs()

    print(
        f"{col:15s} "
        f"max_diff = "
        f"{diff.max():.12f}"
    )


# ============================================================
# COMPARAISON PIXEL PAR PIXEL
# ============================================================

records = []


for row in range(5):

    for col in range(5):

        pilot_name = (
            f"pilot_r{row:02d}"
            f"_c{col:02d}.jpg"
        )

        dense_name = (
            f"dense_r{row:02d}"
            f"_c{col:02d}.jpg"
        )


        pilot_path = (
            PILOT_IMAGES
            / pilot_name
        )

        dense_path = (
            DENSE_IMAGES
            / dense_name
        )


        if not pilot_path.exists():
            raise FileNotFoundError(
                pilot_path
            )

        if not dense_path.exists():
            raise FileNotFoundError(
                dense_path
            )


        with Image.open(
            pilot_path
        ) as im:

            pilot = np.asarray(
                im.convert("RGB"),
                dtype=np.int16,
            )


        with Image.open(
            dense_path
        ) as im:

            dense = np.asarray(
                im.convert("RGB"),
                dtype=np.int16,
            )


        if pilot.shape != dense.shape:

            raise RuntimeError(
                f"Taille différente "
                f"{pilot_name}: "
                f"{pilot.shape} vs "
                f"{dense.shape}"
            )


        diff = np.abs(
            pilot - dense
        )


        exact = bool(
            np.array_equal(
                pilot,
                dense
            )
        )


        changed_values = int(
            np.count_nonzero(
                diff
            )
        )


        total_values = int(
            diff.size
        )


        pct_changed = (
            100.0
            * changed_values
            / total_values
        )


        records.append(
            {
                "row":
                    row,

                "col":
                    col,

                "pilot":
                    pilot_name,

                "dense":
                    dense_name,

                "exactement_identiques":
                    exact,

                "mean_abs_diff":
                    float(
                        diff.mean()
                    ),

                "max_abs_diff":
                    int(
                        diff.max()
                    ),

                "pct_valeurs_differentes":
                    pct_changed,
            }
        )


results = pd.DataFrame(
    records
)


# ============================================================
# RESULTATS
# ============================================================

print("\n===== IMAGES =====")

print(
    "Exactement identiques :",
    int(
        results[
            "exactement_identiques"
        ].sum()
    ),
    "/",
    len(results)
)


print(
    "Différence absolue moyenne globale :",
    results[
        "mean_abs_diff"
    ].mean()
)


print(
    "Différence maximale observée :",
    results[
        "max_abs_diff"
    ].max()
)


print(
    "Pourcentage moyen de valeurs RGB différentes :",
    results[
        "pct_valeurs_differentes"
    ].mean()
)


different = results[
    ~results[
        "exactement_identiques"
    ]
]


print(
    "\nPatches différents :",
    len(different)
)


if len(different):

    print(
        "\n===== DETAIL PATCHES DIFFERENTS ====="
    )

    print(
        different.to_string(
            index=False
        )
    )


print("\n====================================")
print("CONCLUSION TECHNIQUE")
print("====================================")


if results[
    "exactement_identiques"
].all():

    print(
        "✅ Les 25 images sont "
        "strictement identiques."
    )

    print(
        "La différence de détection "
        "ne vient pas des images."
    )

else:

    print(
        "⚠️ Les deux générations "
        "ne sont pas pixel-identiques."
    )

    print(
        "Les petites différences JPEG "
        "peuvent modifier les prédictions "
        "proches du seuil de confiance."
    )
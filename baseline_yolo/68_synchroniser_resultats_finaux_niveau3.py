from pathlib import Path
import json

import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MULTI_DIR = (
    DATA_DIR
    / "niveau3_multizone"
)

CANDIDATES_PATH = (
    MULTI_DIR
    / "candidats"
    / "candidats_uniques.csv"
)

DETECTION_SUMMARY_PATH = (
    MULTI_DIR
    / "detections"
    / "resume_par_zone.csv"
)

ZONES_SUMMARY_PATH = (
    MULTI_DIR
    / "resume_zones.csv"
)

ZONES_DIR = (
    MULTI_DIR
    / "zones"
)

OUT_DIR = (
    MULTI_DIR
    / "resultats_finaux"
)

SUMMARY_CSV = (
    OUT_DIR
    / "resume_par_zone_final.csv"
)

PLAUSIBLE_CSV = (
    OUT_DIR
    / "candidats_plausibles.csv"
)

PLAUSIBLE_GEOJSON = (
    OUT_DIR
    / "candidats_plausibles.geojson"
)

FLAGSHIP_CSV = (
    OUT_DIR
    / "candidat_phare.csv"
)

FLAGSHIP_IMAGE = (
    OUT_DIR
    / "candidat_phare_D003.jpg"
)

FINAL_JSON = (
    OUT_DIR
    / "resume_niveau3_final.json"
)

FINAL_MD = (
    OUT_DIR
    / "NIVEAU3_RESULTATS_FINAL.md"
)


ZONE_ORDER = [
    "dense",
    "intermediaire",
    "peu_dense",
]


# ============================================================
# CHARGEMENT
# ============================================================

for path in [
    CANDIDATES_PATH,
    DETECTION_SUMMARY_PATH,
    ZONES_SUMMARY_PATH,
]:

    if not path.exists():
        raise FileNotFoundError(path)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


candidates = pd.read_csv(
    CANDIDATES_PATH
)

detections = pd.read_csv(
    DETECTION_SUMMARY_PATH
)

zones = pd.read_csv(
    ZONES_SUMMARY_PATH
)


for col in [
    "review_status",
    "support_type",
    "review_comment",
]:

    if col not in candidates.columns:
        candidates[col] = ""

    candidates[col] = (
        candidates[col]
        .fillna("")
        .astype(str)
    )


# ============================================================
# CONTROLES
# ============================================================

remaining = int(
    candidates[
        "review_status"
    ]
    .str.strip()
    .eq("")
    .sum()
)


if remaining != 0:

    raise RuntimeError(
        f"{remaining} candidat(s) non examiné(s)."
    )


plausibles = candidates[
    candidates[
        "review_status"
    ] == "plausible"
].copy()


unclassified = int(
    plausibles[
        "support_type"
    ]
    .str.strip()
    .eq("")
    .sum()
)


if unclassified != 0:

    raise RuntimeError(
        f"{unclassified} plausible(s) "
        "sans type visuel."
    )


# ============================================================
# RESUME PAR ZONE
# ============================================================

rows = []


for zone_id in ZONE_ORDER:

    det = detections[
        detections[
            "zone_id"
        ] == zone_id
    ].iloc[0]

    zone = zones[
        zones[
            "zone_id"
        ] == zone_id
    ].iloc[0]

    c = candidates[
        candidates[
            "zone_id"
        ] == zone_id
    ]


    n_plausible = int(
        (
            c[
                "review_status"
            ] == "plausible"
        ).sum()
    )

    n_false = int(
        (
            c[
                "review_status"
            ] == "faux_positif"
        ).sum()
    )

    n_uncertain = int(
        (
            c[
                "review_status"
            ] == "incertain"
        ).sum()
    )


    rows.append(
        {
            "zone_id":
                zone_id,

            "patches":
                int(
                    det["patches"]
                ),

            "supports_500m":
                int(
                    zone[
                        "supports_500m"
                    ]
                ),

            "supports_dans_emprise":
                int(
                    zone[
                        "supports_dans_emprise"
                    ]
                ),

            "detections_brutes":
                int(
                    det[
                        "detections"
                    ]
                ),

            "detections_ge150m":
                int(
                    det[
                        "gt_150m"
                    ]
                ),

            "candidats_uniques":
                len(c),

            "plausibles":
                n_plausible,

            "faux_positifs":
                n_false,

            "incertains":
                n_uncertain,

            "taux_plausibles_pct":
                (
                    100.0
                    * n_plausible
                    / len(c)
                ),
        }
    )


summary = pd.DataFrame(
    rows
)


summary.to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================
# EXPORT PLAUSIBLES
# ============================================================

plausibles.to_csv(
    PLAUSIBLE_CSV,
    index=False,
)


features = []


for _, row in plausibles.iterrows():

    features.append(
        {
            "type":
                "Feature",

            "geometry": {
                "type":
                    "Point",

                "coordinates": [
                    float(
                        row[
                            "longitude"
                        ]
                    ),
                    float(
                        row[
                            "latitude"
                        ]
                    ),
                ],
            },

            "properties": {
                "candidate_id":
                    str(
                        row[
                            "unique_candidate_id"
                        ]
                    ),

                "zone_id":
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

                "status":
                    "plausible",
            },
        }
    )


geojson = {
    "type":
        "FeatureCollection",

    "features":
        features,
}


PLAUSIBLE_GEOJSON.write_text(
    json.dumps(
        geojson,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# CANDIDAT PHARE : D003
# ============================================================

flagship = candidates[
    candidates[
        "unique_candidate_id"
    ] == "D003"
].copy()


if len(flagship) != 1:

    raise RuntimeError(
        "D003 introuvable ou dupliqué."
    )


flagship = flagship.iloc[0]


if flagship[
    "review_status"
] != "plausible":

    raise RuntimeError(
        "D003 n'est plus plausible."
    )


if flagship[
    "support_type"
] != "mat":

    raise RuntimeError(
        "D003 n'est plus classé mat."
    )


pd.DataFrame(
    [flagship]
).to_csv(
    FLAGSHIP_CSV,
    index=False,
)


# ============================================================
# IMAGE D003
# ============================================================

source_path = (
    ZONES_DIR
    / str(
        flagship[
            "zone_id"
        ]
    )
    / "images"
    / str(
        flagship[
            "image"
        ]
    )
)


if not source_path.exists():
    raise FileNotFoundError(
        source_path
    )


with Image.open(
    source_path
) as im:

    image = im.convert(
        "RGB"
    )


x1 = int(
    round(
        float(
            flagship["x1"]
        )
    )
)

y1 = int(
    round(
        float(
            flagship["y1"]
        )
    )
)

x2 = int(
    round(
        float(
            flagship["x2"]
        )
    )
)

y2 = int(
    round(
        float(
            flagship["y2"]
        )
    )
)


margin = 220

cx1 = max(
    0,
    x1 - margin
)

cy1 = max(
    0,
    y1 - margin
)

cx2 = min(
    image.width,
    x2 + margin
)

cy2 = min(
    image.height,
    y2 + margin
)


context = image.crop(
    (
        cx1,
        cy1,
        cx2,
        cy2,
    )
)


draw = ImageDraw.Draw(
    context
)


draw.rectangle(
    (
        x1 - cx1,
        y1 - cy1,
        x2 - cx1,
        y2 - cy1,
    ),
    outline="red",
    width=4,
)


label = (
    f"D003 | mat | "
    f"conf={float(flagship['confidence']):.3f} | "
    f"ANFR={float(flagship['nearest_ANFR_distance_m']):.1f} m"
)


draw.rectangle(
    (
        0,
        0,
        context.width,
        32,
    ),
    fill="black",
)


draw.text(
    (
        7,
        8,
    ),
    label,
    fill="white",
)


context.save(
    FLAGSHIP_IMAGE,
    quality=95,
)


# ============================================================
# STATS GLOBALES
# ============================================================

total_patches = int(
    summary[
        "patches"
    ].sum()
)

total_detections = int(
    summary[
        "detections_brutes"
    ].sum()
)

total_far = int(
    summary[
        "detections_ge150m"
    ].sum()
)

total_unique = int(
    summary[
        "candidats_uniques"
    ].sum()
)

total_plausible = int(
    summary[
        "plausibles"
    ].sum()
)

total_false = int(
    summary[
        "faux_positifs"
    ].sum()
)

total_uncertain = int(
    summary[
        "incertains"
    ].sum()
)


support_counts = (
    plausibles[
        "support_type"
    ]
    .value_counts()
    .to_dict()
)


result = {
    "zones":
        3,

    "patches":
        total_patches,

    "detections_brutes":
        total_detections,

    "detections_ge150m":
        total_far,

    "candidats_uniques":
        total_unique,

    "plausibles":
        total_plausible,

    "faux_positifs":
        total_false,

    "incertains":
        total_uncertain,

    "taux_plausibles_pct":
        (
            100.0
            * total_plausible
            / total_unique
        ),

    "types_plausibles":
        support_counts,

    "candidat_phare": {
        "candidate_id":
            "D003",

        "support_type":
            "mat",

        "confidence":
            float(
                flagship[
                    "confidence"
                ]
            ),

        "distance_anfr_m":
            float(
                flagship[
                    "nearest_ANFR_distance_m"
                ]
            ),

        "latitude":
            float(
                flagship[
                    "latitude"
                ]
            ),

        "longitude":
            float(
                flagship[
                    "longitude"
                ]
            ),
    },
}


FINAL_JSON.write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# MARKDOWN FINAL
# ============================================================

lines = [
    "# Niveau 3 — Résultats finaux",
    "",
    "## Pipeline multizone",
    "",
    f"- 3 zones de densité ANFR différente",
    f"- {total_patches} patches analysés",
    f"- {total_detections} détections brutes",
    f"- {total_far} détections à au moins 150 m du support ANFR le plus proche",
    f"- {total_unique} candidats uniques après déduplication",
    f"- {total_plausible} candidats visuellement plausibles",
    f"- {total_false} faux positifs",
    f"- {total_uncertain} candidats incertains",
    "",
    "Un candidat visuellement plausible n'est pas une preuve "
    "de site radioélectrique non déclaré.",
    "",
    "## Résultats par zone",
    "",
    "| Zone | Patches | Détections | >=150 m | Uniques | Plausibles | Faux positifs | Incertains |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]


for _, row in summary.iterrows():

    lines.append(
        f"| {row['zone_id']} "
        f"| {int(row['patches'])} "
        f"| {int(row['detections_brutes'])} "
        f"| {int(row['detections_ge150m'])} "
        f"| {int(row['candidats_uniques'])} "
        f"| {int(row['plausibles'])} "
        f"| {int(row['faux_positifs'])} "
        f"| {int(row['incertains'])} |"
    )


lines += [
    "",
    "## Nature visuelle des candidats plausibles",
    "",
]


for key, value in support_counts.items():

    lines.append(
        f"- {key} : {value}"
    )


lines += [
    "",
    "## Candidat phare",
    "",
    f"- ID : D003",
    f"- type visuel : mât",
    f"- confiance : {float(flagship['confidence']):.3f}",
    f"- distance au support ANFR le plus proche : "
    f"{float(flagship['nearest_ANFR_distance_m']):.1f} m",
    f"- coordonnées : "
    f"{float(flagship['latitude']):.6f}, "
    f"{float(flagship['longitude']):.6f}",
    "",
    "D003 est présenté comme un candidat particulièrement intéressant "
    "car la structure visible ressemble à un mât et aucune correspondance "
    "ANFR proche n'est observée selon le seuil choisi.",
    "",
    "Cette observation ne permet toutefois pas de conclure qu'il s'agit "
    "d'un site radioélectrique non déclaré : il peut s'agir d'un support "
    "sans installation radio, d'un décalage temporel ou géographique, "
    "ou d'une erreur de classification.",
]


FINAL_MD.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# ============================================================
# SORTIE
# ============================================================

print("====================================")
print("SYNCHRONISATION FINALE NIVEAU 3")
print("====================================")

print(
    "\nRésumé :"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nTypes plausibles :"
)

print(
    plausibles[
        "support_type"
    ].value_counts()
)

print(
    "\nCandidat phare : D003"
)

print(
    "Confiance :",
    float(
        flagship[
            "confidence"
        ]
    )
)

print(
    "Distance ANFR :",
    float(
        flagship[
            "nearest_ANFR_distance_m"
        ]
    )
)

print(
    "\nMarkdown :",
    FINAL_MD
)

print(
    "GeoJSON :",
    PLAUSIBLE_GEOJSON
)

print(
    "Image D003 :",
    FLAGSHIP_IMAGE
)

print(
    "\n✅ Livrables Niveau 3 synchronisés."
)

print(
    "✅ Classification visuelle incluse."
)

print(
    "✅ D003 devient le candidat phare."
)

print(
    "⚠️ Les anciens fichiers top3_* sont désormais obsolètes "
    "et seront nettoyés plus tard."
)
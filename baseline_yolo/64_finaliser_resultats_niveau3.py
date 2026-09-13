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

TOP3_DIR = (
    OUT_DIR
    / "top3"
)

FINAL_SUMMARY_CSV = (
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

TOP3_CSV = (
    OUT_DIR
    / "top3_candidats.csv"
)

TOP3_MONTAGE = (
    OUT_DIR
    / "top3_montage.jpg"
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

CONTEXT_MARGIN = 180


# ============================================================
# VERIFICATIONS
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

TOP3_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHARGEMENT
# ============================================================

candidates = pd.read_csv(
    CANDIDATES_PATH
)

detections_summary = pd.read_csv(
    DETECTION_SUMMARY_PATH
)

zones_summary = pd.read_csv(
    ZONES_SUMMARY_PATH
)


candidates["review_status"] = (
    candidates["review_status"]
    .fillna("")
    .astype(str)
)


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
        f"{remaining} candidat(s) "
        "encore non examiné(s)."
    )


print("====================================")
print("FINALISATION NIVEAU 3 MULTIZONE")
print("====================================")

print(
    "Candidats uniques :",
    len(candidates)
)

print(
    "\nStatuts :"
)

print(
    candidates[
        "review_status"
    ].value_counts()
)


# ============================================================
# RESUME PAR ZONE
# ============================================================

rows = []


for zone_id in ZONE_ORDER:

    det = detections_summary[
        detections_summary[
            "zone_id"
        ] == zone_id
    ]

    zone_info = zones_summary[
        zones_summary[
            "zone_id"
        ] == zone_id
    ]

    cand = candidates[
        candidates[
            "zone_id"
        ] == zone_id
    ]


    if len(det) != 1:
        raise RuntimeError(
            f"Résumé détection absent "
            f"ou ambigu pour {zone_id}."
        )

    if len(zone_info) != 1:
        raise RuntimeError(
            f"Résumé zone absent "
            f"ou ambigu pour {zone_id}."
        )


    det = det.iloc[0]
    zone_info = zone_info.iloc[0]


    plausible = int(
        (
            cand["review_status"]
            == "plausible"
        ).sum()
    )

    false_positive = int(
        (
            cand["review_status"]
            == "faux_positif"
        ).sum()
    )

    uncertain = int(
        (
            cand["review_status"]
            == "incertain"
        ).sum()
    )

    unique_count = len(cand)


    plausible_rate = (
        100.0
        * plausible
        / unique_count
        if unique_count
        else 0.0
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
                    zone_info[
                        "supports_500m"
                    ]
                ),

            "supports_dans_emprise":
                int(
                    zone_info[
                        "supports_dans_emprise"
                    ]
                ),

            "detections_brutes":
                int(
                    det["detections"]
                ),

            "detections_ge150m":
                int(
                    det["gt_150m"]
                ),

            "candidats_uniques":
                unique_count,

            "plausibles":
                plausible,

            "faux_positifs":
                false_positive,

            "incertains":
                uncertain,

            "taux_plausibles_pct":
                plausible_rate,
        }
    )


summary = pd.DataFrame(
    rows
)


summary.to_csv(
    FINAL_SUMMARY_CSV,
    index=False,
)


# ============================================================
# PLAUSIBLES
# ============================================================

plausibles = candidates[
    candidates[
        "review_status"
    ] == "plausible"
].copy()


plausibles = plausibles.sort_values(
    [
        "zone_id",
        "confidence",
        "nearest_ANFR_distance_m",
    ],
    ascending=[
        True,
        False,
        False,
    ],
)


plausibles.to_csv(
    PLAUSIBLE_CSV,
    index=False,
)


# ============================================================
# GEOJSON
# ============================================================

features = []


for _, row in plausibles.iterrows():

    properties = {
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
    }


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

            "properties":
                properties,
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
# SELECTION TOP 3
# ============================================================

selected_rows = []


for zone_id in ZONE_ORDER:

    zone_plausible = plausibles[
        plausibles[
            "zone_id"
        ] == zone_id
    ].copy()


    if len(zone_plausible) == 0:

        print(
            f"⚠️ Aucun plausible "
            f"dans {zone_id}."
        )

        continue


    best = (
        zone_plausible
        .sort_values(
            [
                "confidence",
                "nearest_ANFR_distance_m",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]
        .copy()
    )


    selected_rows.append(
        best
    )


top3 = pd.DataFrame(
    selected_rows
)


top3[
    "selection_reason"
] = (
    "Meilleure confiance parmi les "
    "candidats visuellement plausibles "
    "de sa zone."
)


top3.to_csv(
    TOP3_CSV,
    index=False,
)


# ============================================================
# VISUELS TOP 3
# ============================================================

context_images = []


for _, row in top3.iterrows():

    uid = str(
        row[
            "unique_candidate_id"
        ]
    )

    zone_id = str(
        row[
            "zone_id"
        ]
    )

    image_name = str(
        row[
            "image"
        ]
    )


    source_path = (
        ZONES_DIR
        / zone_id
        / "images"
        / image_name
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
                row["x1"]
            )
        )
    )

    y1 = int(
        round(
            float(
                row["y1"]
            )
        )
    )

    x2 = int(
        round(
            float(
                row["x2"]
            )
        )
    )

    y2 = int(
        round(
            float(
                row["y2"]
            )
        )
    )


    ctx_x1 = max(
        0,
        x1 - CONTEXT_MARGIN
    )

    ctx_y1 = max(
        0,
        y1 - CONTEXT_MARGIN
    )

    ctx_x2 = min(
        image.width,
        x2 + CONTEXT_MARGIN
    )

    ctx_y2 = min(
        image.height,
        y2 + CONTEXT_MARGIN
    )


    context = image.crop(
        (
            ctx_x1,
            ctx_y1,
            ctx_x2,
            ctx_y2,
        )
    )


    draw = ImageDraw.Draw(
        context
    )


    draw.rectangle(
        (
            x1 - ctx_x1,
            y1 - ctx_y1,
            x2 - ctx_x1,
            y2 - ctx_y1,
        ),
        outline="red",
        width=5,
    )


    label = (
        f"{uid} | {zone_id} | "
        f"conf={float(row['confidence']):.3f} | "
        f"ANFR={float(row['nearest_ANFR_distance_m']):.1f} m"
    )


    draw.rectangle(
        (
            0,
            0,
            context.width,
            34,
        ),
        fill="black",
    )

    draw.text(
        (
            8,
            9,
        ),
        label,
        fill="white",
    )


    out_path = (
        TOP3_DIR
        / f"{uid}_contexte.jpg"
    )


    context.save(
        out_path,
        quality=95,
    )


    context_images.append(
        (
            uid,
            zone_id,
            context.copy(),
        )
    )


# ============================================================
# MONTAGE
# ============================================================

if context_images:

    target_height = 500

    resized = []


    for uid, zone_id, image in context_images:

        ratio = (
            target_height
            / image.height
        )

        target_width = int(
            image.width
            * ratio
        )


        resized_image = image.resize(
            (
                target_width,
                target_height,
            )
        )


        resized.append(
            (
                uid,
                zone_id,
                resized_image,
            )
        )


    separator = 20

    montage_width = (
        sum(
            image.width
            for _, _, image
            in resized
        )
        +
        separator
        * (
            len(resized) - 1
        )
    )


    montage = Image.new(
        "RGB",
        (
            montage_width,
            target_height,
        ),
        "white",
    )


    x = 0


    for uid, zone_id, image in resized:

        montage.paste(
            image,
            (
                x,
                0,
            )
        )

        x += (
            image.width
            + separator
        )


    montage.save(
        TOP3_MONTAGE,
        quality=95,
    )


# ============================================================
# RESUME GLOBAL
# ============================================================

total_patches = int(
    summary[
        "patches"
    ].sum()
)

total_raw = int(
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

global_plausible_rate = (
    100.0
    * total_plausible
    / total_unique
)


final_json = {
    "zones":
        len(summary),

    "patches":
        total_patches,

    "detections_brutes":
        total_raw,

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
        global_plausible_rate,

    "top3":
        [
            {
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
            for _, row
            in top3.iterrows()
        ],
    }


FINAL_JSON.write_text(
    json.dumps(
        final_json,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# MARKDOWN
# ============================================================

lines = []

lines.append(
    "# Niveau 3 — Résultats multizone finaux"
)

lines.append("")

lines.append(
    "Le pipeline a été appliqué à trois "
    "zones de densité ANFR différente."
)

lines.append("")

lines.append(
    f"- {total_patches} patches analysés"
)

lines.append(
    f"- {total_raw} détections brutes"
)

lines.append(
    f"- {total_far} détections à au moins 150 m "
    "du support ANFR le plus proche"
)

lines.append(
    f"- {total_unique} candidats géographiques "
    "uniques après déduplication à 30 m"
)

lines.append(
    f"- {total_plausible} candidats "
    "visuellement plausibles"
)

lines.append(
    f"- {total_false} faux positifs"
)

lines.append(
    f"- {total_uncertain} candidats incertains"
)

lines.append(
    f"- taux global de plausibilité visuelle : "
    f"{global_plausible_rate:.1f} %"
)

lines.append("")

lines.append(
    "Important : un candidat visuellement "
    "plausible et éloigné du référentiel ANFR "
    "ne constitue pas la preuve d'un site "
    "radioélectrique non déclaré."
)

lines.append("")

lines.append(
    "## Résultats par zone"
)

lines.append("")

lines.append(
    "| Zone | Patches | Détections | >=150 m | "
    "Uniques | Plausibles | Faux positifs | "
    "Incertains | Plausibles (%) |"
)

lines.append(
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
)


for _, row in summary.iterrows():

    lines.append(
        f"| {row['zone_id']} "
        f"| {int(row['patches'])} "
        f"| {int(row['detections_brutes'])} "
        f"| {int(row['detections_ge150m'])} "
        f"| {int(row['candidats_uniques'])} "
        f"| {int(row['plausibles'])} "
        f"| {int(row['faux_positifs'])} "
        f"| {int(row['incertains'])} "
        f"| {row['taux_plausibles_pct']:.1f} |"
    )


lines.append("")

lines.append(
    "## Trois candidats représentatifs"
)

lines.append("")

lines.append(
    "Un candidat plausible est retenu dans "
    "chaque zone afin de montrer le comportement "
    "du pipeline dans des contextes géographiques "
    "différents."
)

lines.append("")

lines.append(
    "| ID | Zone | Confiance | Distance ANFR (m) | "
    "Latitude | Longitude |"
)

lines.append(
    "|---|---|---:|---:|---:|---:|"
)


for _, row in top3.iterrows():

    lines.append(
        f"| {row['unique_candidate_id']} "
        f"| {row['zone_id']} "
        f"| {float(row['confidence']):.3f} "
        f"| {float(row['nearest_ANFR_distance_m']):.1f} "
        f"| {float(row['latitude']):.6f} "
        f"| {float(row['longitude']):.6f} |"
    )


FINAL_MD.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# ============================================================
# SORTIE
# ============================================================

print("\n====================================")
print("RESULTATS FINAUX NIVEAU 3")
print("====================================")

print(
    summary.to_string(
        index=False
    )
)

print("\n===== TOTAL =====")

print(
    "Patches :",
    total_patches
)

print(
    "Détections brutes :",
    total_raw
)

print(
    "Détections >=150 m :",
    total_far
)

print(
    "Candidats uniques :",
    total_unique
)

print(
    "Plausibles :",
    total_plausible
)

print(
    "Faux positifs :",
    total_false
)

print(
    "Incertains :",
    total_uncertain
)

print(
    "Taux plausible global :",
    f"{global_plausible_rate:.1f}%"
)


print("\n===== TOP 3 PROVISOIRE =====")

print(
    top3[
        [
            "unique_candidate_id",
            "zone_id",
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
    "\nMontage :",
    TOP3_MONTAGE
)

print(
    "Résumé Markdown :",
    FINAL_MD
)

print(
    "GeoJSON :",
    PLAUSIBLE_GEOJSON
)

print(
    "\n✅ Résultats multizone consolidés."
)

print(
    "✅ Un candidat représentatif "
    "sélectionné par zone."
)

print(
    "⚠️ Sélection encore à valider "
    "visuellement avant le rendu final."
)
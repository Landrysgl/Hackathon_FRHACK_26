from pathlib import Path
import json

import pandas as pd
import folium


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PILOT_DIR = (
    DATA_DIR
    / "niveau3_pilote"
)

CANDIDATES_PATH = (
    PILOT_DIR
    / "candidats"
    / "candidats_uniques.csv"
)

ANFR_ZONE_PATH = (
    PILOT_DIR
    / "anfr_dans_zone.csv"
)

OUT_DIR = (
    PILOT_DIR
    / "resultats_finaux"
)

MAP_PATH = (
    OUT_DIR
    / "carte_candidats_niveau3.html"
)

PLAUSIBLE_CSV = (
    OUT_DIR
    / "candidats_plausibles.csv"
)

GEOJSON_PATH = (
    OUT_DIR
    / "candidats_plausibles.geojson"
)

SUMMARY_PATH = (
    OUT_DIR
    / "resume_niveau3.json"
)

REPORT_PATH = (
    OUT_DIR
    / "NIVEAU3_RESULTATS.md"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHARGEMENT
# ============================================================

candidates = pd.read_csv(
    CANDIDATES_PATH
)

anfr = pd.read_csv(
    ANFR_ZONE_PATH
)


candidates["review_status"] = (
    candidates["review_status"]
    .fillna("")
    .astype(str)
)

candidates["review_comment"] = (
    candidates["review_comment"]
    .fillna("")
    .astype(str)
)


print("====================================")
print("SYNTHESE FINALE NIVEAU 3")
print("====================================")


# ============================================================
# STATUTS
# ============================================================

counts = (
    candidates["review_status"]
    .value_counts()
)


print("\n===== REVUE HUMAINE =====")
print(counts)


unreviewed = (
    candidates["review_status"]
    .str.strip()
    .eq("")
    .sum()
)


if unreviewed != 0:
    raise RuntimeError(
        f"{unreviewed} candidat(s) non examiné(s)."
    )


plausible = candidates[
    candidates["review_status"]
    == "plausible"
].copy()


false_positive = candidates[
    candidates["review_status"]
    == "faux_positif"
].copy()


uncertain = candidates[
    candidates["review_status"]
    == "incertain"
].copy()


plausible_rate = (
    len(plausible)
    / len(candidates)
    if len(candidates)
    else 0.0
)


print(
    "\nCandidats uniques :",
    len(candidates)
)

print(
    "Plausibles :",
    len(plausible)
)

print(
    "Faux positifs :",
    len(false_positive)
)

print(
    "Incertains :",
    len(uncertain)
)

print(
    "Taux plausible :",
    f"{100 * plausible_rate:.1f} %"
)


# ============================================================
# CSV FINAL
# ============================================================

plausible = plausible.sort_values(
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


plausible.to_csv(
    PLAUSIBLE_CSV,
    index=False
)


# ============================================================
# GEOJSON
# ============================================================

features = []

for _, row in plausible.iterrows():

    properties = {
        "unique_candidate_id":
            str(row["unique_candidate_id"]),

        "candidate_id":
            str(row["candidate_id"]),

        "confidence":
            float(row["confidence"]),

        "distance_anfr_m":
            float(row["nearest_ANFR_distance_m"]),

        "review_status":
            str(row["review_status"]),
    }

    features.append(
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    float(row["longitude"]),
                    float(row["latitude"]),
                ],
            },
            "properties": properties,
        }
    )


geojson = {
    "type": "FeatureCollection",
    "features": features,
}


GEOJSON_PATH.write_text(
    json.dumps(
        geojson,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# CARTE
# ============================================================

all_lat = pd.concat(
    [
        pd.to_numeric(
            candidates["latitude"],
            errors="coerce",
        ),
        pd.to_numeric(
            anfr["latitude"],
            errors="coerce",
        ),
    ]
).dropna()


all_lon = pd.concat(
    [
        pd.to_numeric(
            candidates["longitude"],
            errors="coerce",
        ),
        pd.to_numeric(
            anfr["longitude"],
            errors="coerce",
        ),
    ]
).dropna()


center_lat = float(
    all_lat.mean()
)

center_lon = float(
    all_lon.mean()
)


m = folium.Map(
    location=[
        center_lat,
        center_lon,
    ],
    zoom_start=16,
    control_scale=True,
)


# ------------------------------------------------------------
# SUPPORTS ANFR
# ------------------------------------------------------------

anfr_group = folium.FeatureGroup(
    name="Supports ANFR connus",
    show=True,
)


for _, row in anfr.iterrows():

    if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
        continue

    popup = (
        f"<b>Support ANFR</b><br>"
        f"SUP_ID : {row.get('SUP_ID', '')}<br>"
        f"Nature : {row.get('NAT_LB_NOM', '')}<br>"
        f"Hauteur : {row.get('SUP_NM_HAUT', '')}"
    )

    folium.CircleMarker(
        location=[
            float(row["latitude"]),
            float(row["longitude"]),
        ],
        radius=4,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        popup=popup,
    ).add_to(
        anfr_group
    )


anfr_group.add_to(m)


# ------------------------------------------------------------
# CANDIDATS
# ------------------------------------------------------------

candidate_group = folium.FeatureGroup(
    name="Candidats inspectés",
    show=True,
)


status_colors = {
    "plausible": "green",
    "faux_positif": "red",
    "incertain": "orange",
}


for _, row in candidates.iterrows():

    status = str(
        row["review_status"]
    )

    color = status_colors.get(
        status,
        "gray",
    )

    popup = (
        f"<b>{row['unique_candidate_id']}</b><br>"
        f"Statut : {status}<br>"
        f"Confiance : {float(row['confidence']):.3f}<br>"
        f"Distance ANFR : "
        f"{float(row['nearest_ANFR_distance_m']):.1f} m<br>"
        f"Latitude : {float(row['latitude']):.6f}<br>"
        f"Longitude : {float(row['longitude']):.6f}<br>"
        f"Cluster : {row['cluster_candidate_ids']}"
    )

    folium.CircleMarker(
        location=[
            float(row["latitude"]),
            float(row["longitude"]),
        ],
        radius=8,
        color=color,
        weight=2,
        fill=True,
        fill_opacity=0.85,
        popup=popup,
        tooltip=(
            f"{row['unique_candidate_id']} "
            f"| {status}"
        ),
    ).add_to(
        candidate_group
    )


candidate_group.add_to(m)


folium.LayerControl().add_to(m)


# Zoom automatique
if len(all_lat) > 0:

    m.fit_bounds(
        [
            [
                float(all_lat.min()),
                float(all_lon.min()),
            ],
            [
                float(all_lat.max()),
                float(all_lon.max()),
            ],
        ]
    )


m.save(
    MAP_PATH
)


# ============================================================
# RESUME JSON
# ============================================================

summary = {
    "zone_pilote_patches": 25,
    "detections_initiales": 138,
    "detections_distance_ge_150m": 28,
    "candidats_uniques_apres_deduplication": int(
        len(candidates)
    ),
    "candidats_plausibles": int(
        len(plausible)
    ),
    "faux_positifs": int(
        len(false_positive)
    ),
    "incertains": int(
        len(uncertain)
    ),
    "taux_plausible": float(
        plausible_rate
    ),
    "seuil_distance_candidat_m": 150,
    "calibration_humaine_max_distance_anfr_m": 50.494606,
    "calibration_humaine_p95_m": 30.71,
    "note": (
        "Les candidats plausibles ne constituent pas "
        "une preuve de site radioélectrique non déclaré."
    ),
}


SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# RAPPORT MARKDOWN
# ============================================================

report = f"""# Niveau 3 — Recherche de candidats hors catalogue ANFR

## Méthode

Le modèle final du Niveau 2 a été appliqué sur une zone pilote
découpée en 25 patches IGN de 1024 × 1024 pixels.

Chaque détection a été géoréférencée puis comparée aux 1 464 supports
du catalogue ANFR des Yvelines.

La calibration sur 175 supports visibles annotés manuellement
(train + validation uniquement) montre :

- médiane : 11,73 m
- P95 : 30,71 m
- P99 : 47,16 m
- maximum : 50,49 m

Le seuil de sélection des candidats Niveau 3 a donc été fixé à 150 m,
soit très au-delà de la dispersion observée sur les sites connus.

## Résultats

- Détections brutes : 138
- Détections à au moins 150 m d'un support ANFR : 28
- Candidats géographiques après déduplication à 30 m : {len(candidates)}
- Candidats visuellement plausibles : {len(plausible)}
- Faux positifs : {len(false_positive)}
- Incertains : {len(uncertain)}
- Taux de candidats plausibles : {100 * plausible_rate:.1f} %

## Interprétation

Les candidats classés plausibles présentent visuellement une structure
compatible avec un support ou une infrastructure pouvant accueillir des
équipements radioélectriques.

Ils ne doivent cependant pas être interprétés comme des sites
radioélectriques non déclarés confirmés. Une validation terrain,
administrative ou par une source externe serait nécessaire pour conclure.

## Fichiers produits

- `candidats_plausibles.csv`
- `candidats_plausibles.geojson`
- `carte_candidats_niveau3.html`
- `resume_niveau3.json`
"""


REPORT_PATH.write_text(
    report,
    encoding="utf-8",
)


# ============================================================
# RESULTATS
# ============================================================

print("\n====================================")
print("NIVEAU 3 FINALISE")
print("====================================")

print(
    "Carte :",
    MAP_PATH
)

print(
    "CSV plausibles :",
    PLAUSIBLE_CSV
)

print(
    "GeoJSON :",
    GEOJSON_PATH
)

print(
    "Résumé :",
    SUMMARY_PATH
)

print(
    "Rapport :",
    REPORT_PATH
)

print(
    "\n✅ Carte interactive créée."
)

print(
    "✅ Candidats plausibles exportés."
)

print(
    "✅ Résultats quantitatifs sauvegardés."
)
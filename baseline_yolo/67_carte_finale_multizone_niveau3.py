from pathlib import Path

import pandas as pd
import folium
from folium.plugins import Fullscreen


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

ZONES_SUMMARY_PATH = (
    MULTI_DIR
    / "resume_zones.csv"
)

ZONES_ROOT = (
    MULTI_DIR
    / "zones"
)

OUT_DIR = (
    MULTI_DIR
    / "resultats_finaux"
)

OUT_HTML = (
    OUT_DIR
    / "carte_finale_multizone_niveau3.html"
)


# ============================================================
# VERIFICATIONS
# ============================================================

for path in [
    CANDIDATES_PATH,
    ZONES_SUMMARY_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHARGEMENT
# ============================================================

candidates = pd.read_csv(
    CANDIDATES_PATH
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


remaining = int(
    candidates[
        "review_status"
    ]
    .str.strip()
    .eq("")
    .sum()
)


if remaining:
    raise RuntimeError(
        f"{remaining} candidat(s) non examiné(s)."
    )


# ============================================================
# CENTRE GLOBAL
# ============================================================

center_lat = float(
    zones[
        "centre_latitude"
    ].mean()
)

center_lon = float(
    zones[
        "centre_longitude"
    ].mean()
)


m = folium.Map(
    location=[
        center_lat,
        center_lon,
    ],
    zoom_start=9,
    tiles="OpenStreetMap",
    control_scale=True,
)


Fullscreen(
    position="topright"
).add_to(m)


# ============================================================
# GROUPES
# ============================================================

zones_group = folium.FeatureGroup(
    name="Emprises des 3 zones",
    show=True,
)

anfr_group = folium.FeatureGroup(
    name="Sites ANFR dans les zones",
    show=True,
)

plausible_group = folium.FeatureGroup(
    name="Candidats plausibles",
    show=True,
)

false_group = folium.FeatureGroup(
    name="Faux positifs",
    show=False,
)

uncertain_group = folium.FeatureGroup(
    name="Candidats incertains",
    show=True,
)


zones_group.add_to(m)
anfr_group.add_to(m)
plausible_group.add_to(m)
false_group.add_to(m)
uncertain_group.add_to(m)


# ============================================================
# ZONES + ANFR
# ============================================================

zone_labels = {
    "dense":
        "Zone dense",

    "intermediaire":
        "Zone intermédiaire",

    "peu_dense":
        "Zone peu dense",
}


for _, zone in zones.iterrows():

    zone_id = str(
        zone["zone_id"]
    )

    label = zone_labels.get(
        zone_id,
        zone_id,
    )


    bounds = [
        [
            float(
                zone["south"]
            ),
            float(
                zone["west"]
            ),
        ],
        [
            float(
                zone["north"]
            ),
            float(
                zone["east"]
            ),
        ],
    ]


    folium.Rectangle(
        bounds=bounds,
        color="purple",
        weight=3,
        fill=False,
        tooltip=(
            f"{label} — "
            f"{int(zone['patches'])} patches"
        ),
        popup=folium.Popup(
            (
                f"<b>{label}</b><br>"
                f"Supports dans 500 m : "
                f"{int(zone['supports_500m'])}<br>"
                f"Supports ANFR dans l'emprise : "
                f"{int(zone['supports_dans_emprise'])}<br>"
                f"Patches : "
                f"{int(zone['patches'])}"
            ),
            max_width=350,
        ),
    ).add_to(
        zones_group
    )


    anfr_path = (
        ZONES_ROOT
        / zone_id
        / "anfr_dans_zone.csv"
    )


    if not anfr_path.exists():

        print(
            f"⚠️ ANFR absent pour {zone_id}"
        )

        continue


    anfr = pd.read_csv(
        anfr_path
    )


    for _, support in anfr.iterrows():

        lat = float(
            support["latitude"]
        )

        lon = float(
            support["longitude"]
        )

        sup_id = support.get(
            "SUP_ID",
            "?"
        )


        popup_parts = [
            "<b>Support ANFR connu</b>",
            f"SUP_ID : {sup_id}",
            f"Zone : {label}",
        ]


        if (
            "NAT_LB_NOM"
            in support.index
            and
            pd.notna(
                support["NAT_LB_NOM"]
            )
        ):

            popup_parts.append(
                f"Nature : "
                f"{support['NAT_LB_NOM']}"
            )


        if (
            "SUP_NM_HAUT"
            in support.index
            and
            pd.notna(
                support["SUP_NM_HAUT"]
            )
        ):

            popup_parts.append(
                f"Hauteur : "
                f"{support['SUP_NM_HAUT']} m"
            )


        folium.CircleMarker(
            location=[
                lat,
                lon,
            ],
            radius=5,
            color="blue",
            fill=True,
            fill_opacity=0.8,
            tooltip=(
                f"ANFR {sup_id}"
            ),
            popup=folium.Popup(
                "<br>".join(
                    popup_parts
                ),
                max_width=350,
            ),
        ).add_to(
            anfr_group
        )


# ============================================================
# CANDIDATS
# ============================================================

status_config = {
    "plausible": {
        "color":
            "green",

        "group":
            plausible_group,

        "label":
            "Candidat plausible",
    },

    "faux_positif": {
        "color":
            "red",

        "group":
            false_group,

        "label":
            "Faux positif",
    },

    "incertain": {
        "color":
            "orange",

        "group":
            uncertain_group,

        "label":
            "Candidat incertain",
    },
}


for _, row in candidates.iterrows():

    status = str(
        row[
            "review_status"
        ]
    ).strip()


    if status not in status_config:
        continue


    config = status_config[
        status
    ]


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

    support_type = str(
        row[
            "support_type"
        ]
    ).strip()


    if not support_type:
        support_type = "non classé"


    confidence = float(
        row[
            "confidence"
        ]
    )

    distance = float(
        row[
            "nearest_ANFR_distance_m"
        ]
    )

    latitude = float(
        row[
            "latitude"
        ]
    )

    longitude = float(
        row[
            "longitude"
        ]
    )


    popup = (
        f"<b>{uid}</b><br>"
        f"Statut : "
        f"{config['label']}<br>"
        f"Zone : {zone_id}<br>"
        f"Type visuel : "
        f"{support_type}<br>"
        f"Confiance modèle : "
        f"{confidence:.3f}<br>"
        f"Distance ANFR : "
        f"{distance:.1f} m<br>"
        f"Latitude : "
        f"{latitude:.6f}<br>"
        f"Longitude : "
        f"{longitude:.6f}"
    )


    folium.CircleMarker(
        location=[
            latitude,
            longitude,
        ],
        radius=8,
        color=config[
            "color"
        ],
        weight=3,
        fill=True,
        fill_opacity=0.8,
        tooltip=(
            f"{uid} | "
            f"{status} | "
            f"{support_type}"
        ),
        popup=folium.Popup(
            popup,
            max_width=380,
        ),
    ).add_to(
        config[
            "group"
        ]
    )


# ============================================================
# MISE EN AVANT DU MAT D003
# ============================================================

highlight = candidates[
    candidates[
        "unique_candidate_id"
    ] == "D003"
]


if len(highlight) == 1:

    row = highlight.iloc[0]

    folium.Marker(
        location=[
            float(
                row[
                    "latitude"
                ]
            ),
            float(
                row[
                    "longitude"
                ]
            ),
        ],
        icon=folium.Icon(
            color="darkgreen",
            icon="star",
            prefix="fa",
        ),
        tooltip=(
            "D003 — mât candidat"
        ),
        popup=folium.Popup(
            (
                "<b>D003 — mât candidat</b><br>"
                f"Confiance : "
                f"{float(row['confidence']):.3f}<br>"
                f"Distance ANFR : "
                f"{float(row['nearest_ANFR_distance_m']):.1f} m<br>"
                "<br>"
                "<b>Attention :</b> cette détection "
                "reste un candidat visuel et ne prouve "
                "pas l'existence d'un site radioélectrique "
                "non déclaré."
            ),
            max_width=400,
        ),
    ).add_to(
        plausible_group
    )


# ============================================================
# TITRE + LEGENDE
# ============================================================

title_html = """
<div style="
    position: fixed;
    top: 10px;
    left: 50px;
    right: 50px;
    z-index: 9999;
    background: white;
    padding: 10px 14px;
    border: 2px solid #444;
    border-radius: 6px;
    font-size: 16px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.35);
">
<b>FRHack 2026 — Challenge 4 — Niveau 3 multizone</b><br>
<span style="font-size:13px;">
Candidats détectés dans l'imagerie et éloignés d'au moins
150 m du support ANFR le plus proche.
Un candidat plausible n'est pas une preuve de site non déclaré.
</span>
</div>
"""

m.get_root().html.add_child(
    folium.Element(
        title_html
    )
)


legend_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    z-index: 9999;
    background: white;
    border: 2px solid #777;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 13px;
">
<b>Légende</b><br>
<span style="color:blue;">●</span>
Support ANFR connu<br>

<span style="color:green;">●</span>
Candidat plausible<br>

<span style="color:orange;">●</span>
Candidat incertain<br>

<span style="color:red;">●</span>
Faux positif<br>

<span style="color:purple;">▭</span>
Emprise analysée
</div>
"""

m.get_root().html.add_child(
    folium.Element(
        legend_html
    )
)


# ============================================================
# CONTROLES
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# FIT BOUNDS
# ============================================================

global_bounds = [
    [
        float(
            zones[
                "south"
            ].min()
        ),
        float(
            zones[
                "west"
            ].min()
        ),
    ],
    [
        float(
            zones[
                "north"
            ].max()
        ),
        float(
            zones[
                "east"
            ].max()
        ),
    ],
]


m.fit_bounds(
    global_bounds
)


# ============================================================
# EXPORT
# ============================================================

m.save(
    OUT_HTML
)


# ============================================================
# AUDIT
# ============================================================

print("====================================")
print("CARTE FINALE MULTIZONE")
print("====================================")

print(
    "Zones :",
    len(zones)
)

print(
    "Candidats :",
    len(candidates)
)

print(
    "Plausibles :",
    int(
        (
            candidates[
                "review_status"
            ]
            == "plausible"
        ).sum()
    )
)

print(
    "Faux positifs :",
    int(
        (
            candidates[
                "review_status"
            ]
            == "faux_positif"
        ).sum()
    )
)

print(
    "Incertains :",
    int(
        (
            candidates[
                "review_status"
            ]
            == "incertain"
        ).sum()
    )
)

print(
    "\nCarte :",
    OUT_HTML
)

print(
    "\n✅ Carte multizone générée."
)

print(
    "✅ Supports ANFR et candidats séparés en couches."
)

print(
    "✅ D003 mis en évidence comme candidat de type mât."
)

print(
    "⚠️ Aucun candidat n'est présenté comme "
    "un site non déclaré confirmé."
)
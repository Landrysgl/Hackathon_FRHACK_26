import os
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html


# ============================================================
# CONFIGURATION
# ============================================================

# Le CSV est dans le même dossier que ce script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(
    BASE_DIR,
    "SUP_SUPPORT_AVEC_TYPE.csv"
)

# Fichiers de sortie
PNG_FILE = os.path.join(
    BASE_DIR,
    "carte_antennes.png"
)

HTML_FILE = os.path.join(
    BASE_DIR,
    "carte_antennes.html"
)


# ============================================================
# LECTURE DU CSV
# ============================================================

print("Lecture du fichier :")
print(CSV_FILE)

df = pd.read_csv(
    CSV_FILE,
    sep=";",
    encoding="utf-8"
)

print(f"Nombre de lignes : {len(df)}")
print("Colonnes disponibles :")
print(df.columns.tolist())


# ============================================================
# CONVERSION DES COORDONNÉES DMS
# ============================================================

def dms_to_decimal(degrees, minutes, seconds, direction):
    """
    Convertit une coordonnée degrés/minutes/secondes
    en coordonnées décimales.
    """

    if pd.isna(degrees) or pd.isna(minutes) or pd.isna(seconds):
        return None

    decimal = (
        float(degrees)
        + float(minutes) / 60
        + float(seconds) / 3600
    )

    direction = str(direction).strip().upper()

    if direction in ["S", "W"]:
        decimal = -decimal

    return decimal


df["latitude"] = df.apply(
    lambda row: dms_to_decimal(
        row["COR_NB_DG_LAT"],
        row["COR_NB_MN_LAT"],
        row["COR_NB_SC_LAT"],
        row["COR_CD_NS_LAT"]
    ),
    axis=1
)

df["longitude"] = df.apply(
    lambda row: dms_to_decimal(
        row["COR_NB_DG_LON"],
        row["COR_NB_MN_LON"],
        row["COR_NB_SC_LON"],
        row["COR_CD_EW_LON"]
    ),
    axis=1
)


# ============================================================
# NETTOYAGE
# ============================================================

# On garde uniquement les lignes avec des coordonnées valides
df = df.dropna(
    subset=["latitude", "longitude"]
).copy()

# TYPE doit être présent pour pouvoir colorer les points
if "TYPE" not in df.columns:
    raise ValueError(
        "La colonne 'TYPE' n'existe pas dans "
        "SUP_SUPPORT_AVEC_TYPE.csv"
    )

# Évite les valeurs NaN dans la légende
df["TYPE"] = df["TYPE"].fillna("Type inconnu").astype(str)


print()
print(f"Nombre de supports avec coordonnées : {len(df)}")
print()
print("Répartition par TYPE :")
print(df["TYPE"].value_counts())


# ============================================================
# CRÉATION DE LA CARTE
# ============================================================

fig = px.scatter_map(
    df,
    lat="latitude",
    lon="longitude",
    color="TYPE",

    # Informations affichées au survol
    hover_name="SUP_ID",

    hover_data={
        "TYPE": True,
        "STA_NM_ANFR": True,
        "ADR_NM_CP": True,
        "COM_CD_INSEE": True,
        "latitude": ":.6f",
        "longitude": ":.6f",
    },

    zoom=5,
    height=750,

    # Fond de carte
    map_style="open-street-map",

    title="Répartition des antennes par type"
)


fig.update_layout(
    margin={
        "r": 0,
        "t": 50,
        "l": 0,
        "b": 0
    },

    legend={
        "title": "Type d'antenne"
    }
)


# ============================================================
# SAUVEGARDE AVANT AFFICHAGE
# ============================================================

print()
print("Sauvegarde de la carte...")


# ------------------------------------------------------------
# 1. Image PNG
# ------------------------------------------------------------

try:

    fig.write_image(
        PNG_FILE,
        width=1600,
        height=900,
        scale=2
    )

    print("Image PNG sauvegardée :")
    print(f"  {PNG_FILE}")

except Exception as e:

    print()
    print("Impossible de sauvegarder le PNG.")
    print("Installe probablement Kaleido avec :")
    print()
    print("    pip install kaleido")
    print()
    print(f"Détail : {e}")


# ------------------------------------------------------------
# 2. Version HTML interactive
# ------------------------------------------------------------

fig.write_html(
    HTML_FILE,
    include_plotlyjs=True
)

print("Carte HTML sauvegardée :")
print(f"  {HTML_FILE}")


# ============================================================
# DASH
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    [

        html.H1(
            "Carte des antennes",
            style={
                "textAlign": "center"
            }
        ),

        html.Div(
            f"{len(df):,} antennes affichées".replace(",", " "),
            style={
                "textAlign": "center",
                "marginBottom": "10px"
            }
        ),

        dcc.Graph(
            id="carte-antennes",
            figure=fig,
            style={
                "height": "85vh"
            }
        )

    ]
)


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":

    print()
    print("================================================")
    print("Carte sauvegardée.")
    print("Lancement de Dash...")
    print("================================================")
    print()

    app.run(
        host="0.0.0.0",
        port=8050,
        debug=False
    )
import pandas as pd

# ---------------------------------------------------------
# 1. Chargement des données ANFR
# ---------------------------------------------------------

chemin_supports = "data/raw/anfr/SUP_SUPPORT.txt"

supports = pd.read_csv(
    chemin_supports, sep=";", dtype={"STA_NM_ANFR": str, "COM_CD_INSEE": str}
)

print("Nombre de supports :", len(supports))
print("Nombre de colonnes :", len(supports.columns))

print("\nColonnes disponibles :")
print(supports.columns.tolist())


# ---------------------------------------------------------
# 2. Conversion des coordonnées DMS vers degrés décimaux
# ---------------------------------------------------------


def convertir_dms(degres, minutes, secondes, direction):
    """
    Convertit une coordonnée en degrés/minutes/secondes
    vers des degrés décimaux.
    """

    coordonnee = degres + minutes / 60 + secondes / 3600

    if direction in ["S", "W"]:
        coordonnee = -coordonnee

    return coordonnee


supports["latitude"] = supports.apply(
    lambda ligne: convertir_dms(
        ligne["COR_NB_DG_LAT"],
        ligne["COR_NB_MN_LAT"],
        ligne["COR_NB_SC_LAT"],
        ligne["COR_CD_NS_LAT"],
    ),
    axis=1,
)

supports["longitude"] = supports.apply(
    lambda ligne: convertir_dms(
        ligne["COR_NB_DG_LON"],
        ligne["COR_NB_MN_LON"],
        ligne["COR_NB_SC_LON"],
        ligne["COR_CD_EW_LON"],
    ),
    axis=1,
)


# ---------------------------------------------------------
# 3. Vérification des coordonnées obtenues
# ---------------------------------------------------------

colonnes_utiles = [
    "SUP_ID",
    "STA_NM_ANFR",
    "NAT_ID",
    "latitude",
    "longitude",
    "SUP_NM_HAUT",
    "ADR_NM_CP",
    "COM_CD_INSEE",
]

print("\nPremiers supports :")
print(supports[colonnes_utiles].head(10))


# ---------------------------------------------------------
# 4. Informations générales
# ---------------------------------------------------------

print("\nNombre de stations ANFR différentes :")
print(supports["STA_NM_ANFR"].nunique())

print("\nNombre de types NAT_ID différents :")
print(supports["NAT_ID"].nunique())

print("\nTypes les plus fréquents :")
print(supports["NAT_ID"].value_counts().head(20))

# ---------------------------------------------------------
# 5. Chargement de la table des natures de supports
# ---------------------------------------------------------

chemin_natures = "data/raw/references/SUP_NATURE.txt"

natures = pd.read_csv(
    chemin_natures, sep=";", header=None, names=["NAT_ID", "NATURE_SUPPORT"]
)
# Harmonisation du type de la clé NAT_ID
natures["NAT_ID"] = pd.to_numeric(natures["NAT_ID"], errors="coerce")

natures = natures.dropna(subset=["NAT_ID"])
natures["NAT_ID"] = natures["NAT_ID"].astype("int64")

print("\nTable des natures :")
print(natures.head(10))


# ---------------------------------------------------------
# 6. Association des supports avec leur nature
# ---------------------------------------------------------

supports = supports.merge(natures, on="NAT_ID", how="left")

print("\nSupports avec leur nature :")

colonnes_resultat = [
    "SUP_ID",
    "STA_NM_ANFR",
    "NAT_ID",
    "NATURE_SUPPORT",
    "latitude",
    "longitude",
    "SUP_NM_HAUT",
    "COM_CD_INSEE",
]

print(supports[colonnes_resultat].head(20))


# ---------------------------------------------------------
# 7. Répartition des principales natures de supports
# ---------------------------------------------------------

print("\nNatures de supports les plus fréquentes :")

print(supports["NATURE_SUPPORT"].value_counts().head(20))

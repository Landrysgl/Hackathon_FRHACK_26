import pandas as pd

chemin_supports = "data/raw/anfr/SUP_SUPPORT.txt"

supports = pd.read_csv(
    chemin_supports, sep=";", dtype={"STA_NM_ANFR": str, "COM_CD_INSEE": str}
)

# Création du code département à partir du code INSEE
supports["DEPARTEMENT"] = supports["COM_CD_INSEE"].str[:2]

# Comptage du nombre de supports par département
comptage_departements = (
    supports["DEPARTEMENT"].value_counts().sort_values(ascending=False)
)

print("Nombre de supports par département :")
print(comptage_departements.head(30))

# ---------------------------------------------------------
# Sélection du département des Yvelines (78)
# ---------------------------------------------------------

supports_78 = supports[supports["DEPARTEMENT"] == "78"].copy()

print("\nNombre de supports sélectionnés dans le département 78 :")
print(len(supports_78))

print("\nPremiers supports sélectionnés :")
print(
    supports_78[
        [
            "SUP_ID",
            "STA_NM_ANFR",
            "NAT_ID",
            "COR_NB_DG_LAT",
            "COR_NB_MN_LAT",
            "COR_NB_SC_LAT",
            "COR_NB_DG_LON",
            "COR_NB_MN_LON",
            "COR_NB_SC_LON",
            "COM_CD_INSEE",
        ]
    ].head(10)
)

# ---------------------------------------------------------
# Conversion des coordonnées DMS en degrés décimaux
# ---------------------------------------------------------


def convertir_dms(degres, minutes, secondes, direction):
    coordonnee = degres + minutes / 60 + secondes / 3600

    if direction in ["S", "W"]:
        coordonnee = -coordonnee

    return coordonnee


supports_78["latitude"] = supports_78.apply(
    lambda ligne: convertir_dms(
        ligne["COR_NB_DG_LAT"],
        ligne["COR_NB_MN_LAT"],
        ligne["COR_NB_SC_LAT"],
        ligne["COR_CD_NS_LAT"],
    ),
    axis=1,
)

supports_78["longitude"] = supports_78.apply(
    lambda ligne: convertir_dms(
        ligne["COR_NB_DG_LON"],
        ligne["COR_NB_MN_LON"],
        ligne["COR_NB_SC_LON"],
        ligne["COR_CD_EW_LON"],
    ),
    axis=1,
)


# ---------------------------------------------------------
# Création d'un fichier propre pour la zone d'étude
# ---------------------------------------------------------

colonnes_utiles = [
    "SUP_ID",
    "STA_NM_ANFR",
    "NAT_ID",
    "latitude",
    "longitude",
    "SUP_NM_HAUT",
    "COM_CD_INSEE",
]

supports_78_propres = supports_78[colonnes_utiles].copy()

supports_78_propres.to_csv("data/supports_yvelines.csv", index=False)

print("\nFichier créé : data/supports_yvelines.csv")
print("Nombre de supports :", len(supports_78_propres))

print("\nAperçu :")
print(supports_78_propres.head(10))

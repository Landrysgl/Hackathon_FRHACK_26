from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def trouver_fichier(*chemins_possibles: Path) -> Path:
    for chemin in chemins_possibles:
        if chemin.exists():
            return chemin
    essais = "\n".join(f" - {p}" for p in chemins_possibles)
    raise FileNotFoundError(f"Aucun fichier trouvé parmi :\n{essais}")


def lire_csv_robuste(path: Path, **kwargs) -> pd.DataFrame:
    derniere_erreur = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            derniere_erreur = exc
    raise derniere_erreur


SUPPORT_PATH = trouver_fichier(
    DATA_DIR / "raw" / "anfr" / "SUP_SUPPORT.txt",
    ROOT / "baseline_yolo" / "data" / "raw" / "anfr" / "SUP_SUPPORT.txt",
)

NATURE_PATH = trouver_fichier(
    DATA_DIR / "raw" / "references" / "SUP_NATURE.txt",
    DATA_DIR / "raw" / "ref_antennes" / "SUP_NATURE.txt",
    DATA_DIR / "raw" / "ref_atennes" / "SUP_NATURE.txt",
    ROOT / "baseline_yolo" / "data" / "raw" / "references" / "SUP_NATURE.txt",
)

OUTPUT_PATH = DATA_DIR / "supports_yvelines.csv"

print("===== FICHIERS UTILISES =====")
print("SUP_SUPPORT :", SUPPORT_PATH)
print("SUP_NATURE  :", NATURE_PATH)
print("Sortie      :", OUTPUT_PATH)

supports = lire_csv_robuste(
    SUPPORT_PATH,
    sep=";",
    dtype={
        "SUP_ID": "string",
        "STA_NM_ANFR": "string",
        "COM_CD_INSEE": "string",
        "ADR_NM_CP": "string",
        "COR_CD_NS_LAT": "string",
        "COR_CD_EW_LON": "string",
    },
    low_memory=False,
)

print("\n===== DONNEES INITIALES =====")
print("Nombre de lignes ANFR :", len(supports))
print("Nombre de supports physiques uniques :", supports["SUP_ID"].nunique())

if "COM_CD_INSEE" not in supports.columns:
    raise ValueError("La colonne COM_CD_INSEE est absente.")

supports["DEPARTEMENT"] = supports["COM_CD_INSEE"].str[:2]

print("\n===== DEPARTEMENTS LES PLUS REPRESENTES =====")
print(supports["DEPARTEMENT"].value_counts().head(30))

yvelines = supports[supports["DEPARTEMENT"] == "78"].copy()

print("\n===== YVELINES =====")
print("Nombre de lignes ANFR dans le 78 :", len(yvelines))
print("Nombre de SUP_ID uniques dans le 78 :", yvelines["SUP_ID"].nunique())
print(
    "Nombre de lignes supplémentaires liées aux doublons :",
    len(yvelines) - yvelines["SUP_ID"].nunique(),
)

physical_cols = [
    "NAT_ID",
    "COR_NB_DG_LAT", "COR_NB_MN_LAT", "COR_NB_SC_LAT", "COR_CD_NS_LAT",
    "COR_NB_DG_LON", "COR_NB_MN_LON", "COR_NB_SC_LON", "COR_CD_EW_LON",
    "SUP_NM_HAUT",
    "COM_CD_INSEE",
]

physical_cols = [c for c in physical_cols if c in yvelines.columns]

duplicated_ids = yvelines.loc[
    yvelines["SUP_ID"].duplicated(keep=False),
    "SUP_ID",
].unique()

incoherents = []
for sup_id in duplicated_ids:
    groupe = yvelines[yvelines["SUP_ID"] == sup_id]
    for col in physical_cols:
        if groupe[col].nunique(dropna=False) > 1:
            incoherents.append((sup_id, col))

if incoherents:
    raise ValueError(
        "Certains SUP_ID répétés ont des caractéristiques physiques incohérentes. "
        f"Exemples : {incoherents[:10]}"
    )

print("Vérification des doublons : OK (les caractéristiques physiques sont cohérentes)")

nb_stations = (
    yvelines.groupby("SUP_ID")["STA_NM_ANFR"]
    .nunique()
    .rename("NB_STATIONS_ANFR")
)

supports_uniques = (
    yvelines
    .drop_duplicates("SUP_ID", keep="first")
    .merge(nb_stations, on="SUP_ID", how="left", validate="one_to_one")
)

print("\nNombre final de supports physiques uniques :", len(supports_uniques))
print(
    "Supports accueillant plusieurs stations ANFR :",
    int((supports_uniques["NB_STATIONS_ANFR"] > 1).sum()),
)

for col in [
    "COR_NB_DG_LAT", "COR_NB_MN_LAT", "COR_NB_SC_LAT",
    "COR_NB_DG_LON", "COR_NB_MN_LON", "COR_NB_SC_LON",
]:
    supports_uniques[col] = pd.to_numeric(supports_uniques[col], errors="coerce")

ns = supports_uniques["COR_CD_NS_LAT"].str.strip().str.upper()
ew = supports_uniques["COR_CD_EW_LON"].str.strip().str.upper()


def dms_decimal(deg, minute, seconde, direction):
    valeur = deg + minute / 60.0 + seconde / 3600.0
    signe = np.where(direction.isin(["S", "W"]), -1.0, 1.0)
    return valeur * signe


supports_uniques["latitude"] = dms_decimal(
    supports_uniques["COR_NB_DG_LAT"],
    supports_uniques["COR_NB_MN_LAT"],
    supports_uniques["COR_NB_SC_LAT"],
    ns,
)

supports_uniques["longitude"] = dms_decimal(
    supports_uniques["COR_NB_DG_LON"],
    supports_uniques["COR_NB_MN_LON"],
    supports_uniques["COR_NB_SC_LON"],
    ew,
)

coord_invalides = (
    supports_uniques["latitude"].isna()
    | supports_uniques["longitude"].isna()
    | ~supports_uniques["latitude"].between(-90, 90)
    | ~supports_uniques["longitude"].between(-180, 180)
)

print("\nCoordonnées invalides :", int(coord_invalides.sum()))
if coord_invalides.any():
    raise ValueError("Des supports des Yvelines ont des coordonnées invalides.")

print(
    "Latitude min / max :",
    supports_uniques["latitude"].min(),
    "/",
    supports_uniques["latitude"].max(),
)
print(
    "Longitude min / max :",
    supports_uniques["longitude"].min(),
    "/",
    supports_uniques["longitude"].max(),
)

natures = lire_csv_robuste(NATURE_PATH, sep=";", low_memory=False)

if not {"NAT_ID", "NAT_LB_NOM"}.issubset(natures.columns):
    raise ValueError("SUP_NATURE.txt doit contenir NAT_ID et NAT_LB_NOM.")

natures["NAT_ID"] = pd.to_numeric(natures["NAT_ID"], errors="raise")
supports_uniques["NAT_ID"] = pd.to_numeric(supports_uniques["NAT_ID"], errors="coerce")

supports_uniques = supports_uniques.merge(
    natures[["NAT_ID", "NAT_LB_NOM"]],
    on="NAT_ID",
    how="left",
    validate="many_to_one",
)

if supports_uniques["NAT_LB_NOM"].isna().any():
    raise ValueError("Certaines natures de support n'ont pas été résolues.")

colonnes_sortie = [
    "SUP_ID",
    "STA_NM_ANFR",
    "NB_STATIONS_ANFR",
    "NAT_ID",
    "NAT_LB_NOM",
    "latitude",
    "longitude",
    "SUP_NM_HAUT",
    "ADR_LB_LIEU",
    "ADR_LB_ADD1",
    "ADR_NM_CP",
    "COM_CD_INSEE",
]

colonnes_sortie = [c for c in colonnes_sortie if c in supports_uniques.columns]

sortie = (
    supports_uniques[colonnes_sortie]
    .sort_values("SUP_ID")
    .reset_index(drop=True)
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
sortie.to_csv(OUTPUT_PATH, index=False)

print("\n==============================")
print("FICHIER YVELINES CREE")
print("==============================")
print("Fichier :", OUTPUT_PATH)
print("Nombre de supports physiques :", len(sortie))

print("\n===== TYPES DE SUPPORT =====")
print(sortie["NAT_LB_NOM"].value_counts())

print("\n===== APERCU =====")
print(sortie.head(10).to_string(index=False))

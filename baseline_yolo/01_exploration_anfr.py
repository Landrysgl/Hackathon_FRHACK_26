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

print("===== FICHIERS UTILISES =====")
print("SUP_SUPPORT :", SUPPORT_PATH)
print("SUP_NATURE  :", NATURE_PATH)

dtype_support = {
    "SUP_ID": "string",
    "STA_NM_ANFR": "string",
    "COM_CD_INSEE": "string",
    "ADR_NM_CP": "string",
    "COR_CD_NS_LAT": "string",
    "COR_CD_EW_LON": "string",
}

supports = lire_csv_robuste(
    SUPPORT_PATH,
    sep=";",
    dtype=dtype_support,
    low_memory=False,
)

colonnes_requises = [
    "SUP_ID", "STA_NM_ANFR", "NAT_ID",
    "COR_NB_DG_LAT", "COR_NB_MN_LAT", "COR_NB_SC_LAT", "COR_CD_NS_LAT",
    "COR_NB_DG_LON", "COR_NB_MN_LON", "COR_NB_SC_LON", "COR_CD_EW_LON",
]

manquantes = [c for c in colonnes_requises if c not in supports.columns]
if manquantes:
    raise ValueError(f"Colonnes manquantes dans SUP_SUPPORT.txt : {manquantes}")

print("\n===== DONNEES INITIALES =====")
print("Nombre de lignes ANFR :", len(supports))
print("Nombre de colonnes :", len(supports.columns))
print("Nombre de supports physiques uniques :", supports["SUP_ID"].nunique())
print("Nombre de stations ANFR uniques :", supports["STA_NM_ANFR"].nunique())
print(
    "Nombre de lignes supplémentaires liées aux SUP_ID répétés :",
    len(supports) - supports["SUP_ID"].nunique(),
)

dms_cols = [
    "COR_NB_DG_LAT", "COR_NB_MN_LAT", "COR_NB_SC_LAT",
    "COR_NB_DG_LON", "COR_NB_MN_LON", "COR_NB_SC_LON",
]

for col in dms_cols:
    supports[col] = pd.to_numeric(supports[col], errors="coerce")

coord_incompletes = supports[dms_cols].isna().any(axis=1)
print("Lignes avec coordonnées DMS incomplètes :", int(coord_incompletes.sum()))

ns = supports["COR_CD_NS_LAT"].str.strip().str.upper()
ew = supports["COR_CD_EW_LON"].str.strip().str.upper()

directions_invalides = (~ns.isin(["N", "S"])) | (~ew.isin(["E", "W"]))
print("Directions N/S/E/W invalides :", int(directions_invalides.sum()))


def dms_decimal(deg, minute, seconde, direction):
    valeur = deg + minute / 60.0 + seconde / 3600.0
    signe = np.where(direction.isin(["S", "W"]), -1.0, 1.0)
    return valeur * signe


supports["latitude"] = dms_decimal(
    supports["COR_NB_DG_LAT"],
    supports["COR_NB_MN_LAT"],
    supports["COR_NB_SC_LAT"],
    ns,
)

supports["longitude"] = dms_decimal(
    supports["COR_NB_DG_LON"],
    supports["COR_NB_MN_LON"],
    supports["COR_NB_SC_LON"],
    ew,
)

coords_invalides = (
    ~supports["latitude"].between(-90, 90)
    | ~supports["longitude"].between(-180, 180)
)

print("Coordonnées décimales invalides :", int(coords_invalides.sum()))

natures = lire_csv_robuste(
    NATURE_PATH,
    sep=";",
    low_memory=False,
)

if not {"NAT_ID", "NAT_LB_NOM"}.issubset(natures.columns):
    raise ValueError(
        "SUP_NATURE.txt doit contenir les colonnes NAT_ID et NAT_LB_NOM. "
        f"Colonnes trouvées : {list(natures.columns)}"
    )

natures["NAT_ID"] = pd.to_numeric(natures["NAT_ID"], errors="raise")

if natures["NAT_ID"].duplicated().any():
    raise ValueError("SUP_NATURE.txt contient des NAT_ID dupliqués.")

supports["NAT_ID"] = pd.to_numeric(supports["NAT_ID"], errors="coerce")

fusion = supports.merge(
    natures[["NAT_ID", "NAT_LB_NOM"]],
    on="NAT_ID",
    how="left",
    validate="many_to_one",
)

print("Natures manquantes après fusion :", int(fusion["NAT_LB_NOM"].isna().sum()))

print("\n===== TYPES DE SUPPORT - LIGNES ANFR =====")
print(fusion["NAT_LB_NOM"].value_counts().head(20))

supports_uniques = fusion.drop_duplicates("SUP_ID")

print("\n===== TYPES DE SUPPORT - SUPPORTS PHYSIQUES UNIQUES =====")
print(supports_uniques["NAT_LB_NOM"].value_counts().head(20))

print("\n===== RESUME =====")
print("Lignes ANFR :", len(fusion))
print("Supports physiques uniques :", fusion["SUP_ID"].nunique())
print("Stations ANFR uniques :", fusion["STA_NM_ANFR"].nunique())
print("Coordonnées invalides :", int(coords_invalides.sum()))
print("Natures non résolues :", int(fusion["NAT_LB_NOM"].isna().sum()))

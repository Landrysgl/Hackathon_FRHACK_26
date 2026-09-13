from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geo import EARTH_RADIUS_M

REQUIRED_COLUMNS = {"SUP_ID", "latitude", "longitude"}


def load_supports(path: str | Path) -> pd.DataFrame:
    """Charge un catalogue de supports ANFR déjà géocodé en WGS84."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, dtype={"SUP_ID": "string"})
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes ANFR manquantes: {sorted(missing)}")

    for col in ("latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)]
    df = df.drop_duplicates(subset=["SUP_ID"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Le catalogue ANFR ne contient aucun support géolocalisé valide.")
    return df


class SupportIndex:
    """Index vectorisé simple pour le plus proche support ANFR."""

    def __init__(self, supports: pd.DataFrame):
        if supports.empty:
            raise ValueError("Le catalogue ANFR est vide.")
        self.supports = supports.reset_index(drop=True).copy()
        self._lat_rad = np.radians(self.supports["latitude"].to_numpy(dtype=float))
        self._lon_rad = np.radians(self.supports["longitude"].to_numpy(dtype=float))

    def nearest(self, latitude: float, longitude: float) -> tuple[pd.Series, float]:
        lat1 = np.radians(float(latitude))
        lon1 = np.radians(float(longitude))
        dlat = self._lat_rad - lat1
        dlon = self._lon_rad - lon1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(self._lat_rad) * np.sin(dlon / 2.0) ** 2
        )
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        distances = EARTH_RADIUS_M * c
        idx = int(np.argmin(distances))
        return self.supports.iloc[idx], float(distances[idx])


def _read_semicolon_file(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep=";", encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def prepare_department_reference(
    support_file: str | Path,
    nature_file: str | Path,
    department_code: str,
) -> pd.DataFrame:
    """Construit un catalogue de supports physiques à partir des exports bruts ANFR.

    `support_file` doit être un `SUP_SUPPORT.txt` et `nature_file` un `SUP_NATURE.txt`.
    Les coordonnées DMS sont converties en WGS84 décimal et les lignes partageant le
    même `SUP_ID` sont ramenées à un support physique unique.
    """
    dtype = {
        "SUP_ID": "string",
        "STA_NM_ANFR": "string",
        "COM_CD_INSEE": "string",
        "ADR_NM_CP": "string",
        "COR_CD_NS_LAT": "string",
        "COR_CD_EW_LON": "string",
    }
    supports = _read_semicolon_file(support_file, dtype=dtype)
    required = {
        "SUP_ID", "STA_NM_ANFR", "NAT_ID", "COM_CD_INSEE",
        "COR_NB_DG_LAT", "COR_NB_MN_LAT", "COR_NB_SC_LAT", "COR_CD_NS_LAT",
        "COR_NB_DG_LON", "COR_NB_MN_LON", "COR_NB_SC_LON", "COR_CD_EW_LON",
    }
    missing = required - set(supports.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans SUP_SUPPORT.txt: {sorted(missing)}")

    supports = supports[supports["COM_CD_INSEE"].str[:2] == str(department_code)].copy()
    if supports.empty:
        raise ValueError(f"Aucun support trouvé pour le département {department_code}.")

    # Vérifie qu'un SUP_ID dupliqué ne décrit pas plusieurs géométries physiques.
    physical_cols = [
        "NAT_ID", "COR_NB_DG_LAT", "COR_NB_MN_LAT", "COR_NB_SC_LAT", "COR_CD_NS_LAT",
        "COR_NB_DG_LON", "COR_NB_MN_LON", "COR_NB_SC_LON", "COR_CD_EW_LON",
        "SUP_NM_HAUT", "COM_CD_INSEE",
    ]
    physical_cols = [c for c in physical_cols if c in supports.columns]
    for sup_id, group in supports.groupby("SUP_ID", dropna=False):
        for col in physical_cols:
            if group[col].nunique(dropna=False) > 1:
                raise ValueError(f"SUP_ID {sup_id} incohérent pour la colonne {col}.")

    station_counts = supports.groupby("SUP_ID")["STA_NM_ANFR"].nunique().rename("NB_STATIONS_ANFR")
    unique = (
        supports.drop_duplicates("SUP_ID", keep="first")
        .merge(station_counts, on="SUP_ID", how="left", validate="one_to_one")
    )

    numeric_cols = [
        "COR_NB_DG_LAT", "COR_NB_MN_LAT", "COR_NB_SC_LAT",
        "COR_NB_DG_LON", "COR_NB_MN_LON", "COR_NB_SC_LON",
    ]
    for col in numeric_cols:
        unique[col] = pd.to_numeric(unique[col], errors="coerce")

    ns = unique["COR_CD_NS_LAT"].str.strip().str.upper()
    ew = unique["COR_CD_EW_LON"].str.strip().str.upper()
    if (~ns.isin(["N", "S"])).any() or (~ew.isin(["E", "W"])).any():
        raise ValueError("Direction N/S/E/W invalide dans les coordonnées ANFR.")

    lat = unique["COR_NB_DG_LAT"] + unique["COR_NB_MN_LAT"] / 60 + unique["COR_NB_SC_LAT"] / 3600
    lon = unique["COR_NB_DG_LON"] + unique["COR_NB_MN_LON"] / 60 + unique["COR_NB_SC_LON"] / 3600
    unique["latitude"] = lat * np.where(ns.eq("S"), -1.0, 1.0)
    unique["longitude"] = lon * np.where(ew.eq("W"), -1.0, 1.0)

    invalid = unique["latitude"].isna() | unique["longitude"].isna()
    invalid |= ~unique["latitude"].between(-90, 90) | ~unique["longitude"].between(-180, 180)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} support(s) avec coordonnées invalides.")

    natures = _read_semicolon_file(nature_file)
    if not {"NAT_ID", "NAT_LB_NOM"}.issubset(natures.columns):
        raise ValueError("SUP_NATURE.txt doit contenir NAT_ID et NAT_LB_NOM.")
    natures["NAT_ID"] = pd.to_numeric(natures["NAT_ID"], errors="raise")
    if natures["NAT_ID"].duplicated().any():
        raise ValueError("SUP_NATURE.txt contient des NAT_ID dupliqués.")
    unique["NAT_ID"] = pd.to_numeric(unique["NAT_ID"], errors="coerce")
    unique = unique.merge(
        natures[["NAT_ID", "NAT_LB_NOM"]], on="NAT_ID", how="left", validate="many_to_one"
    )
    if unique["NAT_LB_NOM"].isna().any():
        raise ValueError("Certaines natures de support n'ont pas été résolues.")

    columns = [
        "SUP_ID", "STA_NM_ANFR", "NB_STATIONS_ANFR", "NAT_ID", "NAT_LB_NOM",
        "latitude", "longitude", "SUP_NM_HAUT", "ADR_LB_LIEU", "ADR_LB_ADD1",
        "ADR_NM_CP", "COM_CD_INSEE",
    ]
    return unique[[c for c in columns if c in unique.columns]].sort_values("SUP_ID").reset_index(drop=True)

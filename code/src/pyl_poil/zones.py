from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .geo import EARTH_RADIUS_M, haversine_m


def local_density_counts(
    supports: pd.DataFrame,
    radius_m: float = 500.0,
) -> np.ndarray:
    """Compte les supports situés dans ``radius_m`` autour de chaque support.

    Cette fonction reproduit la stratégie utilisée pendant le hackathon pour
    choisir des contextes dense / intermédiaire / peu dense.
    """
    if radius_m <= 0:
        raise ValueError("radius_m doit être strictement positif.")
    required = {"latitude", "longitude"}
    missing = required - set(supports.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")
    if supports.empty:
        raise ValueError("Le catalogue de supports est vide.")

    lat = pd.to_numeric(supports["latitude"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(supports["longitude"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(lat).any() or np.isnan(lon).any():
        raise ValueError("Le catalogue contient des coordonnées invalides.")

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat_ref = float(lat_rad.mean())
    x_m = EARTH_RADIUS_M * lon_rad * math.cos(lat_ref)
    y_m = EARTH_RADIUS_M * lat_rad
    radius2 = radius_m**2

    counts = np.zeros(len(supports), dtype=int)
    for i in range(len(supports)):
        dx = x_m - x_m[i]
        dy = y_m - y_m[i]
        counts[i] = int(np.sum(dx * dx + dy * dy <= radius2))
    return counts


def _far_enough(
    row: pd.Series,
    selected: Iterable[pd.Series],
    min_separation_m: float,
) -> bool:
    for other in selected:
        if haversine_m(
            float(row["latitude"]),
            float(row["longitude"]),
            float(other["latitude"]),
            float(other["longitude"]),
        ) < min_separation_m:
            return False
    return True


def select_contrasting_zones(
    supports: pd.DataFrame,
    *,
    density_radius_m: float = 500.0,
    min_separation_m: float = 5_000.0,
) -> pd.DataFrame:
    """Sélectionne trois centres: dense, intermédiaire et peu dense.

    Algorithme fidèle à la sélection canonique du Niveau 3:
    1. zone dense = densité locale maximale;
    2. zone intermédiaire = densité la plus proche de la médiane parmi les
       supports assez éloignés de la zone dense;
    3. zone peu dense = densité minimale parmi les supports assez éloignés des
       deux premières zones, puis candidat maximisant la distance minimale aux
       deux centres déjà retenus.
    """
    if min_separation_m <= 0:
        raise ValueError("min_separation_m doit être strictement positif.")
    required = {"SUP_ID", "latitude", "longitude"}
    missing = required - set(supports.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")

    df = supports.copy().reset_index(drop=True)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Au moins trois supports sont nécessaires.")

    df["density_500m"] = local_density_counts(df, density_radius_m)

    dense = df.loc[int(df["density_500m"].idxmax())]

    medium_pool = df[
        df.apply(
            lambda row: _far_enough(row, [dense], min_separation_m),
            axis=1,
        )
    ].copy()
    if medium_pool.empty:
        raise ValueError("Impossible de trouver une zone intermédiaire assez éloignée.")
    target_median = float(medium_pool["density_500m"].median())
    medium_pool["distance_to_target_density"] = (
        medium_pool["density_500m"] - target_median
    ).abs()
    medium = df.loc[int(medium_pool["distance_to_target_density"].idxmin())]

    sparse_pool = df[
        df.apply(
            lambda row: _far_enough(row, [dense, medium], min_separation_m),
            axis=1,
        )
    ].copy()
    if sparse_pool.empty:
        raise ValueError("Impossible de trouver une zone peu dense assez éloignée.")
    min_density = int(sparse_pool["density_500m"].min())
    sparse_candidates = sparse_pool[sparse_pool["density_500m"] == min_density].copy()

    def min_distance_to_selected(row: pd.Series) -> float:
        return min(
            haversine_m(
                float(row["latitude"]),
                float(row["longitude"]),
                float(selected["latitude"]),
                float(selected["longitude"]),
            )
            for selected in (dense, medium)
        )

    sparse_candidates["distance_selected"] = sparse_candidates.apply(
        min_distance_to_selected, axis=1
    )
    sparse = df.loc[int(sparse_candidates["distance_selected"].idxmax())]

    return pd.DataFrame(
        [
            {
                "zone_id": "dense",
                "SUP_ID": dense["SUP_ID"],
                "latitude": float(dense["latitude"]),
                "longitude": float(dense["longitude"]),
                "supports_500m": int(dense["density_500m"]),
            },
            {
                "zone_id": "intermediaire",
                "SUP_ID": medium["SUP_ID"],
                "latitude": float(medium["latitude"]),
                "longitude": float(medium["longitude"]),
                "supports_500m": int(medium["density_500m"]),
            },
            {
                "zone_id": "peu_dense",
                "SUP_ID": sparse["SUP_ID"],
                "latitude": float(sparse["latitude"]),
                "longitude": float(sparse["longitude"]),
                "supports_500m": int(sparse["density_500m"]),
            },
        ]
    )

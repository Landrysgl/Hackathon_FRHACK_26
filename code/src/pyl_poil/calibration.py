from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .anfr import SupportIndex
from .geo import haversine_m, latlon_to_world_pixel, world_pixel_to_latlon


def calibrate_visual_anfr_offset(
    annotation_files: list[str | Path],
    supports: pd.DataFrame,
    *,
    zoom: int = 19,
    image_size: int = 1024,
    tile_size: int = 256,
) -> pd.DataFrame:
    """Mesure l'écart entre centre visuel annoté et référentiel ANFR.

    Seules les lignes ``visible`` avec bbox complète sont utilisées. Les images
    d'apprentissage étant centrées sur la coordonnée ANFR, la position du
    centre de bbox peut être convertie directement en WGS84.
    """
    frames: list[pd.DataFrame] = []
    for file in annotation_files:
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"SUP_ID": "string"})
        frame["source"] = path.stem
        frames.append(frame)
    if not frames:
        raise ValueError("Aucun fichier d'annotations fourni.")

    annotations = pd.concat(frames, ignore_index=True)
    required = {
        "SUP_ID", "status", "latitude", "longitude", "x1", "y1", "x2", "y2"
    }
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Colonnes d'annotations manquantes: {sorted(missing)}")

    visible = annotations[annotations["status"].fillna("").astype(str).eq("visible")].copy()
    if visible.empty:
        raise ValueError("Aucune annotation visible disponible pour la calibration.")
    for col in ("latitude", "longitude", "x1", "y1", "x2", "y2"):
        visible[col] = pd.to_numeric(visible[col], errors="coerce")
    if visible[["latitude", "longitude", "x1", "y1", "x2", "y2"]].isna().any().any():
        raise ValueError("Certaines annotations visibles ont des coordonnées/bbox manquantes.")

    support_index = SupportIndex(supports)
    support_lookup = supports.copy()
    support_lookup["SUP_ID"] = support_lookup["SUP_ID"].astype(str)
    support_lookup = support_lookup.set_index("SUP_ID", drop=False)

    rows: list[dict] = []
    for _, row in visible.iterrows():
        official_lat = float(row["latitude"])
        official_lon = float(row["longitude"])
        bbox_cx = (float(row["x1"]) + float(row["x2"])) / 2.0
        bbox_cy = (float(row["y1"]) + float(row["y2"])) / 2.0

        official_px_x, official_px_y = latlon_to_world_pixel(
            official_lat, official_lon, zoom, tile_size
        )
        visual_lat, visual_lon = world_pixel_to_latlon(
            official_px_x + bbox_cx - image_size / 2.0,
            official_px_y + bbox_cy - image_size / 2.0,
            zoom,
            tile_size,
        )
        distance_own = haversine_m(
            official_lat, official_lon, visual_lat, visual_lon
        )
        nearest, nearest_distance = support_index.nearest(visual_lat, visual_lon)

        rows.append(
            {
                "SUP_ID": str(row["SUP_ID"]),
                "source": str(row.get("source", "")),
                "classe_niveau2": row.get("classe_niveau2", None),
                "official_latitude": official_lat,
                "official_longitude": official_lon,
                "bbox_center_x": bbox_cx,
                "bbox_center_y": bbox_cy,
                "visual_latitude": visual_lat,
                "visual_longitude": visual_lon,
                "distance_to_own_anfr_m": distance_own,
                "nearest_SUP_ID": str(nearest["SUP_ID"]),
                "distance_to_nearest_anfr_m": nearest_distance,
            }
        )

    return pd.DataFrame(rows)


def calibration_summary(
    calibration: pd.DataFrame,
    candidate_distance_threshold_m: float = 150.0,
) -> dict[str, float | int | str]:
    distances = pd.to_numeric(
        calibration["distance_to_nearest_anfr_m"], errors="coerce"
    ).dropna()
    if distances.empty:
        raise ValueError("Aucune distance exploitable dans la calibration.")
    return {
        "visible_supports_used": int(len(distances)),
        "mean_m": float(distances.mean()),
        "median_m": float(distances.median()),
        "p95_m": float(np.percentile(distances, 95)),
        "p97_5_m": float(np.percentile(distances, 97.5)),
        "p99_m": float(np.percentile(distances, 99)),
        "max_m": float(distances.max()),
        "candidate_distance_threshold_m": float(candidate_distance_threshold_m),
        "interpretation": (
            "Le seuil candidat est volontairement conservateur par rapport à la "
            "dispersion observée sur les supports visibles annotés."
        ),
    }

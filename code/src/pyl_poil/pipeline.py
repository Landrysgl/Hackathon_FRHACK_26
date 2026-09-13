from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .anfr import load_supports
from .candidates import build_candidates
from .config import AppConfig
from .detector import detect_geolocated
from .geo import cartoradio_url
from .imagery import IgnOrthophotoClient
from .mapping import build_interactive_map


def analyze_area(
    center_latitude: float,
    center_longitude: float,
    supports_path: str | Path,
    weights_path: str | Path,
    output_dir: str | Path,
    config: AppConfig,
    *,
    zone_id: str = "analyse",
    device: str | int | None = "auto",
) -> dict[str, Path | int]:
    """Pipeline complet: IGN -> YOLO -> géolocalisation -> filtre ANFR -> carte."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    supports = load_supports(supports_path)

    imagery_dir = output_dir / "imagery"
    client = IgnOrthophotoClient(config.ign, output_dir / "cache_ign")
    metadata = client.download_grid(
        center_latitude, center_longitude, imagery_dir, zone_id=zone_id
    )

    detections = detect_geolocated(
        metadata,
        imagery_dir,
        weights_path,
        supports,
        imgsz=config.detection.imgsz,
        confidence=config.detection.confidence,
        nms_iou=config.detection.nms_iou,
        tile_size=config.ign.tile_size,
        device=device,
    )
    detections_path = output_dir / "detections.csv"
    detections.to_csv(detections_path, index=False)

    far, candidates = build_candidates(
        detections,
        min_distance_m=config.detection.min_anfr_distance_m,
        cluster_distance_m=config.detection.dedup_distance_m,
    )
    far_path = output_dir / "detections_hors_correspondance_anfr.csv"
    candidates_path = output_dir / "candidats_uniques.csv"
    far.to_csv(far_path, index=False)

    if not candidates.empty:
        candidates["cartoradio_url"] = [
            cartoradio_url(float(lat), float(lon))
            for lat, lon in zip(candidates["latitude"], candidates["longitude"])
        ]
    candidates.to_csv(candidates_path, index=False)

    map_path = build_interactive_map(
        candidates,
        output_dir / "carte_candidats.html",
        supports=supports,
    )

    summary = {
        "center_latitude": center_latitude,
        "center_longitude": center_longitude,
        "patches": int(len(metadata)),
        "detections": int(len(detections)),
        "detections_min_distance": int(len(far)),
        "candidates_unique": int(len(candidates)),
        "confidence_threshold": config.detection.confidence,
        "min_anfr_distance_m": config.detection.min_anfr_distance_m,
        "dedup_distance_m": config.detection.dedup_distance_m,
        "warning": "Un candidat n'est pas une preuve de site radioélectrique non déclaré.",
    }
    summary_path = output_dir / "resume.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "metadata": imagery_dir / "metadata_patches.csv",
        "detections": detections_path,
        "far_detections": far_path,
        "candidates": candidates_path,
        "map": map_path,
        "summary": summary_path,
        "detections_count": len(detections),
        "candidates_count": len(candidates),
    }

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class IgnConfig:
    wmts_url: str = "https://data.geopf.fr/wmts"
    layer: str = "ORTHOIMAGERY.ORTHOPHOTOS"
    tile_matrix_set: str = "PM"
    style: str = "normal"
    image_format: str = "image/jpeg"
    zoom: int = 19
    tile_size: int = 256
    patch_size: int = 1024
    grid_size: int = 5
    jpeg_quality: int = 95
    request_timeout_s: int = 30
    request_retries: int = 3
    request_delay_s: float = 0.03


@dataclass(frozen=True)
class DetectionConfig:
    imgsz: int = 1024
    confidence: float = 0.05
    nms_iou: float = 0.50
    min_anfr_distance_m: float = 150.0
    dedup_distance_m: float = 30.0


@dataclass(frozen=True)
class AppConfig:
    ign: IgnConfig = IgnConfig()
    detection: DetectionConfig = DetectionConfig()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"La section '{name}' doit être un objet YAML.")
    return value


def load_config(path: str | Path | None = None) -> AppConfig:
    """Charge la configuration YAML, ou renvoie les valeurs par défaut."""
    if path is None:
        return AppConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Le fichier de configuration doit contenir un objet YAML.")

    return AppConfig(
        ign=IgnConfig(**_section(raw, "ign")),
        detection=DetectionConfig(**_section(raw, "detection")),
    )

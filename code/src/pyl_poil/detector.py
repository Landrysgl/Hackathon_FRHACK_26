from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .anfr import SupportIndex
from .geo import world_pixel_to_latlon


def _resolve_device(device: str | int | None) -> str | int:
    if device not in (None, "auto"):
        return device
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_yolo(weights: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics n'est pas installé. Exécutez: pip install -r requirements.txt"
        ) from exc
    return YOLO(str(weights))


def detect_geolocated(
    metadata: pd.DataFrame,
    images_root: str | Path,
    weights: str | Path,
    supports: pd.DataFrame,
    *,
    imgsz: int = 1024,
    confidence: float = 0.05,
    nms_iou: float = 0.50,
    tile_size: int = 256,
    device: str | int | None = "auto",
) -> pd.DataFrame:
    """Lance YOLO sur des patches géoréférencés et rattache chaque détection à l'ANFR."""
    images_root = Path(images_root)
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    required_metadata = {
        "image", "relative_image_path", "world_left", "world_top", "zoom", "row", "col"
    }
    missing = required_metadata - set(metadata.columns)
    if missing:
        raise ValueError(f"Métadonnées de patches incomplètes: {sorted(missing)}")

    model = _load_yolo(weights)
    support_index = SupportIndex(supports)
    resolved_device = _resolve_device(device)
    detections: list[dict] = []
    detection_id = 0

    for patch in metadata.to_dict("records"):
        image_path = images_root / str(patch["relative_image_path"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        result = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=confidence,
            iou=nms_iou,
            device=resolved_device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            detection_id += 1
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].detach().cpu().tolist()]
            conf = float(box.conf[0].detach().cpu())
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            latitude, longitude = world_pixel_to_latlon(
                float(patch["world_left"]) + center_x,
                float(patch["world_top"]) + center_y,
                int(patch["zoom"]),
                tile_size,
            )
            nearest, distance_m = support_index.nearest(latitude, longitude)

            record = {
                "detection_id": detection_id,
                "zone_id": str(patch.get("zone_id", "zone")),
                "image": str(patch["image"]),
                "row": int(patch["row"]),
                "col": int(patch["col"]),
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y,
                "latitude": latitude,
                "longitude": longitude,
                "nearest_SUP_ID": str(nearest["SUP_ID"]),
                "nearest_ANFR_latitude": float(nearest["latitude"]),
                "nearest_ANFR_longitude": float(nearest["longitude"]),
                "nearest_ANFR_distance_m": distance_m,
                "nearest_ANFR_nature": nearest.get("NAT_LB_NOM", None),
                "nearest_ANFR_height": nearest.get("SUP_NM_HAUT", None),
            }
            detections.append(record)

    columns = [
        "detection_id", "zone_id", "image", "row", "col", "confidence",
        "x1", "y1", "x2", "y2", "center_x", "center_y", "latitude", "longitude",
        "nearest_SUP_ID", "nearest_ANFR_latitude", "nearest_ANFR_longitude",
        "nearest_ANFR_distance_m", "nearest_ANFR_nature", "nearest_ANFR_height",
    ]
    return pd.DataFrame(detections, columns=columns)

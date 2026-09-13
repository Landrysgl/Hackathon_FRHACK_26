from __future__ import annotations

from pathlib import Path
from typing import Any


def _yolo_class() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics n'est pas installé. Exécutez: pip install -r requirements.txt"
        ) from exc
    return YOLO


def _auto_device() -> str | int:
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def train_detector(
    dataset_yaml: str | Path,
    output_dir: str | Path,
    *,
    base_model: str = "yolov8n.pt",
    run_name: str = "pyl_poil_training",
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = 8,
    patience: int = 15,
    seed: int = 42,
    degrees: float = 0.0,
    device: str | int | None = "auto",
):
    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(dataset_yaml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _yolo_class()(base_model)
    resolved_device = _auto_device() if device in (None, "auto") else device
    return model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        seed=seed,
        deterministic=True,
        device=resolved_device,
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
        degrees=degrees,
    )


def evaluate_detector(
    weights: str | Path,
    dataset_yaml: str | Path,
    *,
    imgsz: int = 1024,
    device: str | int | None = "auto",
):
    weights = Path(weights)
    dataset_yaml = Path(dataset_yaml)
    if not weights.exists():
        raise FileNotFoundError(weights)
    if not dataset_yaml.exists():
        raise FileNotFoundError(dataset_yaml)
    resolved_device = _auto_device() if device in (None, "auto") else device
    model = _yolo_class()(str(weights))
    return model.val(data=str(dataset_yaml), imgsz=imgsz, device=resolved_device)

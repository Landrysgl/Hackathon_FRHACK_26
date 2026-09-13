#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pyl_poil.training import train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne le détecteur Pyl-Poil.")
    parser.add_argument("--data", type=Path, required=True, help="dataset.yaml YOLO")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--run-name", default="pyl_poil_training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = int(args.device) if str(args.device).isdigit() else args.device
    train_detector(
        args.data,
        args.output,
        base_model=args.base_model,
        run_name=args.run_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        degrees=args.degrees,
        device=device,
    )


if __name__ == "__main__":
    main()

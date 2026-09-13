#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pyl_poil.training import evaluate_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue un modèle YOLO sur un dataset annoté.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = int(args.device) if str(args.device).isdigit() else args.device
    metrics = evaluate_detector(args.weights, args.data, imgsz=args.imgsz, device=device)
    print(f"mAP50    : {metrics.box.map50:.6f}")
    print(f"mAP50-95 : {metrics.box.map:.6f}")
    print(f"Precision: {metrics.box.mp:.6f}")
    print(f"Recall   : {metrics.box.mr:.6f}")


if __name__ == "__main__":
    main()

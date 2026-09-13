#!/usr/bin/env python3
"""Construit le dataset YOLO Niveau 1 à annotations faibles centrales."""
from __future__ import annotations

import argparse
from pathlib import Path

from pyl_poil.level1 import build_weak_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Crée les labels faibles du Niveau 1.")
    parser.add_argument("--images", type=Path, default=Path("data/generated/support_images"))
    parser.add_argument("--split", type=Path, default=Path("data/reference/spatial_split_300.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/dataset_level1_weak"))
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--weak-box-size", type=int, default=160)
    args = parser.parse_args()

    yaml_path = build_weak_yolo_dataset(
        args.images,
        args.split,
        args.output,
        image_size=args.image_size,
        weak_box_size_px=args.weak_box_size,
    )
    print("Dataset Niveau 1 créé :", yaml_path)


if __name__ == "__main__":
    main()

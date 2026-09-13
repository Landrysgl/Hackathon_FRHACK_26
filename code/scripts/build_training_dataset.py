#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pyl_poil.dataset import build_human_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruit le dataset YOLO final depuis les annotations humaines.")
    parser.add_argument("--images", type=Path, default=Path("data/generated/support_images"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/dataset_final"))
    args = parser.parse_args()

    result = build_human_yolo_dataset(
        args.images,
        [
            Path("data/annotations/level2_train_batch1.csv"),
            Path("data/annotations/level2_train_batch2.csv"),
        ],
        [Path("data/annotations/level2_validation.csv")],
        args.output,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

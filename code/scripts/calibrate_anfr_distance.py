#!/usr/bin/env python3
"""Reproduit la calibration du seuil de distance ANFR utilisée au Niveau 3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pyl_poil.calibration import calibrate_visual_anfr_offset, calibration_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mesure l'écart entre le centre visuel des supports annotés et le "
            "référentiel ANFR sur le train + validation."
        )
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        type=Path,
        default=[
            Path("data/annotations/level2_train_batch1.csv"),
            Path("data/annotations/level2_train_batch2.csv"),
            Path("data/annotations/level2_validation.csv"),
        ],
    )
    parser.add_argument(
        "--supports",
        type=Path,
        default=Path("data/reference/supports_yvelines.csv"),
    )
    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--candidate-threshold", type=float, default=150.0)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/calibration/anfr_visual_offset_calibration.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/calibration/anfr_distance_calibration.json"),
    )
    args = parser.parse_args()

    supports = pd.read_csv(args.supports, dtype={"SUP_ID": "string"})
    calibration = calibrate_visual_anfr_offset(
        args.annotations,
        supports,
        zoom=args.zoom,
        image_size=args.image_size,
        tile_size=args.tile_size,
    )
    summary = calibration_summary(calibration, args.candidate_threshold)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(args.output_csv, index=False)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nCSV  : {args.output_csv}")
    print(f"JSON : {args.output_json}")


if __name__ == "__main__":
    main()

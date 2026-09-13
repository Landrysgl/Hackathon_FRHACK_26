#!/usr/bin/env python3
"""Reproduit la sélection de 300 supports et le split spatial du Niveau 1."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pyl_poil.level1 import create_spatial_split, minimum_inter_split_distance, select_level1_supports


def main() -> None:
    parser = argparse.ArgumentParser(description="Sélectionne les supports Niveau 1 et crée le split spatial 70/15/15.")
    parser.add_argument("--supports", type=Path, default=Path("data/reference/supports_yvelines.csv"))
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-distance", type=float, default=200.0)
    parser.add_argument("--selection-output", type=Path, default=Path("outputs/level1/supports_selected_300.csv"))
    parser.add_argument("--split-output", type=Path, default=Path("outputs/level1/spatial_split_300.csv"))
    args = parser.parse_args()

    supports = pd.read_csv(args.supports, dtype={"SUP_ID": "string"})
    selected = select_level1_supports(supports, count=args.count, seed=args.seed)
    split = create_spatial_split(
        selected,
        grouping_distance_m=args.group_distance,
        seed=args.seed,
    )
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.selection_output, index=False)
    split.to_csv(args.split_output, index=False)

    print("Sélection :", len(selected))
    print(split["split"].value_counts().to_string())
    print(f"Distance inter-split minimale : {minimum_inter_split_distance(split):.1f} m")
    print("Sélection ->", args.selection_output)
    print("Split      ->", args.split_output)


if __name__ == "__main__":
    main()

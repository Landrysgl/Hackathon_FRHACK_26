#!/usr/bin/env python3
"""Reproduit la sélection des trois contextes géographiques du Niveau 3."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pyl_poil.zones import select_contrasting_zones


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sélectionne trois centres contrastés (dense, intermédiaire, peu dense) "
            "à partir du catalogue ANFR."
        )
    )
    parser.add_argument(
        "--supports",
        type=Path,
        default=Path("data/reference/supports_yvelines.csv"),
    )
    parser.add_argument("--density-radius", type=float, default=500.0)
    parser.add_argument("--min-separation", type=float, default=5000.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/level3/zones_selected.csv"),
    )
    args = parser.parse_args()

    supports = pd.read_csv(args.supports, dtype={"SUP_ID": "string"})
    zones = select_contrasting_zones(
        supports,
        density_radius_m=args.density_radius,
        min_separation_m=args.min_separation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    zones.to_csv(args.output, index=False)

    print(zones.to_string(index=False))
    print(f"\nÉcrit dans : {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rejoue le filtre 150 m + la déduplication 30 m à partir des détections sauvegardées."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pyl_poil.candidates import build_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, default=Path("data/level3/detections_multizone.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/rebuild_level3"))
    parser.add_argument("--min-distance", type=float, default=150.0)
    parser.add_argument("--dedup-distance", type=float, default=30.0)
    args = parser.parse_args()

    if not args.detections.exists():
        raise FileNotFoundError(args.detections)
    detections = pd.read_csv(args.detections)
    far, unique = build_candidates(detections, args.min_distance, args.dedup_distance)
    args.output.mkdir(parents=True, exist_ok=True)
    far.to_csv(args.output / "detections_ge150m.csv", index=False)
    unique.to_csv(args.output / "candidats_uniques.csv", index=False)

    print(f"Détections brutes : {len(detections)}")
    print(f"Détections >= {args.min_distance:g} m : {len(far)}")
    print(f"Candidats uniques : {len(unique)}")


if __name__ == "__main__":
    main()

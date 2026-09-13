from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import analyze_area


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyl-poil",
        description="Détection de candidats radioélectriques sur orthophotos IGN.",
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude WGS84 du centre")
    parser.add_argument("--lon", type=float, required=True, help="Longitude WGS84 du centre")
    parser.add_argument("--supports", type=Path, required=True, help="CSV de supports ANFR géolocalisés")
    parser.add_argument("--weights", type=Path, required=True, help="Poids YOLO .pt")
    parser.add_argument("--output", type=Path, default=Path("outputs/analyse"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zone-id", default="analyse")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device: str | int = int(args.device) if str(args.device).isdigit() else args.device
    outputs = analyze_area(
        args.lat,
        args.lon,
        args.supports,
        args.weights,
        args.output,
        load_config(args.config),
        zone_id=args.zone_id,
        device=device,
    )
    print("Analyse terminée")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pyl_poil.config import load_config
from pyl_poil.dataset import download_centered_support_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharge les 300 orthophotos de référence du hackathon.")
    parser.add_argument("--supports", type=Path, default=Path("data/reference/supports_selected_300.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/support_images"))
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    metadata = download_centered_support_images(
        args.supports,
        args.output,
        config=config.ign,
        cache_dir=args.output.parent / "cache_ign",
        overwrite=args.overwrite,
    )
    print(f"{len(metadata)} images disponibles dans {args.output}")


if __name__ == "__main__":
    main()

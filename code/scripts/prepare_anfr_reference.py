#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pyl_poil.anfr import prepare_department_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare un catalogue départemental à partir des exports ANFR bruts.")
    parser.add_argument("--supports", type=Path, required=True, help="SUP_SUPPORT.txt")
    parser.add_argument("--natures", type=Path, required=True, help="SUP_NATURE.txt")
    parser.add_argument("--department", default="78", help="Code département, par défaut 78")
    parser.add_argument("--output", type=Path, default=Path("data/reference/supports_yvelines.csv"))
    args = parser.parse_args()

    df = prepare_department_reference(args.supports, args.natures, args.department)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"{len(df)} supports physiques écrits dans {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

from pyl_poil.anfr import load_supports
from pyl_poil.mapping import build_interactive_map

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "level3" / "reviewed_candidates.csv"
SUPPORTS = ROOT / "data" / "reference" / "supports_yvelines.csv"
OUTPUT = ROOT / "results" / "level3" / "carte_candidats_finale.html"


def main() -> None:
    candidates = pd.read_csv(CANDIDATES)
    supports = load_supports(SUPPORTS)
    path = build_interactive_map(
        candidates,
        OUTPUT,
        supports=supports,
        final_ids={"D003", "D006", "D007"},
        title="Pyl-Poil - résultats Niveau 3",
    )
    print(path)


if __name__ == "__main__":
    main()

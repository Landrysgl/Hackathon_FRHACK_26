from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from pyl_poil.zones import select_contrasting_zones

ROOT = Path(__file__).resolve().parents[1]


class ZoneSelectionTests(unittest.TestCase):
    def test_reference_zone_selection_is_reproduced(self) -> None:
        supports = pd.read_csv(
            ROOT / "data/reference/supports_yvelines.csv", dtype={"SUP_ID": "string"}
        )
        expected = pd.read_csv(
            ROOT / "data/level3/zones_selected.csv", dtype={"SUP_ID": "string"}
        )
        actual = select_contrasting_zones(supports)
        actual["SUP_ID"] = actual["SUP_ID"].astype(str)

        self.assertEqual(actual["zone_id"].tolist(), expected["zone_id"].tolist())
        self.assertEqual(actual["SUP_ID"].tolist(), expected["SUP_ID"].tolist())
        self.assertEqual(actual["supports_500m"].tolist(), expected["supports_500m"].tolist())
        self.assertTrue(
            abs(actual["latitude"].to_numpy() - expected["latitude"].to_numpy()).max() < 1e-12
        )
        self.assertTrue(
            abs(actual["longitude"].to_numpy() - expected["longitude"].to_numpy()).max() < 1e-12
        )


if __name__ == "__main__":
    unittest.main()

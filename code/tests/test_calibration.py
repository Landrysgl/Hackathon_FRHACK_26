from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyl_poil.calibration import calibrate_visual_anfr_offset, calibration_summary

ROOT = Path(__file__).resolve().parents[1]


class CalibrationTests(unittest.TestCase):
    def test_reference_calibration_is_reproduced(self) -> None:
        files = [
            ROOT / "data/annotations/level2_train_batch1.csv",
            ROOT / "data/annotations/level2_train_batch2.csv",
            ROOT / "data/annotations/level2_validation.csv",
        ]
        supports = pd.read_csv(
            ROOT / "data/reference/supports_yvelines.csv", dtype={"SUP_ID": "string"}
        )
        actual = calibrate_visual_anfr_offset(files, supports)
        expected = pd.read_csv(
            ROOT / "data/level3/anfr_visual_offset_calibration.csv",
            dtype={"SUP_ID": "string"},
        )

        self.assertEqual(len(actual), 175)
        self.assertEqual(set(actual["SUP_ID"]), set(expected["SUP_ID"]))

        a = actual.set_index("SUP_ID").sort_index()
        e = expected.set_index("SUP_ID").sort_index()
        self.assertTrue(
            np.allclose(
                a["distance_to_nearest_anfr_m"],
                e["distance_to_nearest_anfr_m"],
                rtol=0,
                atol=1e-8,
            )
        )

        summary = calibration_summary(actual, 150.0)
        reference = json.loads(
            (ROOT / "results/metrics/anfr_distance_calibration.json").read_text(
                encoding="utf-8"
            )
        )
        for key in ("mean_m", "median_m", "p95_m", "p97_5_m", "p99_m", "max_m"):
            self.assertAlmostEqual(float(summary[key]), float(reference[key]), places=8)
        self.assertEqual(summary["visible_supports_used"], 175)
        self.assertEqual(summary["candidate_distance_threshold_m"], 150.0)


if __name__ == "__main__":
    unittest.main()

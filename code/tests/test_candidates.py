import unittest

import pandas as pd

from pyl_poil.candidates import build_candidates


class CandidateTests(unittest.TestCase):
    def test_filter_and_dedup(self):
        df = pd.DataFrame(
            [
                {"detection_id": 1, "zone_id": "A", "confidence": 0.4, "latitude": 48.0, "longitude": 2.0, "nearest_ANFR_distance_m": 160.0},
                {"detection_id": 2, "zone_id": "A", "confidence": 0.8, "latitude": 48.00005, "longitude": 2.00005, "nearest_ANFR_distance_m": 165.0},
                {"detection_id": 3, "zone_id": "A", "confidence": 0.9, "latitude": 48.01, "longitude": 2.01, "nearest_ANFR_distance_m": 80.0},
                {"detection_id": 4, "zone_id": "B", "confidence": 0.5, "latitude": 48.00005, "longitude": 2.00005, "nearest_ANFR_distance_m": 170.0},
            ]
        )
        far, unique = build_candidates(df, 150.0, 30.0)
        self.assertEqual(len(far), 3)
        self.assertEqual(len(unique), 2)
        representative_ids = set(unique["detection_id"].astype(int))
        self.assertEqual(representative_ids, {2, 4})


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

import pandas as pd

from pyl_poil.candidates import build_candidates
from pyl_poil.geo import world_pixel_to_latlon

ROOT = Path(__file__).resolve().parents[1]


class ReferenceResultTests(unittest.TestCase):
    def test_level3_geometric_pipeline_reproduces_counts_and_representatives(self):
        detections = pd.read_csv(ROOT / "data" / "level3" / "detections_multizone.csv")
        far, unique = build_candidates(detections, 150.0, 30.0)
        reviewed = pd.read_csv(ROOT / "data" / "level3" / "reviewed_candidates.csv")
        self.assertEqual(len(detections), 174)
        self.assertEqual(len(far), 53)
        self.assertEqual(len(unique), 46)
        self.assertEqual(
            set(unique["detection_id"].astype(int)),
            set(reviewed["detection_id"].astype(int)),
        )

    def test_saved_geolocation_is_consistent_with_patch_metadata(self):
        detections = pd.read_csv(ROOT / "data" / "level3" / "detections_multizone.csv")
        metadata = pd.read_csv(ROOT / "data" / "level3" / "multizone_patch_metadata.csv")
        detection = detections.iloc[0]
        patch = metadata[
            (metadata["zone_id"] == detection["zone_id"])
            & (metadata["image"] == detection["image"])
        ].iloc[0]
        lat, lon = world_pixel_to_latlon(
            float(patch["world_left"]) + float(detection["center_x"]),
            float(patch["world_top"]) + float(detection["center_y"]),
            int(patch["zoom"]),
        )
        self.assertAlmostEqual(lat, float(detection["latitude"]), places=9)
        self.assertAlmostEqual(lon, float(detection["longitude"]), places=9)

    def test_final_candidate_ids(self):
        final = pd.read_csv(ROOT / "results" / "level3" / "final_candidates.csv")
        self.assertEqual(final["unique_candidate_id"].tolist(), ["D003", "D006", "D007"])
        self.assertTrue((final["review_status"] == "plausible").all())
        self.assertFalse((final["support_type"] == "indetermine").any())


if __name__ == "__main__":
    unittest.main()

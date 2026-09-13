from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
from PIL import Image
import tempfile

from pyl_poil.level1 import (
    build_weak_yolo_dataset,
    create_spatial_split,
    minimum_inter_split_distance,
    select_level1_supports,
)

ROOT = Path(__file__).resolve().parents[1]


class Level1ReproductionTests(unittest.TestCase):
    def test_selection_and_spatial_split_are_reproduced(self) -> None:
        supports = pd.read_csv(ROOT / "data/reference/supports_yvelines.csv", dtype={"SUP_ID": "string"})
        expected_selection = pd.read_csv(ROOT / "data/reference/supports_selected_300.csv", dtype={"SUP_ID": "string"})
        expected_split = pd.read_csv(ROOT / "data/reference/spatial_split_300.csv", dtype={"SUP_ID": "string"})

        selected = select_level1_supports(supports)
        split = create_spatial_split(selected)

        self.assertEqual(selected["SUP_ID"].tolist(), expected_selection["SUP_ID"].tolist())
        self.assertEqual(split["SUP_ID"].tolist(), expected_split["SUP_ID"].tolist())
        self.assertEqual(split["split"].tolist(), expected_split["split"].tolist())
        self.assertEqual(split["split"].value_counts().to_dict(), {"train": 210, "val": 45, "test": 45})
        self.assertAlmostEqual(minimum_inter_split_distance(split), 216.2, delta=0.1)

    def test_weak_dataset_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            images = base / "images"
            images.mkdir()
            manifest = pd.DataFrame(
                [
                    {"SUP_ID": "1", "split": "train"},
                    {"SUP_ID": "2", "split": "val"},
                    {"SUP_ID": "3", "split": "test"},
                ]
            )
            for support_id in ("1", "2", "3"):
                Image.new("RGB", (1024, 1024), "white").save(
                    images / f"support_{support_id}.jpg"
                )
            yaml_path = build_weak_yolo_dataset(
                images, manifest, base / "dataset", weak_box_size_px=160
            )
            self.assertTrue(yaml_path.exists())
            expected = "0 0.50000000 0.50000000 0.15625000 0.15625000\n"
            for support_id, split in (("1", "train"), ("2", "val"), ("3", "test")):
                label = base / "dataset" / "labels" / split / f"support_{support_id}.txt"
                self.assertEqual(label.read_text(encoding="utf-8"), expected)


if __name__ == "__main__":
    unittest.main()

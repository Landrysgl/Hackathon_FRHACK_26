import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from pyl_poil.dataset import build_human_yolo_dataset


class DatasetBuilderTests(unittest.TestCase):
    def test_build_human_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            for name in ("support_1.jpg", "support_2.jpg", "support_3.jpg", "support_4.jpg"):
                Image.new("RGB", (1024, 1024), "white").save(images / name)

            train = pd.DataFrame([
                {"SUP_ID": "1", "image": "support_1.jpg", "status": "visible", "x1": 100, "y1": 200, "x2": 300, "y2": 400},
                {"SUP_ID": "2", "image": "support_2.jpg", "status": "non_visible"},
                {"SUP_ID": "3", "image": "support_3.jpg", "status": "ambigu"},
            ])
            val = pd.DataFrame([
                {"SUP_ID": "4", "image": "support_4.jpg", "status": "visible", "x1": 400, "y1": 400, "x2": 500, "y2": 500},
            ])
            train_path = root / "train.csv"
            val_path = root / "val.csv"
            train.to_csv(train_path, index=False)
            val.to_csv(val_path, index=False)

            out = root / "dataset"
            result = build_human_yolo_dataset(images, [train_path], [val_path], out)
            self.assertEqual(result["train_usable"], 2)
            self.assertEqual(result["train_ambiguous_excluded"], 1)
            self.assertEqual(result["val_usable"], 1)
            self.assertTrue((out / "dataset.yaml").exists())
            self.assertTrue((out / "labels/train/support_1.txt").read_text().startswith("0 "))
            self.assertEqual((out / "labels/train/support_2.txt").read_text(), "")
            self.assertFalse((out / "images/train/support_3.jpg").exists())


if __name__ == "__main__":
    unittest.main()

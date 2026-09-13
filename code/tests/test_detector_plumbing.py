import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from pyl_poil.detector import detect_geolocated
from pyl_poil.geo import latlon_to_world_pixel


class _FakeTensor:
    def __init__(self, value):
        self.value = value
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self.value
    def __float__(self):
        if isinstance(self.value, list):
            return float(self.value[0])
        return float(self.value)
    def __getitem__(self, index):
        if isinstance(self.value, list) and self.value and isinstance(self.value[0], list):
            return _FakeTensor(self.value[index])
        if isinstance(self.value, list):
            return _FakeTensor(self.value[index])
        raise TypeError


class _FakeBox:
    xyxy = _FakeTensor([[500.0, 500.0, 524.0, 524.0]])
    conf = _FakeTensor([0.75])


class _FakeResult:
    boxes = [_FakeBox()]


class _FakeYOLO:
    def __init__(self, weights):
        self.weights = weights
    def predict(self, **kwargs):
        return [_FakeResult()]


class DetectorPlumbingTests(unittest.TestCase):
    def test_detection_geolocation_and_nearest_anfr(self):
        fake_module = types.ModuleType("ultralytics")
        fake_module.YOLO = _FakeYOLO
        previous = sys.modules.get("ultralytics")
        sys.modules["ultralytics"] = fake_module
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                image = root / "patch.jpg"
                Image.new("RGB", (1024, 1024), "white").save(image)
                weights = root / "weights.pt"
                weights.write_bytes(b"fake")
                center_lat, center_lon = 48.8, 2.1
                world_x, world_y = latlon_to_world_pixel(center_lat, center_lon, 19)
                metadata = pd.DataFrame([{
                    "zone_id": "test", "image": "patch.jpg", "relative_image_path": "patch.jpg",
                    "row": 0, "col": 0, "zoom": 19,
                    "world_left": world_x - 512, "world_top": world_y - 512,
                }])
                supports = pd.DataFrame([{
                    "SUP_ID": "S1", "latitude": center_lat, "longitude": center_lon,
                    "NAT_LB_NOM": "Pylône", "SUP_NM_HAUT": 30,
                }])
                detections = detect_geolocated(
                    metadata, root, weights, supports, device="cpu"
                )
                self.assertEqual(len(detections), 1)
                row = detections.iloc[0]
                self.assertAlmostEqual(float(row["latitude"]), center_lat, places=6)
                self.assertAlmostEqual(float(row["longitude"]), center_lon, places=6)
                self.assertLess(float(row["nearest_ANFR_distance_m"]), 0.01)
                self.assertEqual(str(row["nearest_SUP_ID"]), "S1")
        finally:
            if previous is None:
                del sys.modules["ultralytics"]
            else:
                sys.modules["ultralytics"] = previous


if __name__ == "__main__":
    unittest.main()

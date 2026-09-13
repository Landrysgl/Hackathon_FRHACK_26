import unittest

from pyl_poil.geo import (
    cartoradio_url,
    haversine_m,
    latlon_to_world_pixel,
    world_pixel_to_latlon,
)


class GeoTests(unittest.TestCase):
    def test_web_mercator_roundtrip(self):
        lat, lon = 48.7975, 2.13
        x, y = latlon_to_world_pixel(lat, lon, 19)
        lat2, lon2 = world_pixel_to_latlon(x, y, 19)
        self.assertAlmostEqual(lat, lat2, places=8)
        self.assertAlmostEqual(lon, lon2, places=8)

    def test_haversine_identity(self):
        self.assertAlmostEqual(haversine_m(48.8, 2.1, 48.8, 2.1), 0.0, places=6)

    def test_cartoradio_lonlat_order(self):
        url = cartoradio_url(48.824979, 2.280235)
        self.assertTrue(url.endswith("/2.280235/48.824979"))


if __name__ == "__main__":
    unittest.main()

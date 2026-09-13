from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_008.8
WEB_MERCATOR_MAX_LAT = 85.05112878


def validate_latlon(latitude: float, longitude: float) -> None:
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError(f"Latitude invalide: {latitude}")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError(f"Longitude invalide: {longitude}")


def latlon_to_world_pixel(
    latitude: float,
    longitude: float,
    zoom: int,
    tile_size: int = 256,
) -> tuple[float, float]:
    """Convertit WGS84 (lat, lon) en pixels Web Mercator globaux."""
    validate_latlon(latitude, longitude)
    latitude = max(-WEB_MERCATOR_MAX_LAT, min(WEB_MERCATOR_MAX_LAT, latitude))
    latitude_rad = math.radians(latitude)
    world_size = tile_size * (2**zoom)

    pixel_x = (longitude + 180.0) / 360.0 * world_size
    pixel_y = (
        (1.0 - math.asinh(math.tan(latitude_rad)) / math.pi)
        / 2.0
        * world_size
    )
    return pixel_x, pixel_y


def world_pixel_to_latlon(
    pixel_x: float,
    pixel_y: float,
    zoom: int,
    tile_size: int = 256,
) -> tuple[float, float]:
    """Convertit des pixels Web Mercator globaux en WGS84 (lat, lon)."""
    world_size = tile_size * (2**zoom)
    longitude = pixel_x / world_size * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * pixel_y / world_size
    latitude = math.degrees(math.atan(math.sinh(n)))
    return latitude, longitude


def haversine_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Distance orthodromique en mètres."""
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_M * c


def cartoradio_url(latitude: float, longitude: float) -> str:
    """URL Cartoradio centrée sur une longitude/latitude."""
    validate_latlon(latitude, longitude)
    return f"https://www.cartoradio.fr/#/cartographie/lonlat/{longitude:.6f}/{latitude:.6f}"

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import math
import time

import pandas as pd
import requests
from PIL import Image

from .config import IgnConfig
from .geo import latlon_to_world_pixel, world_pixel_to_latlon


class IgnOrthophotoClient:
    """Client minimal pour les orthophotos IGN via le service WMTS GeoPF."""

    def __init__(self, config: IgnConfig, cache_dir: str | Path):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Pyl-Poil-FrHack-2026/1.0"})

    def _cache_path(self, tile_x: int, tile_y: int, zoom: int) -> Path:
        return self.cache_dir / str(zoom) / str(tile_x) / f"{tile_y}.jpg"

    def _read_cached_tile(self, path: Path) -> Image.Image | None:
        if not path.exists():
            return None
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (self.config.tile_size, self.config.tile_size):
                    return None
                return image.convert("RGB")
        except (OSError, ValueError):
            return None

    def download_tile(self, tile_x: int, tile_y: int, zoom: int | None = None) -> Image.Image:
        zoom = self.config.zoom if zoom is None else zoom
        cache_path = self._cache_path(tile_x, tile_y, zoom)
        cached = self._read_cached_tile(cache_path)
        if cached is not None:
            return cached

        params = {
            "SERVICE": "WMTS",
            "VERSION": "1.0.0",
            "REQUEST": "GetTile",
            "LAYER": self.config.layer,
            "STYLE": self.config.style,
            "FORMAT": self.config.image_format,
            "TILEMATRIXSET": self.config.tile_matrix_set,
            "TILEMATRIX": zoom,
            "TILEROW": tile_y,
            "TILECOL": tile_x,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.config.request_retries + 1):
            try:
                response = self.session.get(
                    self.config.wmts_url,
                    params=params,
                    timeout=self.config.request_timeout_s,
                )
                response.raise_for_status()
                if "image" not in response.headers.get("Content-Type", "").lower():
                    raise RuntimeError("Le service WMTS n'a pas renvoyé une image.")
                image = Image.open(BytesIO(response.content)).convert("RGB")
                if image.size != (self.config.tile_size, self.config.tile_size):
                    raise RuntimeError(f"Taille de tuile inattendue: {image.size}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(cache_path, format="JPEG", quality=self.config.jpeg_quality)
                time.sleep(self.config.request_delay_s)
                return image
            except Exception as exc:  # réseau: conserver le contexte du dernier échec
                last_error = exc
                if attempt < self.config.request_retries:
                    time.sleep(min(2.0 * attempt, 5.0))

        raise RuntimeError(
            f"Échec du téléchargement WMTS de la tuile z={zoom} x={tile_x} y={tile_y}: {last_error}"
        )

    def create_patch(self, world_left: int, world_top: int, zoom: int | None = None) -> Image.Image:
        zoom = self.config.zoom if zoom is None else zoom
        patch_size = self.config.patch_size
        tile_size = self.config.tile_size
        right = world_left + patch_size
        bottom = world_top + patch_size

        tile_x_min = math.floor(world_left / tile_size)
        tile_x_max = math.floor((right - 1) / tile_size)
        tile_y_min = math.floor(world_top / tile_size)
        tile_y_max = math.floor((bottom - 1) / tile_size)

        mosaic = Image.new(
            "RGB",
            ((tile_x_max - tile_x_min + 1) * tile_size, (tile_y_max - tile_y_min + 1) * tile_size),
        )
        for tile_x in range(tile_x_min, tile_x_max + 1):
            for tile_y in range(tile_y_min, tile_y_max + 1):
                tile = self.download_tile(tile_x, tile_y, zoom)
                mosaic.paste(
                    tile,
                    ((tile_x - tile_x_min) * tile_size, (tile_y - tile_y_min) * tile_size),
                )

        origin_x = tile_x_min * tile_size
        origin_y = tile_y_min * tile_size
        crop_left = int(round(world_left - origin_x))
        crop_top = int(round(world_top - origin_y))
        patch = mosaic.crop(
            (crop_left, crop_top, crop_left + patch_size, crop_top + patch_size)
        )
        if patch.size != (patch_size, patch_size):
            raise RuntimeError(f"Patch IGN de taille inattendue: {patch.size}")
        return patch

    def download_grid(
        self,
        center_latitude: float,
        center_longitude: float,
        output_dir: str | Path,
        zone_id: str = "zone",
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Télécharge une grille carrée de patches autour d'un centre WGS84."""
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config
        center_x, center_y = latlon_to_world_pixel(
            center_latitude, center_longitude, cfg.zoom, cfg.tile_size
        )
        total_pixels = cfg.grid_size * cfg.patch_size
        zone_left = int(round(center_x - total_pixels / 2))
        zone_top = int(round(center_y - total_pixels / 2))

        records: list[dict] = []
        for row in range(cfg.grid_size):
            for col in range(cfg.grid_size):
                left = zone_left + col * cfg.patch_size
                top = zone_top + row * cfg.patch_size
                right = left + cfg.patch_size
                bottom = top + cfg.patch_size
                filename = f"{zone_id}_r{row:02d}_c{col:02d}.jpg"
                image_path = images_dir / filename

                if overwrite or not image_path.exists():
                    patch = self.create_patch(left, top, cfg.zoom)
                    patch.save(image_path, format="JPEG", quality=cfg.jpeg_quality)

                center_lat, center_lon = world_pixel_to_latlon(
                    left + cfg.patch_size / 2,
                    top + cfg.patch_size / 2,
                    cfg.zoom,
                    cfg.tile_size,
                )
                north, west = world_pixel_to_latlon(left, top, cfg.zoom, cfg.tile_size)
                south, east = world_pixel_to_latlon(right, bottom, cfg.zoom, cfg.tile_size)
                records.append(
                    {
                        "zone_id": zone_id,
                        "image": filename,
                        "relative_image_path": str(Path("images") / filename),
                        "row": row,
                        "col": col,
                        "zoom": cfg.zoom,
                        "world_left": left,
                        "world_top": top,
                        "world_right": right,
                        "world_bottom": bottom,
                        "center_latitude": center_lat,
                        "center_longitude": center_lon,
                        "north": north,
                        "south": south,
                        "west": west,
                        "east": east,
                    }
                )

        metadata = pd.DataFrame(records)
        metadata.to_csv(output_dir / "metadata_patches.csv", index=False)
        return metadata

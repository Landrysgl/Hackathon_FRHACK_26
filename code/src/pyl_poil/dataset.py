from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd
from PIL import Image

from .config import IgnConfig
from .geo import latlon_to_world_pixel
from .imagery import IgnOrthophotoClient


def download_centered_support_images(
    supports_csv: str | Path,
    output_dir: str | Path,
    *,
    config: IgnConfig | None = None,
    cache_dir: str | Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Télécharge un patch IGN centré sur chaque support d'un CSV WGS84.

    Le CSV doit contenir `SUP_ID`, `latitude` et `longitude`. Les images sont
    nommées `support_<SUP_ID>.jpg`, comme dans les annotations livrées.
    """
    supports_csv = Path(supports_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or IgnConfig()
    cache_dir = Path(cache_dir) if cache_dir is not None else output_dir.parent / "cache_ign"

    supports = pd.read_csv(supports_csv, dtype={"SUP_ID": "string"})
    required = {"SUP_ID", "latitude", "longitude"}
    missing = required - set(supports.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans la sélection de supports: {sorted(missing)}")
    supports["latitude"] = pd.to_numeric(supports["latitude"], errors="coerce")
    supports["longitude"] = pd.to_numeric(supports["longitude"], errors="coerce")
    if supports[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Des supports ne possèdent pas de coordonnées valides.")

    client = IgnOrthophotoClient(cfg, cache_dir)
    records: list[dict] = []
    for index, row in enumerate(supports.to_dict("records"), start=1):
        support_id = str(row["SUP_ID"])
        center_x, center_y = latlon_to_world_pixel(
            float(row["latitude"]), float(row["longitude"]), cfg.zoom, cfg.tile_size
        )
        left = int(round(center_x - cfg.patch_size / 2))
        top = int(round(center_y - cfg.patch_size / 2))
        filename = f"support_{support_id}.jpg"
        image_path = output_dir / filename
        if overwrite or not image_path.exists():
            patch = client.create_patch(left, top, cfg.zoom)
            patch.save(image_path, format="JPEG", quality=cfg.jpeg_quality)
        records.append(
            {
                "SUP_ID": support_id,
                "image": filename,
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "zoom": cfg.zoom,
                "world_left": left,
                "world_top": top,
            }
        )
        if index % 25 == 0 or index == len(supports):
            print(f"Images IGN: {index}/{len(supports)}")

    metadata = pd.DataFrame(records)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    return metadata


def _bbox_to_yolo(row: pd.Series, image_size: int) -> str:
    x1, y1, x2, y2 = [float(row[c]) for c in ("x1", "y1", "x2", "y2")]
    if not (0 <= x1 < x2 <= image_size and 0 <= y1 < y2 <= image_size):
        raise ValueError(f"BBox invalide pour SUP_ID {row.get('SUP_ID')}: {(x1, y1, x2, y2)}")
    xc = ((x1 + x2) / 2.0) / image_size
    yc = ((y1 + y2) / 2.0) / image_size
    width = (x2 - x1) / image_size
    height = (y2 - y1) / image_size
    return f"0 {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}\n"


def build_human_yolo_dataset(
    images_dir: str | Path,
    train_annotation_files: list[str | Path],
    val_annotation_files: list[str | Path],
    output_dir: str | Path,
    *,
    image_size: int = 1024,
) -> dict[str, int | Path]:
    """Reconstruit le dataset mono-classe final à partir des annotations humaines.

    `visible` -> une bbox YOLO, `non_visible` -> label vide, `ambigu` -> exclu.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    def load(files: list[str | Path], split: str) -> pd.DataFrame:
        frames = []
        for file in files:
            path = Path(file)
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, dtype={"SUP_ID": "string"})
            frame["_source_file"] = path.name
            frames.append(frame)
        if not frames:
            raise ValueError(f"Aucun fichier d'annotation pour {split}.")
        df = pd.concat(frames, ignore_index=True)
        if "status" not in df or "image" not in df:
            raise ValueError(f"Annotations {split}: colonnes `status`/`image` manquantes.")
        df["status"] = df["status"].fillna("").astype(str).str.strip()
        invalid = df[~df["status"].isin(["visible", "non_visible", "ambigu"])]
        if len(invalid):
            raise ValueError(f"Annotations {split}: {len(invalid)} statut(s) invalide(s).")
        if df["SUP_ID"].duplicated().any():
            duplicated = df.loc[df["SUP_ID"].duplicated(keep=False), "SUP_ID"].tolist()
            raise ValueError(f"Annotations {split}: SUP_ID dupliqués: {duplicated[:10]}")
        return df

    train = load(train_annotation_files, "train")
    val = load(val_annotation_files, "val")
    overlap = set(train["SUP_ID"]) & set(val["SUP_ID"])
    if overlap:
        raise ValueError(f"Fuite train/val: {len(overlap)} SUP_ID commun(s).")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split, df in (("train", train), ("val", val)):
        usable = df[df["status"] != "ambigu"].copy()
        counts[f"{split}_total_reviewed"] = len(df)
        counts[f"{split}_usable"] = len(usable)
        counts[f"{split}_visible"] = int((usable["status"] == "visible").sum())
        counts[f"{split}_non_visible"] = int((usable["status"] == "non_visible").sum())
        counts[f"{split}_ambiguous_excluded"] = int((df["status"] == "ambigu").sum())

        for _, row in usable.iterrows():
            source = images_dir / str(row["image"])
            if not source.exists():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                if image.size != (image_size, image_size):
                    raise ValueError(f"Image {source.name} de taille {image.size}, attendu {(image_size, image_size)}")

            destination = output_dir / "images" / split / source.name
            shutil.copy2(source, destination)
            label_path = output_dir / "labels" / split / f"{source.stem}.txt"
            if row["status"] == "non_visible":
                label_path.write_text("", encoding="utf-8")
            else:
                if pd.isna(row.get("x1")) or pd.isna(row.get("y1")) or pd.isna(row.get("x2")) or pd.isna(row.get("y2")):
                    raise ValueError(f"BBox manquante pour l'image visible {source.name}")
                label_path.write_text(_bbox_to_yolo(row, image_size), encoding="utf-8")

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: support_radio",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {**counts, "dataset_yaml": yaml_path}

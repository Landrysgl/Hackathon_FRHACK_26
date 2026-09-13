from __future__ import annotations

from pathlib import Path
import math
import random
import shutil

import pandas as pd
from PIL import Image

DEFAULT_EXCLUDED_NATURES = {
    "Tunnel",
    "Intérieur sous-terrain",
    "Intérieur galerie",
}


def _haversine_legacy_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance utilisée par le script historique du split Niveau 1."""
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def select_level1_supports(
    supports: pd.DataFrame,
    *,
    count: int = 300,
    seed: int = 42,
    excluded_natures: set[str] | None = None,
) -> pd.DataFrame:
    """Reproduit la sélection canonique de 300 supports du Niveau 1."""
    required = {"SUP_ID", "NAT_LB_NOM", "latitude", "longitude"}
    missing = required - set(supports.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")
    if supports["SUP_ID"].astype(str).duplicated().any():
        raise ValueError("Le catalogue doit contenir un seul enregistrement par SUP_ID.")
    if count <= 0:
        raise ValueError("count doit être strictement positif.")

    excluded = DEFAULT_EXCLUDED_NATURES if excluded_natures is None else excluded_natures
    candidates = supports[~supports["NAT_LB_NOM"].isin(excluded)].copy()
    candidates["latitude"] = pd.to_numeric(candidates["latitude"], errors="coerce")
    candidates["longitude"] = pd.to_numeric(candidates["longitude"], errors="coerce")
    invalid = (
        candidates["latitude"].isna()
        | candidates["longitude"].isna()
        | ~candidates["latitude"].between(-90, 90)
        | ~candidates["longitude"].between(-180, 180)
    )
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} support(s) possèdent des coordonnées invalides.")
    if count > len(candidates):
        raise ValueError(f"Impossible de sélectionner {count} supports parmi {len(candidates)} candidats.")

    candidates = candidates.sort_values("SUP_ID").reset_index(drop=True)
    selected = (
        candidates.sample(n=count, random_state=seed)
        .sort_values("SUP_ID")
        .reset_index(drop=True)
    )
    return selected


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def create_spatial_split(
    selected: pd.DataFrame,
    *,
    grouping_distance_m: float = 200.0,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Reproduit le split spatial Niveau 1 sans groupes proches entre splits."""
    if grouping_distance_m <= 0:
        raise ValueError("grouping_distance_m doit être strictement positif.")
    ratios = ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Les ratios doivent sommer à 1.")
    required = {"SUP_ID", "latitude", "longitude"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")

    df = selected.sort_values("SUP_ID").reset_index(drop=True).copy()
    n = len(df)
    if n < len(ratios):
        raise ValueError("Pas assez de supports pour créer les splits.")
    coords = list(zip(df["latitude"].astype(float), df["longitude"].astype(float)))
    uf = _UnionFind(n)
    for i in range(n):
        lat1, lon1 = coords[i]
        for j in range(i + 1, n):
            lat2, lon2 = coords[j]
            if _haversine_legacy_m(lat1, lon1, lat2, lon2) < grouping_distance_m:
                uf.union(i, j)

    components: dict[int, list[int]] = {}
    for idx in range(n):
        components.setdefault(uf.find(idx), []).append(idx)
    groups = list(components.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    targets = {split: round(ratio * n) for split, ratio in ratios.items()}
    first = next(iter(ratios))
    targets[first] += n - sum(targets.values())
    counts = {split: 0 for split in ratios}
    assignments: dict[int, str] = {}

    for group in groups:
        remaining = {split: targets[split] - counts[split] for split in ratios}
        candidates = [split for split, left in remaining.items() if left > 0]
        if candidates:
            chosen = max(candidates, key=lambda s: (remaining[s], ratios[s]))
        else:
            chosen = min(ratios, key=lambda s: counts[s] / max(targets[s], 1))
        for idx in group:
            assignments[idx] = chosen
        counts[chosen] += len(group)

    df["split"] = [assignments[i] for i in range(n)]
    return df


def minimum_inter_split_distance(split_df: pd.DataFrame) -> float:
    """Retourne la distance minimale entre deux supports appartenant à des splits différents."""
    minimum = float("inf")
    rows = split_df.reset_index(drop=True)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows.at[i, "split"] == rows.at[j, "split"]:
                continue
            d = _haversine_legacy_m(
                float(rows.at[i, "latitude"]),
                float(rows.at[i, "longitude"]),
                float(rows.at[j, "latitude"]),
                float(rows.at[j, "longitude"]),
            )
            minimum = min(minimum, d)
    return minimum


def build_weak_yolo_dataset(
    images_dir: str | Path,
    split_manifest: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    image_size: int = 1024,
    weak_box_size_px: int = 160,
) -> Path:
    """Construit le dataset YOLO Niveau 1 avec une bbox faible centrale fixe."""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    manifest = (
        split_manifest.copy()
        if isinstance(split_manifest, pd.DataFrame)
        else pd.read_csv(split_manifest, dtype={"SUP_ID": "string"})
    )
    if weak_box_size_px <= 0 or weak_box_size_px > image_size:
        raise ValueError("weak_box_size_px doit être compris entre 1 et image_size.")
    required = {"SUP_ID", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")
    allowed_splits = {"train", "val", "test"}
    if not set(manifest["split"].astype(str)).issubset(allowed_splits):
        raise ValueError("Le manifeste contient un split inconnu.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    norm = weak_box_size_px / image_size
    label = f"0 0.50000000 0.50000000 {norm:.8f} {norm:.8f}\n"
    for row in manifest.itertuples(index=False):
        support_id = str(row.SUP_ID)
        split = str(row.split)
        source = images_dir / f"support_{support_id}.jpg"
        if not source.exists():
            raise FileNotFoundError(source)
        with Image.open(source) as image:
            if image.size != (image_size, image_size):
                raise ValueError(f"Image {source.name}: taille {image.size}, attendu {(image_size, image_size)}")
        shutil.copy2(source, output_dir / "images" / split / source.name)
        (output_dir / "labels" / split / f"support_{support_id}.txt").write_text(
            label, encoding="utf-8"
        )

    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names:",
                "  0: support_radio",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path

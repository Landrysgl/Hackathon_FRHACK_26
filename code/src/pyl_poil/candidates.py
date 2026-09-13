from __future__ import annotations

import pandas as pd

from .geo import haversine_m

REQUIRED_DETECTION_COLUMNS = {
    "detection_id", "zone_id", "confidence", "latitude", "longitude", "nearest_ANFR_distance_m"
}


def filter_by_anfr_distance(
    detections: pd.DataFrame,
    min_distance_m: float = 150.0,
) -> pd.DataFrame:
    missing = REQUIRED_DETECTION_COLUMNS - set(detections.columns)
    if missing:
        raise ValueError(f"Colonnes de détection manquantes: {sorted(missing)}")
    if min_distance_m < 0:
        raise ValueError("min_distance_m doit être positif.")
    return detections[
        pd.to_numeric(detections["nearest_ANFR_distance_m"], errors="coerce") >= min_distance_m
    ].copy().reset_index(drop=True)


def deduplicate_candidates(
    detections: pd.DataFrame,
    cluster_distance_m: float = 30.0,
) -> pd.DataFrame:
    """Fusionne les détections proches d'une même zone et garde la confiance maximale."""
    if detections.empty:
        return detections.copy()
    if cluster_distance_m <= 0:
        raise ValueError("cluster_distance_m doit être strictement positif.")

    df = detections.reset_index(drop=True).copy()
    parent = list(range(len(df)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if str(df.at[i, "zone_id"]) != str(df.at[j, "zone_id"]):
                continue
            distance = haversine_m(
                float(df.at[i, "latitude"]),
                float(df.at[i, "longitude"]),
                float(df.at[j, "latitude"]),
                float(df.at[j, "longitude"]),
            )
            if distance <= cluster_distance_m:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(df)):
        groups.setdefault(find(i), []).append(i)

    records: list[dict] = []
    for indices in groups.values():
        cluster = df.iloc[indices]
        representative_idx = cluster["confidence"].astype(float).idxmax()
        record = df.loc[representative_idx].to_dict()
        record["detections_in_cluster"] = len(cluster)
        record["cluster_detection_ids"] = ",".join(
            cluster["detection_id"].astype(str).tolist()
        )
        record["cluster_max_confidence"] = float(cluster["confidence"].astype(float).max())
        record["cluster_max_distance_anfr_m"] = float(
            cluster["nearest_ANFR_distance_m"].astype(float).max()
        )
        records.append(record)

    out = pd.DataFrame(records)
    out = out.sort_values(["zone_id", "confidence"], ascending=[True, False]).reset_index(drop=True)
    out["candidate_id"] = [f"C{i:03d}" for i in range(1, len(out) + 1)]
    return out


def build_candidates(
    detections: pd.DataFrame,
    min_distance_m: float = 150.0,
    cluster_distance_m: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    far = filter_by_anfr_distance(detections, min_distance_m=min_distance_m)
    unique = deduplicate_candidates(far, cluster_distance_m=cluster_distance_m)
    return far, unique

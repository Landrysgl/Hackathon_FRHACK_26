#!/usr/bin/env python3
"""Vérifie l'intégrité des résultats de référence livrés avec Pyl-Poil."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd

from pyl_poil.candidates import build_candidates
from pyl_poil.level1 import create_spatial_split, minimum_inter_split_distance, select_level1_supports
from pyl_poil.zones import select_contrasting_zones

ROOT = Path(__file__).resolve().parents[1]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[OK] {message}")


def main() -> None:
    supports = pd.read_csv(ROOT / "data/reference/supports_yvelines.csv", dtype={"SUP_ID": "string"})
    check(len(supports) == 1464, "catalogue ANFR Yvelines: 1 464 supports")

    selected = select_level1_supports(supports)
    selected_ref = pd.read_csv(ROOT / "data/reference/supports_selected_300.csv", dtype={"SUP_ID": "string"})
    check(selected["SUP_ID"].tolist() == selected_ref["SUP_ID"].tolist(), "Niveau 1: sélection exacte des 300 supports")
    split = create_spatial_split(selected)
    split_ref = pd.read_csv(ROOT / "data/reference/spatial_split_300.csv", dtype={"SUP_ID": "string"})
    check(split["split"].tolist() == split_ref["split"].tolist(), "Niveau 1: split spatial exact 210 / 45 / 45")
    check(minimum_inter_split_distance(split) >= 200.0, "Niveau 1: aucune fuite spatiale à moins de 200 m entre splits")
    weak = pd.read_csv(ROOT / "data/reference/level1_weak_annotations.csv", dtype={"SUP_ID": "string"})
    check(len(weak) == 300, "Niveau 1: 300 annotations faibles matérialisées")
    check(set(weak["SUP_ID"]) == set(selected["SUP_ID"]), "Niveau 1: annotations faibles alignées sur la sélection")

    zones = select_contrasting_zones(supports)
    zones_ref = pd.read_csv(ROOT / "data/level3/zones_selected.csv", dtype={"SUP_ID": "string"})
    zones["SUP_ID"] = zones["SUP_ID"].astype(str)
    check(zones["SUP_ID"].tolist() == zones_ref["SUP_ID"].tolist(), "Niveau 3: sélection des trois zones reproduite")

    manifests = json.loads((ROOT / "models/models_manifest.json").read_text(encoding="utf-8"))["models"]
    roles = {entry["role"] for entry in manifests}
    check(roles == {"niveau1_weak_baseline", "final"}, "deux modèles documentés: baseline N1 + final")
    for manifest in manifests:
        model = ROOT / "models" / manifest["file"]
        check(model.exists(), f"modèle {manifest['role']} présent")
        check(
            model.stat().st_size == int(manifest["size_bytes"]),
            f"taille du modèle {manifest['role']} conforme au manifeste",
        )
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        check(
            digest == manifest["sha256"],
            f"SHA-256 du modèle {manifest['role']} conforme",
        )

    detections = pd.read_csv(ROOT / "data/level3/detections_multizone.csv")
    far, unique = build_candidates(detections, 150.0, 30.0)
    check(len(detections) == 174, "Niveau 3: 174 détections brutes")
    check(len(far) == 53, "Niveau 3: 53 détections à au moins 150 m")
    check(len(unique) == 46, "Niveau 3: 46 candidats après déduplication")

    reviewed = pd.read_csv(ROOT / "data/level3/reviewed_candidates.csv")
    statuses = reviewed["review_status"].fillna("").value_counts().to_dict()
    check(statuses.get("plausible", 0) == 27, "revue humaine: 27 plausibles")
    check(statuses.get("faux_positif", 0) == 17, "revue humaine: 17 faux positifs")
    check(statuses.get("incertain", 0) == 2, "revue humaine: 2 incertains")

    final = pd.read_csv(ROOT / "results/level3/final_candidates.csv")
    ids = final["unique_candidate_id"].tolist()
    check(ids == ["D003", "D006", "D007"], "sélection finale: D003, D006, D007")
    for candidate_id in ids:
        check(
            (ROOT / "results/level3/final_candidates_images" / f"{candidate_id}.jpg").exists(),
            f"image finale {candidate_id} présente",
        )

    print("\nTous les résultats de référence sont cohérents.")


if __name__ == "__main__":
    main()

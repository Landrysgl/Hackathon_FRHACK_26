#!/usr/bin/env python3
"""Relance l'analyse Pyl-Poil sur les trois zones canoniques du Niveau 3.

Cette commande retélécharge les orthophotos IGN actuelles. Les résultats peuvent donc
légèrement différer de ceux du hackathon si l'imagerie du service a été mise à jour.
Pour reproduire exactement le filtrage/déduplication historique à partir des 174
prédictions sauvegardées, utiliser ``rebuild_level3_candidates.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pyl_poil.config import load_config
from pyl_poil.pipeline import analyze_area


def main() -> None:
    parser = argparse.ArgumentParser(description="Relance le pipeline sur les trois zones Niveau 3.")
    parser.add_argument("--zones", type=Path, default=Path("data/level3/zones_selected.csv"))
    parser.add_argument("--supports", type=Path, default=Path("data/reference/supports_yvelines.csv"))
    parser.add_argument("--weights", type=Path, default=Path("models/pyl_poil_final_yolov8n.pt"))
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/level3_multizone"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    zones = pd.read_csv(args.zones)
    required = {"zone_id", "latitude", "longitude"}
    missing = required - set(zones.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier de zones: {sorted(missing)}")

    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    detection_frames: list[pd.DataFrame] = []
    far_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []

    for _, zone in zones.iterrows():
        zone_id = str(zone["zone_id"])
        zone_dir = args.output / zone_id
        print(f"\n=== {zone_id} ===")
        result = analyze_area(
            float(zone["latitude"]),
            float(zone["longitude"]),
            args.supports,
            args.weights,
            zone_dir,
            config,
            zone_id=zone_id,
            device=int(args.device) if str(args.device).isdigit() else args.device,
        )
        summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
        summary["zone_id"] = zone_id
        summaries.append(summary)
        detection_frames.append(pd.read_csv(result["detections"]))
        far_frames.append(pd.read_csv(result["far_detections"]))
        candidate_frames.append(pd.read_csv(result["candidates"]))

    summary_df = pd.DataFrame(summaries)
    summary_path = args.output / "resume_multizone.csv"
    summary_df.to_csv(summary_path, index=False)
    if detection_frames:
        pd.concat(detection_frames, ignore_index=True).to_csv(
            args.output / "detections_multizone.csv", index=False
        )
    if far_frames:
        pd.concat(far_frames, ignore_index=True).to_csv(
            args.output / "detections_ge150m_multizone.csv", index=False
        )
    if candidate_frames:
        pd.concat(candidate_frames, ignore_index=True).to_csv(
            args.output / "candidats_uniques_multizone.csv", index=False
        )
    print(f"\nRésumé : {summary_path}")
    print("Détections consolidées et candidats écrits dans :", args.output)


if __name__ == "__main__":
    main()

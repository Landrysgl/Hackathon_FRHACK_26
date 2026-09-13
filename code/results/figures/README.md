# Figures clés du projet

Ces éléments sont destinés au **PDF de résultats**, au rapport synthétique et au PowerPoint. Ils sont tous issus des données ou résultats réellement conservés dans le dépôt.

| Fichier | Ce qu'il montre | Utilisation conseillée |
|---|---|---|
| `overview/00_level1_weak_annotation_method.png` | schéma exact de la bbox faible centrale 160×160 px | expliquer le Niveau 1 |
| `overview/01_yvelines_support_natures.png` | diversité des supports ANFR dans les Yvelines | dataset / justification de la zone |
| `overview/02_annotation_coverage.png` | volume de revue humaine et statuts visible / non visible / ambigu | qualité du dataset |
| `overview/03_visible_support_types_train.png` | types visuels des 144 supports visibles du train | Niveau 2 / reconnaissance des supports |
| `overview/04_level2_experiments_initial_test_map50.png` | comparaison historique d'expériences sur le même petit test | démarche expérimentale |
| `overview/05_level2_final_holdout_metrics.png` | métriques du modèle final sur le holdout indépendant | performance finale |
| `overview/06_final_training_curve.png` | courbes mAP du run final | entraînement / convergence |
| `overview/07_anfr_distance_calibration.png` | dispersion coordonnée ANFR ↔ centre visuel et seuil 150 m | justification du filtre Niveau 3 |
| `overview/08_level3_candidate_funnel.png` | 174 → 53 → 46 → 27 | pipeline de recherche de candidats |
| `overview/09_level3_review_by_zone.png` | plausibles / FP / incertains selon la densité | analyse qualitative / limites |
| `overview/10_level3_review_examples.jpg` | un plausible, un faux positif, un incertain | analyse visuelle des erreurs |
| `level2/hard_negative_candidates_montage.png` | exemples réels de faux positifs étudiés comme hard negatives | compréhension des erreurs Niveau 2 |
| `../level3/final_candidates_montage.jpg` | D003, D006, D007 | trois cas finaux à présenter |
| `../level3/carte_candidats_finale.html` | carte interactive avec couches et liens Cartoradio | démonstration / cartographie |

Les figures `overview/` sont régénérées par `scripts/generate_result_figures.py`. La carte HTML est régénérée par `scripts/generate_reference_map.py`.

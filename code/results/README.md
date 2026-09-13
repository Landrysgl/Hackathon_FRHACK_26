# Résultats livrés avec Pyl-Poil

Ce dossier contient les **preuves quantitatives et visuelles** du travail réalisé pour le Challenge 4. Les gros jeux d'orthophotos et les caches IGN ne sont pas versionnés, mais les annotations, métriques, historiques d'entraînement, détections géolocalisées et résultats finaux nécessaires à l'audit sont conservés.

## `metrics/` — chiffres consolidés

- `annotation_coverage.csv` : couverture des revues humaines Niveau 1 / Niveau 2 ;
- `experiment_comparison.csv` : expériences comparées sur le test manuel historique et résultat du holdout final ;
- `training_runs_summary.csv` : meilleur point enregistré de chaque run d'entraînement ;
- `level1_manual_test.json` : métriques du baseline faible ;
- `level2_holdout_final.json` : métriques du modèle final sur le holdout indépendant ;
- `anfr_distance_calibration.json` : statistiques justifiant le seuil de recherche à 150 m ;
- `level3_summary.json` : synthèse globale du Niveau 3.

## `training_history/` — historique utile des entraînements

Les `results.csv` Ultralytics et paramètres disponibles des principales variantes sont conservés. Les checkpoints intermédiaires lourds ne le sont pas. Le modèle faible Niveau 1 et le modèle final Niveau 2 sont livrés séparément dans `models/`.

## `evaluation/` — évaluations détaillées

- `level1/` : détails et métriques de la première validation manuelle ;
- `level2_holdout/` : détails et métriques du holdout final indépendant.

## `error_analysis/` — analyse des erreurs

Contient les résumés de faux positifs et la sélection de hard negatives utilisée pendant les itérations Niveau 2. Une mosaïque réelle de ces faux positifs est dans `figures/level2/`.

## `level3/` — recherche de candidats

Contient la carte HTML interactive, les 46 candidats revus, les 27 plausibles, les fichiers GeoJSON/CSV et la sélection finale D003 / D006 / D007 avec leurs images.

## `figures/` — illustrations prêtes pour le rapport / PowerPoint

`figures/overview/` contient onze figures régénérables avec :

```bash
python scripts/generate_result_figures.py
```

Le détail et l'usage recommandé de chaque figure sont documentés dans `figures/README.md`.

> Une catégorie `plausible` signifie qu'une structure observée mérite un examen. Elle ne constitue pas une preuve de site radioélectrique non déclaré.

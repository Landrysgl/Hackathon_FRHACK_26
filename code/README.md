# Pyl-Poil — FrHack! 2026 · Challenge 4 ANFR × ISEP

**Pyl-Poil** est l'outil Python développé par l'équipe pour repérer des **structures susceptibles de supporter des installations radioélectriques** dans des orthophotos aériennes IGN, géolocaliser les détections, les comparer au référentiel ANFR et faire émerger des candidats à examiner.

Le dépôt est organisé comme un outil réutilisable : paramètres centralisés, scripts d'installation/utilisation, deux modèles livrés, annotations humaines, résultats quantitatifs, figures clés, tests et carte HTML interactive.

> **Précaution essentielle** — une détection sans correspondance ANFR proche est un **candidat visuel**, jamais une preuve automatique de site radioélectrique non déclaré. Elle peut être un faux positif, un support sans installation radio, un décalage géographique/temporel, une installation récente/démantelée ou une erreur de classification.

## Résultats en un coup d'œil

| Étape | Résultat |
|---|---:|
| Catalogue ANFR Yvelines | **1 464** supports |
| Dataset sélectionné | **300** supports, split spatial 210 / 45 / 45 |
| Niveau 1 — mAP50 sur test humain initial | **0,0126** |
| Niveau 2 — train revu manuellement | **210** images, 165 utilisables |
| Niveau 2 — validation revue | **45** images, 35 utilisables |
| Niveau 2 — holdout final indépendant | **25** images, 23 utilisables |
| Modèle final — mAP50 holdout | **0,2495** |
| Modèle final — mAP50-95 holdout | **0,1138** |
| Niveau 3 — détections brutes | **174** |
| Après filtre ≥150 m | **53** |
| Après déduplication 30 m | **46** candidats uniques |
| Revue humaine | **27 plausibles**, 17 FP, 2 incertains |
| Cas retenus | **D003, D006, D007** |

Les chiffres consolidés sont dans `results/metrics/`, les illustrations dans `results/figures/` et la carte dans `results/level3/carte_candidats_finale.html`.

## Installation

Python **3.10+** est recommandé.

```bash
cd code
python -m venv .venv
source .venv/bin/activate          # Linux / Onyxia
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/check_installation.py
```

Sous Windows PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

L'analyse d'une nouvelle zone nécessite un accès Internet à l'API WMTS IGN/GeoPF.

## Utiliser Pyl-Poil sur une zone

Exemple sur le centre de la zone dense étudiée au Niveau 3 :

```bash
python scripts/analyze_area.py \
  --lat 48.7975 \
  --lon 2.1300 \
  --supports data/reference/supports_yvelines.csv \
  --weights models/pyl_poil_final_yolov8n.pt \
  --config config/default.yaml \
  --output outputs/demo_dense
```

Après `pip install -e .`, la même analyse peut être lancée avec la commande :

```bash
pyl-poil \
  --lat 48.7975 \
  --lon 2.1300 \
  --supports data/reference/supports_yvelines.csv \
  --weights models/pyl_poil_final_yolov8n.pt \
  --output outputs/demo_dense
```

Sorties principales :

```text
outputs/demo_dense/
├── imagery/                         # orthophotos + métadonnées géographiques
├── detections.csv                   # sorties YOLO géolocalisées
├── detections_hors_correspondance_anfr.csv
├── candidats_uniques.csv            # candidats après déduplication
├── carte_candidats.html             # HTML standalone interactif
└── resume.json
```

Chaque candidat reçoit aussi une URL Cartoradio centrée sur sa longitude/latitude.

## Reproduire la démarche du hackathon

Les scripts sont nommés selon leur fonction, sans dépendre des anciens numéros d'expérimentation :

| Objectif | Commande |
|---|---|
| reconstruire le catalogue ANFR 78 | `python scripts/prepare_anfr_reference.py ...` |
| reproduire la sélection 300 + split spatial | `python scripts/select_level1_supports.py` |
| retélécharger les 300 images | `python scripts/download_training_imagery.py` |
| reconstruire le dataset faible Niveau 1 | `python scripts/build_level1_weak_dataset.py` |
| reconstruire le dataset humain final | `python scripts/build_training_dataset.py` |
| entraîner un modèle | `python scripts/train_model.py --data ...` |
| évaluer un modèle YOLO | `python scripts/evaluate_model.py --weights ... --data ...` |
| reproduire la sélection des 3 zones | `python scripts/select_level3_zones.py` |
| recalibrer l'écart ANFR / centre visuel | `python scripts/calibrate_anfr_distance.py` |
| relancer les 3 zones avec l'imagerie IGN actuelle | `python scripts/reproduce_level3_multizone.py` |
| rejouer exactement filtre 150 m + déduplication sur les 174 détections sauvegardées | `python scripts/rebuild_level3_candidates.py` |
| régénérer la carte finale | `python scripts/generate_reference_map.py` |
| régénérer les figures du rapport | `python scripts/generate_result_figures.py` |
| vérifier les résultats livrés | `python scripts/verify_reference_results.py` |

### Modèles livrés

- `models/pyl_poil_level1_weak_baseline.pt` — baseline Niveau 1 issue des annotations faibles ;
- `models/pyl_poil_final_yolov8n.pt` — modèle final Niveau 2, YOLOv8n initialisé COCO, sans rotation forcée.

Leurs SHA-256 sont dans `models/models_manifest.json`.

## Choix techniques importants

Les paramètres communs sont dans `config/default.yaml` : orthophotos IGN `ORTHOIMAGERY.ORTHOPHOTOS`, zoom 19, patches 1024×1024, grille 5×5, confiance YOLO 0,05, NMS IoU 0,50, filtre ANFR 150 m et déduplication 30 m.

Le seuil de **150 m** a été calibré sur **175 supports visibles** du train + validation : médiane 11,73 m, P95 30,71 m, maximum 50,49 m. Il est volontairement conservateur et n'a aucune valeur réglementaire. La figure correspondante est `results/figures/overview/07_anfr_distance_calibration.png`.

## Aire géographique

L'étude porte sur les **Yvelines (78)**. Le catalogue préparé contient 1 464 supports, avec une diversité suffisante de bâtiments, pylônes, mâts et autres natures tout en restant compatible avec le temps de calcul du hackathon.

Le Niveau 3 compare trois contextes automatiquement sélectionnés et éloignés :

- dense : 16 supports à 500 m ;
- intermédiaire : 3 supports à 500 m ;
- peu dense : 1 support à 500 m.

Cette comparaison met en évidence une limite importante : le taux de candidats visuellement plausibles après filtre/déduplication est 95,5 % dans la zone dense, 40,0 % dans l'intermédiaire et 21,1 % dans la zone peu dense.

## Résultats et illustrations

Les éléments directement réutilisables dans le PDF ou le PowerPoint sont indexés dans `results/figures/README.md`. On y trouve notamment : diversité des supports, couverture des annotations, types visuels, comparaison d'expériences, métriques du holdout, courbe d'entraînement, calibration 150 m, funnel Niveau 3, analyse par zone, hard negatives et montage des trois candidats finaux.

Les trois candidats retenus sont **D003 (mât), D006 (bâtiment) et D007 (bâtiment)**. Images, coordonnées, confiance, distance au support ANFR le plus proche et liens Cartoradio sont dans `results/level3/`.

## Structure du dossier

```text
code/
├── README.md
├── requirements.txt
├── pyproject.toml
├── config/                    # paramètres communs
├── src/pyl_poil/              # bibliothèque de l'outil
├── scripts/                   # commandes simples et reproductibles
├── tests/                     # tests de cohérence
├── models/                    # baseline N1 + modèle final N2
├── data/
│   ├── reference/             # catalogue ANFR, sélection et split
│   ├── annotations/           # annotations humaines
│   └── level3/                # détections et données de reproduction N3
├── results/
│   ├── metrics/
│   ├── training_history/
│   ├── evaluation/
│   ├── error_analysis/
│   ├── figures/
│   └── level3/
└── docs/
    ├── CHALLENGE4_COVERAGE.md
    ├── METHODOLOGY.md
    ├── RESULTS.md
    ├── VALIDATION.md
    └── TEST_REPORT.md
```

Les centaines d'orthophotos, caches WMTS et checkpoints intermédiaires ne sont volontairement pas versionnés : ils alourdissent le dépôt et sont reconstructibles.

## Sources

- **ANFR** — référentiel ouvert des supports/installations radioélectriques ;
- **IGN / GeoPF** — orthophotos par WMTS `https://data.geopf.fr/wmts` ;
- **Cartoradio** — vérification cartographique via une URL `.../lonlat/<longitude>/<latitude>` ;
- **Ultralytics YOLOv8** — détection d'objets.

## Tests et validation

```bash
python -m compileall -q src scripts tests
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/verify_reference_results.py
```

Le rapport exact des contrôles exécutés se trouve dans `docs/TEST_REPORT.md`.

Pour vérifier explicitement comment le dossier répond à chaque attente de la fiche du Challenge 4, lire **`docs/CHALLENGE4_COVERAGE.md`**.

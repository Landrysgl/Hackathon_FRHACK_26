# Méthodologie technique

## Objectif

Le projet répond au Challenge 4 ANFR × ISEP : construire un outil capable de repérer des supports radioélectriques sur des images aériennes, d'évaluer quantitativement la détection et, à un niveau avancé, de faire émerger des candidats éloignés des correspondances évidentes avec la base ANFR.

La démarche a été menée progressivement en trois niveaux afin de séparer les problèmes de données, d'apprentissage et de recherche de candidats.

## Niveau 1 — labels faibles à partir de l'ANFR

La première étape part des coordonnées de supports ANFR dans les Yvelines. Après déduplication, 300 supports physiques ont été sélectionnés de façon reproductible. Les supports explicitement souterrains ont été écartés avant téléchargement des orthophotos.

Chaque orthophoto IGN fait 1024 × 1024 px et est centrée sur la coordonnée ANFR. Comme une coordonnée administrative n'est pas une vérité terrain visuelle exacte, le premier dataset utilise une **annotation faible** : une boîte fixe centrée dans l'image.

Le découpage est spatial afin de limiter la fuite géographique :

- train : 210 images ;
- validation : 45 images ;
- test : 45 images.

Un premier sous-ensemble de 20 images du test a été annoté humainement : 10 `visible`, 3 `non_visible`, 7 `ambigu`. Le baseline faible atteint une mAP50 de 0,0126 sur ces annotations humaines. Cette faiblesse a motivé le Niveau 2.

## Niveau 2 — annotations humaines et comparaison d'expériences

Les 210 images de train ont été revues manuellement :

- 144 supports visibles ;
- 21 non visibles ;
- 45 ambigus.

Les 45 images de validation ont également été revues :

- 31 visibles ;
- 4 non visibles ;
- 10 ambiguës.

Les images ambiguës sont exclues des jeux utilisés pour l'évaluation. Les supports visibles ont une bbox manuelle et un type visuel lorsque celui-ci a pu être identifié.

Plusieurs stratégies ont été testées : entraînement uniquement humain, fine-tuning depuis le Niveau 1, hard negatives, rotation aérienne, mono-classe / multi-classe, initialisation Niveau 1 ou COCO. Les résultats sont conservés dans `results/metrics/experiment_comparison.csv` et `training_runs_summary.csv`.

L'expérience finale retenue est une initialisation **YOLOv8n pré-entraînée COCO**, entraînée sur le dataset humain renforcé, sans rotation forcée. Le choix final a été évalué une seule fois sur un holdout frais de 25 images : 21 visibles, 2 non visibles et 2 ambiguës exclues.

Holdout final :

- mAP50 : 0,2495 ;
- mAP50-95 : 0,1138 ;
- précision Ultralytics : 0,3313 ;
- rappel Ultralytics : 0,4286 ;
- à `conf=0.05` : 13 TP, 111 FP, 8 FN, soit précision 0,1048 et rappel 0,6190.

Le seuil très permissif de 0,05 est volontaire pour la recherche de candidats : il privilégie le rappel et déplace une partie du travail vers le filtrage géographique et la revue humaine.

## Calibration de la distance ANFR

Le Niveau 3 ne doit pas considérer toute petite différence entre le centre d'une bbox et la coordonnée ANFR comme une anomalie. La dispersion a donc été mesurée sur les **175 supports visibles annotés du train + validation**.

Distance du centre visuel au support ANFR le plus proche :

- moyenne : 13,55 m ;
- médiane : 11,73 m ;
- P95 : 30,71 m ;
- P97,5 : 40,02 m ;
- P99 : 47,16 m ;
- maximum : 50,49 m.

Le seuil de recherche avancée a été fixé à **150 m**, soit environ trois fois le maximum observé sur cette calibration. Il s'agit d'un filtre conservateur, pas d'une frontière réglementaire.

## Niveau 3 — recherche multizone

Trois centres spatialement séparés ont été sélectionnés selon la densité locale de supports ANFR dans un rayon de 500 m :

| Zone | Support centre | Supports à 500 m | Patches |
|---|---:|---:|---:|
| dense | 2274825 | 16 | 25 |
| intermédiaire | 1122334 | 3 | 25 |
| peu dense | 458152 | 1 | 25 |

Chaque zone est couverte par une grille 5 × 5 de patches de 1024 px au zoom WMTS 19. Les centres sont séparés de plusieurs dizaines de kilomètres afin de comparer des contextes différents.

Le pipeline est :

1. téléchargement IGN / GeoPF ;
2. détection YOLO à `conf=0.05` et NMS IoU 0,50 ;
3. conversion du centre de bbox en latitude/longitude ;
4. recherche du support ANFR le plus proche ;
5. conservation des détections à au moins 150 m ;
6. déduplication spatiale dans un rayon de 30 m ;
7. revue humaine des candidats ;
8. typage visuel des plausibles ;
9. cartographie interactive et sélection de trois cas à présenter.

Résultat géométrique reproductible : **174 détections → 53 à ≥150 m → 46 candidats uniques**.

## Revue humaine et sélection finale

Parmi les 46 candidats uniques :

- 27 ont été classés `plausible` ;
- 17 `faux_positif` ;
- 2 `incertain`.

Parmi les 27 plausibles : 23 ont été typés `batiment`, 3 `indetermine` et 1 `mat`.

La sélection finale contient trois candidats déjà revus comme plausibles et avec un type identifié :

- D003 — mât ;
- D006 — bâtiment ;
- D007 — bâtiment.

Ils ne sont jamais présentés comme des sites non déclarés confirmés.

## Reproductibilité

Le code final évite les chemins absolus Onyxia. Les paramètres sont centralisés dans `config/default.yaml`. Les orthophotos lourdes ne sont pas versionnées : elles sont régénérées via WMTS. Le modèle final, les annotations humaines, les métriques et les détections intermédiaires du Niveau 3 sont livrés afin de permettre un contrôle du résultat sans relancer l'entraînement.

# Couverture de la fiche du Challenge 4

Ce document relie explicitement les attentes du Challenge 4 aux fichiers livrés avec **Pyl-Poil**.

## Niveau 1 — baseline fonctionnelle

| Attente | Réalisation / preuve |
|---|---|
| récupérer les coordonnées ANFR | `src/pyl_poil/anfr.py`, `scripts/prepare_anfr_reference.py`, `data/reference/supports_yvelines.csv` |
| télécharger automatiquement les images aériennes | `src/pyl_poil/imagery.py`, `scripts/download_training_imagery.py` |
| définir une zone d'intérêt | patches 1024×1024 au zoom WMTS 19, paramètres dans `config/default.yaml` |
| produire des annotations faibles | `data/reference/level1_weak_annotations.csv`, `src/pyl_poil/level1.py`, `scripts/build_level1_weak_dataset.py` |
| séparer train / validation / test | `scripts/select_level1_supports.py` + `data/reference/spatial_split_300.csv` : 210 / 45 / 45, split spatial reproductible |
| entraîner un premier détecteur | checkpoint Niveau 1 livré dans `models/` |
| évaluer précision, rappel, mAP, FP/FN | `results/metrics/level1_manual_test.json`, `results/evaluation/level1/` |

## Niveau 2 — meilleure vérité terrain et supports

| Attente | Réalisation / preuve |
|---|---|
| correction manuelle | `data/annotations/level2_*.csv` : 210 train + 45 validation + 25 holdout final revus |
| meilleure localisation | bbox manuelles pour les supports `visible` |
| exemples négatifs | lignes `non_visible`, analyse dans `results/error_analysis/` |
| recherche des faux positifs récurrents | CSV d'analyse + `results/figures/level2/hard_negative_candidates_montage.png` |
| augmentation / résolution | historique des essais dans `results/training_history/` ; résolution 1024 px |
| nature des supports | colonne `classe_niveau2` ; distribution dans `results/figures/overview/03_visible_support_types_train.png` |
| comparaison mono / multi-classes | runs `niveau2_compare_mono_r180` et `niveau2_compare_multi_r180` conservés dans `results/training_history/` |
| performance finale sur données manuelles indépendantes | `results/metrics/level2_holdout_final.json`, mAP50 = 0,2495 |

Le holdout final n'a pas servi au choix des hyperparamètres.

## Niveau 3 — vision aérienne avancée

La fiche propose plusieurs pistes avancées (pré-entraînement aérien, OBB, **ou** recherche de candidats sans correspondance évidente dans le référentiel). Pyl-Poil a choisi la troisième piste : **recherche multizone de candidats éloignés de l'ANFR**.

| Étape | Réalisation / preuve |
|---|---|
| calibration du décalage ANFR / objet visuel | `src/pyl_poil/calibration.py`, `scripts/calibrate_anfr_distance.py`, 175 supports visibles |
| seuil conservateur | 150 m, contre max 50,49 m observé dans la calibration |
| application à des zones plus larges | 3 zones × 25 patches, `data/level3/zones_selected.csv` |
| contextes contrastés | dense = 16 supports/500m ; intermédiaire = 3 ; peu dense = 1 |
| sélection reproductible des zones | `src/pyl_poil/zones.py`, `scripts/select_level3_zones.py` |
| détection + géolocalisation | `src/pyl_poil/detector.py` |
| filtrage / fusion des détections | `src/pyl_poil/candidates.py` |
| revue critique des candidats | `data/level3/reviewed_candidates.csv`, 27 plausibles / 17 FP / 2 incertains |
| jusqu'à trois cas intéressants | D003, D006, D007 dans `results/level3/` |
| cartographie interactive | `results/level3/carte_candidats_finale.html` |

Aucun candidat n'est présenté comme un site « non déclaré » confirmé. Le résultat est un **outil de priorisation d'observations**, à confronter notamment à Cartoradio, à la date de l'imagerie et au contexte du support.

## Aire géographique et justification

La zone d'étude est le **département des Yvelines (78)**. Elle contient 1 464 supports ANFR dans le catalogue préparé. Ce choix offrait à la fois :

- suffisamment de supports pour construire un jeu de quelques centaines d'exemples ;
- une diversité de natures de supports et de contextes urbains/ruraux ;
- un volume d'orthophotos compatible avec le temps et les ressources GPU du hackathon ;
- une aire unique permettant de maîtriser la cohérence géographique des splits.

Le Niveau 3 utilise trois zones séparées de plusieurs dizaines de kilomètres afin de ne pas conclure à partir d'un seul quartier très dense.

## Livrables demandés par la fiche

| Livrable | Emplacement dans `code/` |
|---|---|
| dataset construit + méthode | `data/`, `docs/METHODOLOGY.md` |
| annotations faibles et manuelles | annotations manuelles dans `data/annotations/`, méthode faible documentée |
| pipeline reproductible | `src/pyl_poil/`, `scripts/`, `config/default.yaml` |
| modèles entraînés | `models/` : baseline Niveau 1 + modèle final Niveau 2 |
| résultats quantitatifs | `results/metrics/`, `results/evaluation/`, `results/training_history/` |
| visualisation cartographique | `results/level3/carte_candidats_finale.html` |
| analyse FP/FN | `results/error_analysis/`, figures de hard negatives, détails d'évaluation |
| rapport technique synthétique | matière technique consolidée dans `docs/METHODOLOGY.md` + `docs/RESULTS.md` ; le PDF d'équipe reste un livrable séparé |
| jusqu'à 3 candidats avancés | D003 / D006 / D007 avec images et analyse critique |

## Correspondance avec les critères d'évaluation

- **30 pts — qualité/robustesse du dataset** : split spatial, 300 supports, 280 revues manuelles N2/holdout, statuts explicites, types visuels, négatifs, calibration.
- **25 pts — mAP50 sur validation manuelle** : métriques du holdout final et détails conservés.
- **20 pts — pertinence des détections candidates** : filtre calibré, déduplication, revue humaine et trois cas documentés.
- **15 pts — cartographie** : HTML standalone avec couches et liens Cartoradio.
- **10 pts — clarté / enjeux opérationnels et réglementaires** : avertissements systématiques sur la nature non probante d'un candidat, limites et démarche reproductible documentées.

## Ce qui n'a volontairement pas été retenu

Les essais DOTA / OBB n'ont pas été implémentés : le Niveau 3 a été consacré à la piste de recherche d'anomalies/candidats proposée par la fiche. Les anciens scripts exploratoires numérotés, caches, orthophotos dupliquées et checkpoints intermédiaires ont été retirés du livrable actif ; leurs résultats utiles sont conservés sous forme de métriques et d'historiques.

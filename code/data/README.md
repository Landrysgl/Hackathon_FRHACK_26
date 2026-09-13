# Données livrées avec le code

Le dépôt ne versionne volontairement pas les centaines d'orthophotos IGN ni les caches WMTS : ils sont lourds et reconstructibles depuis le service GeoPF.

## `reference/`

`supports_yvelines.csv` contient les 1 464 supports ANFR géolocalisés utilisés comme catalogue de référence pour la zone d'étude des Yvelines.

`level1_weak_annotations.csv` matérialise les **300 annotations faibles** du Niveau 1 : bbox centrale fixe 160×160 px (0,15625 × 0,15625 normalisé) et split associé.

## `annotations/`

Les CSV conservent les annotations humaines qui servent à documenter et reproduire l'évaluation :

- `level1_manual_test.csv` : premier test humain Niveau 1 ;
- `level2_train_batch1.csv` et `level2_train_batch2.csv` : 210 images de train revues ;
- `level2_validation.csv` : 45 images de validation ;
- `level2_holdout_final.csv` : 25 images réservées au holdout final.

Les statuts utilisés sont `visible`, `non_visible` et `ambigu`. Les lignes `ambigu` sont exclues des métriques de détection.

## `level3/`

Ces fichiers permettent de vérifier le pipeline géographique sans relancer l'inférence :

- sélection des trois zones ;
- métadonnées géoréférencées des 75 patches ;
- 174 détections brutes géolocalisées ;
- candidats uniques après filtre/déduplication et revue humaine ;
- calibration de l'écart entre coordonnée ANFR et centre visuel du support.

Les images aériennes ne sont pas redistribuées dans ce dossier. Le script d'analyse les télécharge à la demande depuis IGN/GeoPF.

## Reconstruire le catalogue ANFR

Si les exports officiels `SUP_SUPPORT.txt` et `SUP_NATURE.txt` sont disponibles, le catalogue Yvelines peut être reconstruit avec :

```bash
python scripts/prepare_anfr_reference.py \
  --supports /chemin/SUP_SUPPORT.txt \
  --natures /chemin/SUP_NATURE.txt \
  --department 78 \
  --output data/reference/supports_yvelines.csv
```

## Reproduire les images et le dataset final

`reference/supports_selected_300.csv` fige les 300 supports sélectionnés pendant le hackathon et `reference/spatial_split_300.csv` conserve le split spatial 210/45/45.

Pour retélécharger les images à partir d'IGN/GeoPF :

```bash
python scripts/download_training_imagery.py
```

Puis, après téléchargement, reconstruire le dataset mono-classe humain utilisé pour le modèle final :

```bash
python scripts/build_training_dataset.py
```

La construction exclut les annotations `ambigu`, produit des labels vides pour `non_visible` et convertit les bbox `visible` au format YOLO.

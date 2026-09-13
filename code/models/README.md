# Modèles livrés

Deux checkpoints sont conservés pour documenter la progression du Challenge 4.

## `pyl_poil_level1_weak_baseline.pt`

Baseline du **Niveau 1**, entraînée à partir des annotations faibles dérivées des coordonnées ANFR. Sa faible performance sur le premier test humain (mAP50 ≈ 0,0126) justifie le passage à des annotations manuelles.

## `pyl_poil_final_yolov8n.pt`

Modèle final retenu pour **Pyl-Poil** :

- architecture : YOLOv8n pré-entraîné COCO ;
- résolution : 1024 px ;
- entraînement final : 50 époques maximum, batch 8, seed 42 ;
- aucune rotation forcée (`degrees=0`) ;
- holdout final indépendant : mAP50 = 0,2495, mAP50-95 = 0,1138.

`models_manifest.json` contient la taille et le SHA-256 des deux modèles. `model_manifest.json` est conservé comme manifeste de compatibilité pour le modèle final.

Le modèle détecte des **structures/supports visuels candidats**. Une détection éloignée d'un point ANFR connu n'est pas une preuve de site radioélectrique non déclaré.

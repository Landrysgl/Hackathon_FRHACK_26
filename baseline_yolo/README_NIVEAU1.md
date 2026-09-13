# FrHack! 2026 — Challenge 4 ANFR x ISEP
## Niveau 1 corrigé — pipeline reproductible

Ce dossier reconstruit proprement le Niveau 1 après audit du pipeline initial.

### Ce que corrige cette version

- déduplication par `SUP_ID` avant tout échantillonnage ;
- chemins relatifs et portables via `pathlib` ;
- conversion DMS -> latitude/longitude contrôlée ;
- sélection reproductible de 300 supports physiques distincts ;
- exclusion des supports manifestement invisibles depuis une orthophoto
  (`Tunnel`, `Intérieur sous-terrain`, `Intérieur galerie`) ;
- images IGN 1024x1024 centrées exactement sur la coordonnée ANFR ;
- cache local des tuiles IGN pour accélérer les relances ;
- contrôle d'intégrité des 300 JPEG ;
- split train/val/test avec regroupement spatial pour limiter la fuite géographique ;
- labels faibles explicitement séparés de la vérité terrain ;
- validation manuelle prise uniquement dans le split test ;
- `visible` = bbox humaine, `non_visible` = vraie image négative,
  `ambigu` = exclue de l'évaluation ;
- métriques manuelles TP / FP / FN / précision / rappel ;
- mAP50 et mAP50-95 calculés sur les annotations humaines, et non sur les labels faibles ;
- audit final du dataset.

---

## Structure attendue avant de commencer

Place ce dossier dans :

```text
Hackathon_FRHACK_26/
└── baseline_yolo/
    ├── 01_exploration_anfr.py
    ├── ...
    └── data/
        └── raw/
            ├── anfr/
            │   └── SUP_SUPPORT.txt
            └── references/
                └── SUP_NATURE.txt
```

Le code accepte aussi quelques variantes historiques de dossier pour `SUP_NATURE.txt`.

---

## Installation

Depuis `Hackathon_FRHACK_26` :

```bash
cd baseline_yolo
python -m pip install -r requirements_niveau1.txt
```

---

## Ordre d'exécution

```bash
python 01_exploration_anfr.py
python 02_choix_zone.py
python 03_test_image_ign.py
python 04_telechargement_ign_batch.py
python 05_inspection_images.py
python 06_creation_dataset_yolo.py
python 07_preparer_validation_manuelle.py
```

Ouvre ensuite :

```text
07_annotation_manuelle.ipynb
```

et annote les 20 images de validation humaine.

Puis :

```bash
python 08_entrainement_baseline.py
python 09_evaluation_labels_faibles.py
python 10_evaluation_validation_manuelle.py
python 11_audit_final_dataset.py
```

---

## Sorties principales

```text
data/
├── supports_yvelines.csv
├── supports_selectionnes_niveau1.csv
├── images_ign_yvelines/
├── inspection_images/
├── dataset_yolo/
│   ├── images/{train,val,test}
│   ├── labels/{train,val,test}
│   ├── dataset.yaml
│   └── split_manifest.csv
└── manual_validation/
    ├── images/
    ├── previews/
    ├── annotations.csv
    └── evaluation/

runs/
└── baseline_weak/
    └── weights/best.pt
```

---

## Important : labels faibles

Le label automatique est une boîte de 160x160 pixels centrée sur la coordonnée ANFR
dans une image de 1024x1024 :

```text
classe 0
centre = (0.5, 0.5)
largeur = hauteur = 160 / 1024 = 0.15625
```

C'est une **annotation faible**, pas une vérité terrain.

Les métriques de `09_evaluation_labels_faibles.py` sont donc seulement des métriques
de cohérence avec ces labels faibles. Les métriques à présenter comme validation réelle
sont celles de `10_evaluation_validation_manuelle.py`.

---

## Sauvegarde recommandée sur Onyxia

Ne laisse plus l'unique copie du travail dans le disque local d'un service temporaire.

À minima :

- pousse tous les scripts et petits CSV sur GitHub ;
- conserve les données lourdes (`images_ign_yvelines`, poids, runs) dans un stockage persistant/bucket ;
- le dossier `data/raw` peut rester hors Git si les fichiers ANFR sont récupérables ailleurs.

Le pipeline est reproductible : si les images sont perdues, le même `RANDOM_SEED=42`
redonne les mêmes 300 `SUP_ID`.

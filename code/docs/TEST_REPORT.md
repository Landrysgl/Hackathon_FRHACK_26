# Rapport de tests du dossier Pyl-Poil

Date de préparation : **14 septembre 2026**.

## Contrôles exécutés avec succès

- compilation Python de `src/`, `scripts/` et `tests/` avec `compileall` ;
- **14 tests unitaires / intégration simulée réussis** ;
- reproduction exacte de la sélection Niveau 1 : **300 supports** avec le seed 42 ;
- reproduction exacte du split spatial : **210 train / 45 validation / 45 test** ;
- distance inter-split minimale reproduite : **216,2 m**, supérieure au regroupement à 200 m ;
- reconstruction testée d'un dataset faible YOLO avec bbox centrale 160×160 px ;
- catalogue ANFR Yvelines contrôlé : **1 464 supports** ;
- sélection Niveau 3 reproduite exactement : centres `2274825 / 1122334 / 458152`, densités **16 / 3 / 1** dans 500 m ;
- calibration ANFR ↔ centre visuel reproduite sur **175 supports visibles** avec les mêmes distances et percentiles de référence ;
- pipeline géographique Niveau 3 reproduit exactement à partir des prédictions sauvegardées : **174 → 53 → 46**, avec les mêmes `detection_id` représentatifs ;
- revue humaine contrôlée : **27 plausibles / 17 faux positifs / 2 incertains** ;
- sélection finale contrôlée : **D003 / D006 / D007**, images présentes ;
- intégrité des **deux checkpoints** vérifiée par taille + SHA-256 ;
- génération réussie de la carte HTML finale ;
- génération réussie des **11 figures synthétiques** de restitution ;
- installation editable du package vérifiée avec `pip install -e . --no-deps --no-build-isolation` ;
- commande `pyl-poil --help` validée.

Commande de test principale :

```bash
python -m compileall -q src scripts tests
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/verify_reference_results.py
```

## Cohérences corrigées pendant la réorganisation

- nom de l'outil uniformisé en **Pyl-Poil** ;
- ancien ensemble d'environ 70 scripts exploratoires remplacé par des modules et points d'entrée fonctionnels ;
- modèle final uniformisé sur `pyl_poil_final_yolov8n.pt` provenant du run `niveau2_renforce_coco_sans_rotation` ;
- baseline Niveau 1 conservée séparément pour montrer la progression ;
- ancienne sélection Niveau 3 `D001 / I002 / P003` retirée des livrables finaux ;
- sélection finale fixée à `D003 / D006 / D007` ;
- code de sélection 300 supports, split spatial, calibration 150 m et sélection des trois zones réintroduit dans le pipeline reproductible ;
- `folium` et `matplotlib` déclarés explicitement comme dépendances ;
- chemins absolus Onyxia retirés du code actif et normalisés dans les métadonnées historiques ;
- paramètres communs centralisés dans `config/default.yaml` ;
- lien Cartoradio contrôlé dans l'ordre **longitude / latitude** ;
- résultats, figures, historique des expériences et modèles regroupés dans des répertoires explicites.

## Limite du test dans l'environnement de préparation

Le paquet **`ultralytics` n'est pas installé dans l'environnement utilisé pour préparer ce ZIP**, et cet environnement ne permet pas de le récupérer depuis Internet. Le seul contrôle d'installation qui échoue ici est donc l'import `ultralytics`.

Par conséquent, le fichier `.pt` final n'a pas été ré-inféré ici sur les orthophotos réelles. En revanche :

1. les checkpoints sont présents et leurs SHA-256 correspondent aux manifestes ;
2. le raccordement YOLO → géolocalisation → ANFR est couvert par un test avec backend simulé ;
3. toute la chaîne post-inférence a été rejouée sur les **174 prédictions sauvegardées** ;
4. la calibration, le choix des zones, le filtrage, la déduplication, les cartes et les figures ont été relancés.

### Test final à faire sur Onyxia avant la démonstration

Dans l'environnement Onyxia où Ultralytics était déjà utilisé :

```bash
cd code
source .venv/bin/activate   # ou activer l'environnement utilisé pour le hackathon
pip install -r requirements.txt
pip install -e .
python scripts/check_installation.py
```

Puis faire un **smoke test d'inférence** sur une zone :

```bash
pyl-poil \
  --lat 48.7975 \
  --lon 2.1300 \
  --supports data/reference/supports_yvelines.csv \
  --weights models/pyl_poil_final_yolov8n.pt \
  --output outputs/smoke_test_dense
```

Ce dernier test est le seul contrôle important restant à effectuer dans l'environnement GPU/réseau de démonstration.

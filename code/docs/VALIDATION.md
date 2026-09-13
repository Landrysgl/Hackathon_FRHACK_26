# Validation et contrôles du livrable code

## Contrôles statiques

- tous les fichiers Python doivent compiler ;
- aucun chemin absolu Onyxia n'est requis par le code actif ;
- les paramètres IGN / détection / filtrage sont centralisés dans `config/default.yaml` ;
- les deux modèles ont un manifeste de taille + SHA-256 ;
- les gros caches et orthophotos reconstructibles ne sont pas versionnés.

## Suite de tests

`PYTHONPATH=src python -m unittest discover -s tests -v` exécute **14 tests** couvrant :

1. préparation d'un catalogue ANFR ;
2. conversions WGS84 ↔ Web Mercator ;
3. distance de Haversine ;
4. URL Cartoradio et ordre longitude/latitude ;
5. filtre de distance ANFR ;
6. déduplication spatiale ;
7. géolocalisation d'une détection avec backend YOLO simulé ;
8. construction du dataset humain ;
9. sélection exacte des 300 supports Niveau 1 ;
10. split spatial exact 210 / 45 / 45 ;
11. construction d'annotations faibles centrales ;
12. calibration exacte des 175 supports visibles ;
13. sélection exacte des trois zones Niveau 3 ;
14. résultats de référence Niveau 3 et candidats finaux.

## Vérification consolidée

```bash
PYTHONPATH=src python scripts/verify_reference_results.py
```

Cette commande contrôle notamment le catalogue, la sélection/split Niveau 1, la sélection des zones, l'intégrité des modèles, les comptes **174 → 53 → 46**, les statuts de revue et D003 / D006 / D007.

## Limite d'environnement

L'inférence Ultralytics réelle doit être testée une dernière fois sur Onyxia, car l'environnement utilisé pour assembler ce livrable ne contient pas `ultralytics`. Voir `docs/TEST_REPORT.md` pour la commande exacte.

# Résumé de la réorganisation du code

Le dossier de développement initial contenait plus de 70 scripts successifs (`01_...` à `72_...`) correspondant à l'historique du hackathon. Cette organisation était utile pendant l'expérimentation mais pas adaptée à une livraison ou à une reprise par un tiers.

## Ce qui a été remplacé

Les scripts historiques ont été regroupés en modules fonctionnels :

- `anfr.py` : chargement et préparation de la référence ANFR ;
- `imagery.py` : accès WMTS IGN/GeoPF et création des patches ;
- `dataset.py` : téléchargement des images d'apprentissage et reconstruction du dataset humain ;
- `detector.py` : inférence YOLO et géolocalisation ;
- `candidates.py` : filtre de distance et déduplication ;
- `mapping.py` : carte HTML interactive et liens Cartoradio ;
- `pipeline.py` : orchestration d'une analyse complète ;
- `training.py` : entraînement et évaluation du modèle.

Les points d'entrée destinés aux utilisateurs sont regroupés dans `scripts/` avec des noms fonctionnels.

## Éléments historiques volontairement non repris comme code actif

- variantes d'entraînement intermédiaires ;
- scripts temporaires de création de mosaïques/crops ;
- ancien pilote Niveau 3 à une seule zone ;
- ancienne sélection de candidats `D001 / I002 / P003` ;
- cartes devenues obsolètes ;
- anciens fichiers présentant encore le modèle rotation ±180° comme modèle final ;
- caches IGN, runs Ultralytics, `last.pt`, environnements virtuels et images dupliquées.

Les conclusions de ces essais ne sont pas perdues : les métriques et l'analyse d'erreurs utiles sont conservées dans `results/` et expliquées dans `docs/`.

## Incohérences corrigées

- le modèle final est désormais uniformément `niveau2_renforce_coco_sans_rotation` ;
- D003, D006 et D007 sont uniformément les trois candidats finaux ;
- `folium` est déclaré comme dépendance ;
- les chemins absolus Onyxia ont été supprimés ;
- les paramètres sont centralisés dans `config/default.yaml` ;
- le lien Cartoradio utilise bien l'ordre `longitude/latitude` ;
- la carte finale met en évidence les trois candidats finaux et sépare les couches plausible / faux positif / incertain ;
- le catalogue ANFR, les annotations, les résultats et le modèle possèdent maintenant des contrôles automatiques de cohérence.

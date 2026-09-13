# Résultats consolidés

## 1. Progression Niveau 1 → Niveau 2

### Niveau 1

Sur le premier test humain :

- 20 images annotées ;
- 13 utilisables (10 visibles, 3 non visibles) ;
- 7 ambiguës exclues ;
- mAP50 = **0,012614** ;
- mAP50-95 = **0,001670**.

Le résultat confirme que la coordonnée ANFR utilisée comme label faible est insuffisante pour apprendre précisément l'apparence visuelle d'un support.

### Niveau 2 final

Couverture des annotations :

| Jeu | Total | Visible | Non visible | Ambigu | Utilisable |
|---|---:|---:|---:|---:|---:|
| train lot 1 | 80 | 51 | 10 | 19 | 61 |
| train lot 2 | 130 | 93 | 11 | 26 | 104 |
| validation | 45 | 31 | 4 | 10 | 35 |
| holdout final | 25 | 21 | 2 | 2 | 23 |

Le train total représente donc 210 images revues, dont **165 utilisables**. La validation en contient 35. Le holdout final n'a pas été utilisé pour choisir les hyperparamètres.

Métriques du modèle final sur le holdout :

| Mesure | Valeur |
|---|---:|
| mAP50 | **0,249499** |
| mAP50-95 | **0,113819** |
| précision Ultralytics | **0,331273** |
| rappel Ultralytics | **0,428571** |
| TP à conf=0,05 | **13** |
| FP à conf=0,05 | **111** |
| FN à conf=0,05 | **8** |
| précision à conf=0,05 | **0,104839** |
| rappel à conf=0,05 | **0,619048** |

Le grand nombre de faux positifs au seuil 0,05 est connu. Pour la recherche de candidats, ce seuil favorise le rappel ; les sorties sont ensuite croisées avec l'ANFR, dédupliquées et revues visuellement.

## 2. Niveau 3 multizone

### Résultats globaux

| Étape | Nombre |
|---|---:|
| patches analysés | 75 |
| détections brutes | 174 |
| détections à ≥150 m | 53 |
| candidats uniques après déduplication | 46 |
| plausibles après revue | 27 |
| faux positifs | 17 |
| incertains | 2 |

Le taux global de plausibles parmi les candidats filtrés est **58,7 %**. Il ne s'agit pas d'une précision de détection de sites non déclarés : la catégorie `plausible` signifie uniquement que la structure observée mérite un examen.

### Résultats selon le contexte

| Zone | Brutes | ≥150 m | Uniques | Plausibles | Faux positifs | Incertains | Taux plausibles |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 141 | 29 | 22 | 21 | 1 | 0 | **95,5 %** |
| intermédiaire | 12 | 5 | 5 | 2 | 2 | 1 | **40,0 %** |
| peu dense | 21 | 19 | 19 | 4 | 14 | 1 | **21,1 %** |

Cette différence est un résultat important : un simple filtre de distance est beaucoup plus sujet aux faux positifs dans les zones peu denses. Un développement futur devrait donc intégrer davantage de contexte visuel ou adapter le filtrage au type de territoire.

## 3. Trois candidats présentés

| Rang | ID | Type visuel | Confiance | Distance ANFR | Latitude | Longitude |
|---:|---|---|---:|---:|---:|---:|
| 1 | **D003** | mât | 0,489 | 155,6 m | 48,801306 | 2,125993 |
| 2 | **D006** | bâtiment | 0,400 | 162,3 m | 48,799445 | 2,130660 |
| 3 | **D007** | bâtiment | 0,321 | 250,7 m | 48,794343 | 2,130314 |

Les images correspondantes sont dans `results/level3/final_candidates_images/` et le montage dans `results/level3/final_candidates_montage.jpg`.

Pour chaque candidat, `results/level3/final_candidates.csv` contient également un lien Cartoradio centré sur ses coordonnées.

## 4. Précautions d'interprétation

L'outil sert à **prioriser des zones à examiner**. Les candidats doivent être confrontés à Cartoradio, à la date des orthophotos et, si nécessaire, à d'autres sources. Les causes possibles d'une absence de correspondance proche incluent les faux positifs, supports sans radio, décalages de géoréférencement, différences de date, installations récentes/démantelées ou erreurs de classification.

## 5. Figures prêtes pour la restitution

Les figures quantitatives et les exemples visuels sont regroupés dans `results/figures/`. Leur index détaillé est dans `results/figures/README.md` et elles sont régénérables avec `python scripts/generate_result_figures.py`.

La carte HTML interactive de référence est `results/level3/carte_candidats_finale.html`.

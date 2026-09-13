# Niveau 3 — Recherche de candidats hors catalogue ANFR

## Méthode

Le modèle final du Niveau 2 a été appliqué sur une zone pilote
découpée en 25 patches IGN de 1024 × 1024 pixels.

Chaque détection a été géoréférencée puis comparée aux 1 464 supports
du catalogue ANFR des Yvelines.

La calibration sur 175 supports visibles annotés manuellement
(train + validation uniquement) montre :

- médiane : 11,73 m
- P95 : 30,71 m
- P99 : 47,16 m
- maximum : 50,49 m

Le seuil de sélection des candidats Niveau 3 a donc été fixé à 150 m,
soit très au-delà de la dispersion observée sur les sites connus.

## Résultats

- Détections brutes : 138
- Détections à au moins 150 m d'un support ANFR : 28
- Candidats géographiques après déduplication à 30 m : 22
- Candidats visuellement plausibles : 21
- Faux positifs : 1
- Incertains : 0
- Taux de candidats plausibles : 95.5 %

## Interprétation

Les candidats classés plausibles présentent visuellement une structure
compatible avec un support ou une infrastructure pouvant accueillir des
équipements radioélectriques.

Ils ne doivent cependant pas être interprétés comme des sites
radioélectriques non déclarés confirmés. Une validation terrain,
administrative ou par une source externe serait nécessaire pour conclure.

## Fichiers produits

- `candidats_plausibles.csv`
- `candidats_plausibles.geojson`
- `carte_candidats_niveau3.html`
- `resume_niveau3.json`

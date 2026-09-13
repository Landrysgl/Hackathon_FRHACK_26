# Analyse d'erreurs historique

Ces fichiers conservent la phase d'analyse des faux positifs menée avant le modèle final.

- `initial_validation_error_summary.csv` résume, image par image, le nombre de prédictions et le meilleur IoU sur une validation intermédiaire à faible confiance ;
- `hard_negative_candidates.csv` liste les faux positifs extraits comme candidats à l'apprentissage par hard negatives.

Cette expérimentation n'a pas amélioré la mAP sur le test humain initial ; elle est conservée comme trace de la démarche expérimentale, pas comme étape obligatoire du pipeline final.

Les erreurs les plus importantes observées sont cohérentes avec le compromis choisi : à faible seuil de confiance, le rappel augmente mais de nombreuses structures de bâtiments/voirie sont confondues avec des supports. Le Niveau 3 atténue ce problème par croisement ANFR, déduplication et revue humaine.

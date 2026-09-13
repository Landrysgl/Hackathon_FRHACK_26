# Niveau 2 — Résultats finaux

## Données humaines

- 80 images annotées manuellement
- 51 visibles
- 10 non visibles
- 19 ambiguës

## Meilleur modèle

Fine-tuning du modèle Niveau 1 sur annotations humaines avec rotation ±180°.

- mAP50 test humain : 0.016102
- mAP50-95 test humain : 0.002161
- Recall Ultralytics : 0.400

## Conclusions expérimentales

- Les annotations humaines seules sont insuffisantes avec le volume actuel.
- Le fine-tuning depuis le Niveau 1 est préférable à un entraînement humain seul.
- Les hard negatives testés n'améliorent pas la mAP sur le test humain.
- La rotation ±180° améliore le meilleur modèle mono-classe.
- Le multi-classe n'est pas exploitable avec le nombre actuel d'exemples par classe.
- Le modèle reste très sous-confiant : aucune détection exploitable à conf=0.03 sur le test humain.

## Modèle retenu

`runs/niveau2_finetune_rotation180/weights/best.pt`
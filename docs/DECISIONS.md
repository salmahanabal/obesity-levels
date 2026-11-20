# Decisions

- **Nom du projet** : `obesity-levels`.
- **Seed** : 42 partout, lu depuis `config.yaml`.
- **Modalite rare** : `CALC = Always` fusionnee dans `Frequently` (environ 1 ligne).
- **Split unique** : 70/30 stratifie, indices figes dans `data/processed/split_indices.json`.
- **Seuils IMC (pipeline C)** : seuils OMS 18.5/25/30/35/40 ; coupure `Overweight_Level_I/II` = 26.9,
  valeur optimale sur le train. Les bornes min/max observees se chevauchent (lignes SMOTE), voir
  `reports/tables/bmi_by_class.csv`.
- **Champion** : choisi sur la CV du train (F1 macro / RMSE), jamais sur le test.
- **Fuite de donnees** : le pipeline A exclut `Weight` et `Height` (la cible derive de l'IMC).

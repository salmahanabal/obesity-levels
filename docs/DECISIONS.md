# Decisions

- **Nom du projet** : `obesity-levels`.
- **Seed** : 42 partout, lu depuis `config.yaml`.
- **Modalite rare** : `CALC = Always` fusionnee dans `Frequently` (environ 1 ligne).
- **Split unique** : 70/30 stratifie, indices figes dans `data/processed/split_indices.json`.
- **Fuite de donnees** : le pipeline A exclut `Weight` et `Height` (la cible derive de l'IMC).

# Plan du rapport (5-10 pages)

1. **Introduction** : contexte, dataset UCI (2 111 lignes), problématique (habitudes de vie -> niveau d'obésité et poids).
2. **Exploration** : doublons, valeurs, distribution de la cible, fuite (IMC), données synthétiques (proxy), biais de genre.
   Figures : `reports/figures/eda_*.png`, tableau `bmi_by_class.csv`.
3. **Préprocessing** : nettoyage, split unique 70/30, ColumnTransformer (justifications).
4. **Modélisation** : DT, KNN, SVM, RF (GridSearchCV, F1 macro) ; Linear/Ridge/Lasso + OLS (VIF, résidus).
   Tableaux : `model_comparison.csv`, `regression_comparison.csv`.
5. **Évaluation** : matrices de confusion (`model_cm_*.png`), accuracy ±1, pipeline C (`consistency.csv`).
6. **Analyse critique** : surapprentissage, erreurs, données synthétiques, biais de genre, facteurs clés
   (`feature_importances.csv`), fuite (`leakage_comparison.csv`).
7. **MLOps** : MLflow (expériences, registre, alias champion), API FastAPI, Docker, CI.
8. **Conclusion, limites, recommandations.**

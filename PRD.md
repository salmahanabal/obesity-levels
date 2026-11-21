# PRD — Projet Data Mining M1 : niveaux d'obésité (version condensée)

Cahier des charges résumé ; l'état d'avancement est dans le README et `docs/DECISIONS.md`.

## Objectif
Flux complet **Exploration → Préprocessing → Modélisation → Évaluation → Analyse critique** sur le dataset UCI
*Estimation of Obesity Levels Based on Eating Habits and Physical Condition* (2 111 lignes, 17 colonnes), avec
classification multiclasse, régression linéaire et une couche MLOps légère (MLflow, API, Docker, tests, CI).

Livrables : rapport PDF (5-10 pages), code (notebooks en français + package `src/obesity/`), présentation.

## Dataset : particularités
1. La classe est dérivée de l'IMC = Weight / Height² : donner `Weight`/`Height` au classifieur est une **fuite**.
2. ~77 % de lignes synthétiques (SMOTE) : scores potentiellement optimistes.
3. ~24 doublons, supprimés avant le split.
4. Modalités rares (`CALC = Always`) fusionnées dans `Frequently`.
5. Biais possible de genre sur `Obesity_Type_II` / `Obesity_Type_III` : à mesurer et discuter.

## Conception expérimentale (un seul split 70/30 stratifié, seed 42)
| Pipeline | Tâche | Cible | Variables |
|---|---|---|---|
| A | Classification | `NObeyesdad` | tout sauf `Weight`, `Height` |
| B | Régression linéaire | `Weight` | tout sauf la classe (avec `Height`) |
| C | Cohérence | classe dérivée | IMC = poids prédit / Height², comparé à A |
| Réf. fuite | Contraste | `NObeyesdad` | A + `Weight` + `Height` (tag `leakage_reference=true`) |

## Exigences par étape
- **Exploration** : diagnostic, statistiques par classe, figures, outliers IQR, proxy de synthétique, démonstration de la fuite.
- **Préprocessing** : nettoyage déterministe, split unique, `ColumnTransformer` fitté sur le train seulement.
- **Modélisation** : DT, KNN, SVM (+ RF) avec GridSearchCV (F1 macro, 5 plis) ; Linear/Ridge/Lasso (RMSE), OLS statsmodels, VIF, diagnostics.
- **Évaluation** : accuracy, balanced accuracy, F1/recall par classe, matrices de confusion, accuracy ±1, ROC OvR ; MAE/RMSE/R² ; pipeline C. Champion choisi sur la CV, jamais sur le test.
- **Analyse critique** : surapprentissage, erreurs, données synthétiques, biais de genre, facteurs clés, fuite.

## MLOps
- **MLflow** : SQLite, expériences `obesity-classification` / `obesity-regression`, un run par modèle (params, métriques, artefacts, modèle signé), tags `git_commit`, `dataset_hash`, `split_seed`, registre avec alias `champion`.
- **API FastAPI** : `GET /health`, `POST /predict/class`, `POST /predict/weight`, schémas Pydantic v2 stricts.
- **Docker** (utilisateur non-root, healthcheck) et **CI GitHub Actions** (`ruff`, `pytest`).
- **Tests** : split, préprocessing, absence de fuite, seuils IMC, pipelines, API.

## Critères d'acceptation
`make all` de bout en bout · 4 notebooks exécutables · un seul split testé · pipeline A sans `Weight`/`Height` ·
modèles optimisés et tableau comparatif · régression + OLS + VIF · pipeline C comparé à A · analyse critique complète ·
MLflow (2 expériences, 2 modèles enregistrés) · API testée et conteneur répondant sur `/health` · `pytest`/`ruff`/CI verts ·
README complet.

## Optionnel
DVC, Evidently, Streamlit (interface de test réalisée).

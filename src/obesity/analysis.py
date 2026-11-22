"""Analyse critique : surapprentissage, erreurs, donnees synthetiques, biais, facteurs cles."""
from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve

from obesity import config as cfg
from obesity.eda import synthetic_proxy
from obesity.evaluate import accuracy_pm1
from obesity.preprocessing import load_split

SEED = cfg.SEED


def load_models():
    d = cfg.path("models_dir")
    return joblib.load(d / "classifier.joblib"), joblib.load(d / "regressor.joblib")


def load_linear():
    """Meilleure regression lineaire (pour les coefficients), meme si le champion est non lineaire."""
    return joblib.load(cfg.path("models_dir") / "regressor_linear.joblib")


def raw_proxy(df: pd.DataFrame) -> pd.Series:
    """Proxy de synthetique : calcule sur le brut, car le nettoyage arrondit les ordinales."""
    from obesity.data import load_raw

    raw = load_raw().drop_duplicates().reset_index(drop=True)
    return synthetic_proxy(raw).reindex(df.index)


def overfit_table() -> pd.DataFrame:
    """Ecart train/test pour chaque classifieur."""
    t = cfg.path("tables_dir")
    d = pd.read_csv(t / "classification_results.csv")
    d = d[~d["model"].str.endswith("_LEAKAGE")]
    out = d[["model", "train_f1_macro", "cv_f1_macro", "test_f1_macro"]].copy()
    out["ecart_train_test"] = out["train_f1_macro"] - out["test_f1_macro"]
    return out.round(3)


def plot_learning_curves(clf, reg, train: pd.DataFrame, path) -> None:
    from obesity.preprocessing import xy

    _fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, est, task, cv, scoring, title in (
        (ax[0], clf, "classification", StratifiedKFold(5, shuffle=True, random_state=SEED), "f1_macro",
         "Classifieur champion (F1 macro)"),
        (ax[1], reg, "regression", KFold(5, shuffle=True, random_state=SEED),
         "neg_root_mean_squared_error", "Régression champion (-RMSE)"),
    ):
        X, y = xy(train, task)
        sizes, tr, va = learning_curve(est, X, y, cv=cv, scoring=scoring, n_jobs=-1,
                                       train_sizes=np.linspace(0.2, 1, 6))
        a.plot(sizes, tr.mean(1), "o-", label="Train")
        a.plot(sizes, va.mean(1), "o-", label="Validation")
        a.set(title=title, xlabel="Taille du train", ylabel="Score")
        a.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def errors_table() -> pd.DataFrame:
    """Confusions les plus frequentes et part des erreurs entre classes voisines (pipeline A)."""
    p = pd.read_csv(cfg.path("tables_dir") / "consistency_predictions.csv", index_col=0)
    rank = {c: i for i, c in enumerate(cfg.load_config()["class_order"])}
    err = p[p["y"] != p["A"]].copy()
    err["distance"] = (err["y"].map(rank) - err["A"].map(rank)).abs()
    return err.groupby(["y", "A"]).size().sort_values(ascending=False).head(10).reset_index(name="n")


def errors_a_vs_c() -> pd.DataFrame:
    p = pd.read_csv(cfg.path("tables_dir") / "consistency_predictions.csv", index_col=0)
    a_ok, c_ok = p["y"] == p["A"], p["y"] == p["C"]
    return pd.crosstab(pd.Series(a_ok, name="A correct"), pd.Series(c_ok, name="C correct"))


def synthetic_split(clf, df: pd.DataFrame) -> pd.DataFrame:
    """Performance du champion selon le proxy de synthetique (heuristique)."""
    _, test = load_split(df)
    flag = raw_proxy(df).loc[test.index]
    t = cfg.load_config()["target"]
    rows = []
    for val, name in ((True, "synthetique (proxy)"), (False, "collecte (proxy)")):
        s = test[flag == val]
        pred = clf.predict(s[list(clf.feature_names_in_)])
        rows.append({"groupe": name, "n": len(s), "accuracy": accuracy_score(s[t], pred),
                     "f1_macro": f1_score(s[t], pred, average="macro"),
                     "accuracy_pm1": accuracy_pm1(s[t], pred)})
    return pd.DataFrame(rows).round(3)


def regression_synthetic_split(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Erreur de poids (RMSE, MAE, R2) par modele selon le proxy de synthetique : les lignes synthetiques
    sont des interpolations proches du train, donc plus faciles pour les modeles non lineaires."""
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

    _, test = load_split(df)
    flag = raw_proxy(df).loc[test.index]
    w = cfg.load_config()["weight_col"]
    rows = []
    for name, model in models.items():
        for val, group in ((True, "synthetique (proxy)"), (False, "collecte (proxy)")):
            s = test[flag == val]
            pred = model.predict(s[list(model.feature_names_in_)])
            rows.append({"modele": name, "groupe": group, "n": len(s),
                         "rmse": root_mean_squared_error(s[w], pred),
                         "mae": mean_absolute_error(s[w], pred), "r2": r2_score(s[w], pred)})
    return pd.DataFrame(rows).round(3)


def gender_metrics(clf, df: pd.DataFrame) -> pd.DataFrame:
    _, test = load_split(df)
    t = cfg.load_config()["target"]
    rows = []
    for g, s in test.groupby("Gender"):
        pred = clf.predict(s[list(clf.feature_names_in_)])
        rows.append({"Gender": g, "n": len(s), "accuracy": accuracy_score(s[t], pred),
                     "f1_macro": f1_score(s[t], pred, average="macro")})
    return pd.DataFrame(rows).round(3)


def gender_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df[cfg.load_config()["target"]], df["Gender"]).reindex(
        cfg.load_config()["class_order"])


def importances(clf, reg, df: pd.DataFrame) -> pd.DataFrame:
    """Importances RF (impurity + permutation sur test) et coefficients standardises de la regression."""
    _, test = load_split(df)
    t = cfg.load_config()["target"]
    Xc, yc = test[list(clf.feature_names_in_)], test[t]
    model = clf.named_steps["model"]
    names = clf.named_steps["prep"].get_feature_names_out()
    perm = permutation_importance(clf, Xc, yc, n_repeats=10, random_state=SEED, scoring="f1_macro")
    out = pd.DataFrame({"variable": clf.feature_names_in_, "perm_importance_test": perm.importances_mean})
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=names)
        # agregation des dummies par variable d'origine
        agg = {v: imp[[n for n in names if n == v or n.startswith(v + "_")]].sum()
               for v in clf.feature_names_in_}
        out["rf_impurity"] = out["variable"].map(agg)
    lin = reg if hasattr(reg.named_steps["model"], "coef_") else load_linear()
    coefs = pd.Series(np.abs(lin.named_steps["model"].coef_),
                      index=lin.named_steps["prep"].get_feature_names_out())
    agg_r = {v: coefs[[n for n in coefs.index if n == v or n.startswith(v + "_")]].sum()
             for v in clf.feature_names_in_}
    out["abs_coef_regression"] = out["variable"].map(agg_r)
    for c in [c for c in out.columns if c != "variable"]:
        out[f"rang_{c}"] = out[c].rank(ascending=False).astype(int)
    return out.sort_values("perm_importance_test", ascending=False).round(4)

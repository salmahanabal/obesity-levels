"""Pipeline B : regression lineaire du poids (Linear/Ridge/Lasso), OLS statsmodels et diagnostics."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from mlflow.models import infer_signature
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from obesity import config as cfg
from obesity import tracking
from obesity.data import dataset_hash, load_clean
from obesity.evaluate import regression_metrics
from obesity.preprocessing import load_split, make_pipeline, xy

SEED = cfg.SEED
ALPHAS = np.logspace(-3, 2, 20)

MODELS = {
    "LinearRegression": (LinearRegression(), {}),
    "Ridge": (Ridge(random_state=SEED), {"alpha": ALPHAS}),
    "Lasso": (Lasso(random_state=SEED, max_iter=20000), {"alpha": ALPHAS}),
}


def fit_model(name: str, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    est, grid = MODELS[name]
    gs = GridSearchCV(
        make_pipeline(est, "regression"),
        {f"model__{k}": v for k, v in grid.items()} or {"model__fit_intercept": [True]},
        cv=KFold(cfg.load_config()["cv_folds"], shuffle=True, random_state=SEED),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return gs.fit(X, y)


def ols_analysis(X_tr: pd.DataFrame, y_tr: pd.Series, out_dir) -> dict:
    """OLS statsmodels (dummies drop_first) : summary, coefficients IC95, VIF, diagnostics."""
    Xd = pd.get_dummies(X_tr, drop_first=True).astype(float)
    Xc = sm.add_constant(Xd)
    res = sm.OLS(y_tr.astype(float), Xc).fit()
    (out_dir / "ols_summary.txt").write_text(str(res.summary()), encoding="utf-8")
    ci = res.conf_int()
    coefs = pd.DataFrame({"coef": res.params, "ci_low": ci[0], "ci_high": ci[1], "p_value": res.pvalues})
    coefs.to_csv(out_dir / "ols_coefficients.csv")
    vif = pd.DataFrame({
        "variable": Xd.columns,
        "vif": [variance_inflation_factor(Xc.values, i + 1) for i in range(Xd.shape[1])],
    })
    vif.to_csv(out_dir / "vif.csv", index=False)
    fig_dir = cfg.path("figures_dir")
    fitted, resid = res.fittedvalues, res.resid
    _fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(fitted, resid, s=8, alpha=0.5)
    ax[0].axhline(0, color="red")
    ax[0].set(xlabel="Valeurs prédites (kg)", ylabel="Résidus (kg)", title="Résidus vs prédictions")
    st.probplot(resid, plot=ax[1])
    ax[1].set_title("QQ-plot des résidus")
    ax[2].hist(resid, bins=30)
    ax[2].set(xlabel="Résidus (kg)", ylabel="Effectif", title="Histogramme des résidus")
    plt.tight_layout()
    path = fig_dir / "model_residuals_ols.png"
    plt.savefig(path, dpi=150)
    plt.close()
    bp_p = float(het_breuschpagan(resid, Xc)[1])
    return {"r2_train": float(res.rsquared), "bp_pvalue": bp_p, "max_vif": float(vif["vif"].max()),
            "figure": path}


def log_run(name, gs, train, test, ds_hash, extra_artifacts=()) -> dict:
    cols = list(gs.feature_names_in_)
    X_tr, y_tr = train[cols], train[cfg.load_config()["weight_col"]]
    X_te, y_te = test[cols], test[cfg.load_config()["weight_col"]]
    m = {**regression_metrics(y_tr, gs.predict(X_tr), "train_"),
         **regression_metrics(y_te, gs.predict(X_te), "test_")}
    params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(tracking.common_tags(ds_hash))
        mlflow.log_params(params)
        mlflow.log_metric("cv_rmse", -gs.best_score_)
        mlflow.log_metrics(m)
        for a in extra_artifacts:
            mlflow.log_artifact(str(a))
        mlflow.sklearn.log_model(gs.best_estimator_, name="model", signature=infer_signature(
            X_te, gs.predict(X_te)), input_example=X_te.head(3), serialization_format="cloudpickle")
        rid = run.info.run_id
    return {"model": name, "run_id": rid, "cv_rmse": -gs.best_score_, **m,
            "best_params": json.dumps(params, default=float)}


def main() -> pd.DataFrame:
    tracking.setup(cfg.load_config()["mlflow"]["reg_experiment"])
    df = load_clean()
    ds_hash = dataset_hash(df)
    train, test = load_split(df)
    X_tr, y_tr = xy(train, "regression")
    tables = cfg.path("tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    cfg.path("figures_dir").mkdir(parents=True, exist_ok=True)
    ols = ols_analysis(X_tr, y_tr, tables)
    print("OLS:", {k: v for k, v in ols.items() if k != "figure"})
    arts = [tables / "ols_summary.txt", tables / "ols_coefficients.csv", tables / "vif.csv", ols["figure"]]
    rows = []
    for name in MODELS:
        gs = fit_model(name, X_tr, y_tr)
        rows.append(log_run(name, gs, train, test, ds_hash, arts if name == "LinearRegression" else ()))
        print(f"== {name} cv_rmse={rows[-1]['cv_rmse']:.3f} test_rmse={rows[-1]['test_rmse']:.3f} "
              f"test_r2={rows[-1]['test_r2']:.3f}")
    if ols["bp_pvalue"] < 0.05:  # heteroscedasticite : essai log(Weight)
        y_log = np.log(y_tr)
        gs = fit_model("LinearRegression", X_tr, y_log)
        pred = np.exp(gs.predict(test[list(X_tr.columns)]))
        m = regression_metrics(test[cfg.load_config()["weight_col"]], pred, "test_")
        print("log(Weight) -> test:", {k: round(v, 3) for k, v in m.items()})
        pd.Series(m).to_csv(tables / "regression_log_weight.csv", header=["value"])
    out = pd.DataFrame(rows)
    out.to_csv(tables / "regression_results.csv", index=False)
    return out


if __name__ == "__main__":
    main()

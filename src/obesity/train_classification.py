"""Pipeline A : classification multiclasse (GridSearchCV + MLflow) et run de reference 'fuite'."""
from __future__ import annotations

import json
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from obesity import config as cfg
from obesity import tracking
from obesity.data import dataset_hash, load_clean
from obesity.evaluate import classification_metrics, per_class_report, plot_confusion
from obesity.ordinal import OrdinalClassifier
from obesity.preprocessing import load_split, make_pipeline, xy

SEED = cfg.SEED

MODELS = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=SEED),
        {
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_leaf": [1, 3, 5, 10],
            "criterion": ["gini", "entropy"],
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
    ),
    "SVM": (
        SVC(decision_function_shape="ovr", random_state=SEED),
        {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=SEED),
        {"n_estimators": [100, 300], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 3]},
    ),
    # Modeles supplementaires (au-dela du cahier des charges)
    "LogisticRegression": (
        LogisticRegression(max_iter=3000),
        {"C": [0.01, 0.1, 1, 10, 100]},
    ),
    "GradientBoosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, None],
            "max_leaf_nodes": [15, 31],
            "l2_regularization": [0.0, 1.0],
        },
    ),
    "MLP": (
        MLPClassifier(max_iter=1500, random_state=SEED),
        {"hidden_layer_sizes": [(64,), (128,), (64, 32)], "alpha": [0.0001, 0.01, 0.1]},
    ),
    "OrdinalLogReg": (
        OrdinalClassifier(base=LogisticRegression(max_iter=3000)),
        {"base__C": [0.01, 0.1, 1, 10, 100]},
    ),
}


def fit_model(name: str, train: pd.DataFrame, feature_cols: list[str] | None = None) -> GridSearchCV:
    """GridSearchCV (f1_macro, StratifiedKFold 5) sur le train uniquement."""
    est, grid = MODELS[name]
    X, y = xy(train, "classification")
    if feature_cols is not None:
        X = train[feature_cols]
    pipe = make_pipeline(est, "classification")
    if feature_cols is not None:  # run fuite : on ajoute Weight/Height en numeriques standardises
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from obesity.preprocessing import build_preprocessor

        base = build_preprocessor("classification")
        extra = [c for c in feature_cols if c in ("Weight", "Height")]
        prep = ColumnTransformer(
            [("base", base, [c for c in feature_cols if c not in extra]), ("leak", StandardScaler(), extra)],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        pipe = Pipeline([("prep", prep), ("model", est)])
    gs = GridSearchCV(
        pipe,
        {f"model__{k}": v for k, v in grid.items()},
        cv=StratifiedKFold(cfg.load_config()["cv_folds"], shuffle=True, random_state=SEED),
        scoring="f1_macro",
        n_jobs=-1,
    )
    gs.fit(X, y)
    return gs


def log_run(name: str, gs: GridSearchCV, train: pd.DataFrame, test: pd.DataFrame, ds_hash: str,
            leakage: bool = False) -> dict:
    """Un run MLflow par modele : params, metriques CV/test, artefacts, modele signe."""
    cols = list(gs.feature_names_in_)
    X_te, y_te = test[cols], test[cfg.load_config()["target"]]
    X_tr, y_tr = train[cols], train[cfg.load_config()["target"]]
    pred = gs.predict(X_te)
    proba = gs.predict_proba(X_te) if hasattr(gs.best_estimator_, "predict_proba") else None
    classes = gs.best_estimator_.classes_
    test_m = classification_metrics(y_te, pred, proba, classes)
    train_m = classification_metrics(y_tr, gs.predict(X_tr))
    run_name = f"{name}{'_LEAKAGE' if leakage else ''}"
    fig_dir = cfg.path("figures_dir")
    fig_dir.mkdir(parents=True, exist_ok=True)
    cm_path = fig_dir / f"model_cm_{run_name}.png"
    plot_confusion(y_te, pred, cm_path, run_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tracking.common_tags(ds_hash))
        mlflow.set_tag("leakage_reference", str(leakage).lower())
        mlflow.log_params({k.replace("model__", ""): v for k, v in gs.best_params_.items()})
        mlflow.log_metric("cv_f1_macro", gs.best_score_)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
        mlflow.log_artifact(str(cm_path))
        rep = per_class_report(y_te, pred).round(4)
        mlflow.log_dict(json.loads(rep.to_json(orient="index")), "classification_report.json")
        sig = infer_signature(X_te, pred)
        mlflow.sklearn.log_model(gs.best_estimator_, name="model", signature=sig,
                                 serialization_format="cloudpickle",
                                 input_example=X_te.head(3))
        run_id = run.info.run_id
    return {
        "model": run_name, "run_id": run_id, "cv_f1_macro": gs.best_score_,
        **{f"test_{k}": v for k, v in test_m.items()},
        **{f"train_{k}": v for k, v in train_m.items()},
        "best_params": json.dumps({k.replace("model__", ""): v for k, v in gs.best_params_.items()},
                                  default=str),
    }


def main(names: list[str] | None = None) -> pd.DataFrame:
    tracking.setup(cfg.load_config()["mlflow"]["clf_experiment"])
    df = load_clean()
    ds_hash = dataset_hash(df)
    train, test = load_split(df)
    rows = []
    for name in names or list(MODELS):
        print(f"== {name}")
        gs = fit_model(name, train)
        rows.append(log_run(name, gs, train, test, ds_hash))
        print(f"   cv f1_macro={gs.best_score_:.4f} test f1_macro={rows[-1]['test_f1_macro']:.4f}")
    # Run de reference : fuite (Weight + Height), jamais utilise comme modele final
    leak_cols = [*xy(train, "classification")[0].columns, "Weight", "Height"]
    gs = fit_model("RandomForest", train, leak_cols)
    rows.append(log_run("RandomForest", gs, train, test, ds_hash, leakage=True))
    out = pd.DataFrame(rows)
    tables = cfg.path("tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    out.to_csv(tables / "classification_results.csv", index=False)
    return out


if __name__ == "__main__":
    main(sys.argv[1:] or None)

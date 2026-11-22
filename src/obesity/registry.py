"""Enregistrement des champions dans le Model Registry MLflow (alias `champion`) et export joblib."""
from __future__ import annotations

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from obesity import config as cfg
from obesity.consistency import best_run


def register(run_id: str, name: str) -> str:
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    MlflowClient().set_registered_model_alias(name, "champion", mv.version)
    return str(mv.version)


def main() -> None:
    c = cfg.load_config()
    m = c["mlflow"]
    clf_id = best_run(m["clf_experiment"], "cv_f1_macro", False)
    reg_id = best_run(m["reg_experiment"], "cv_rmse", True)
    out = cfg.path("models_dir")
    out.mkdir(parents=True, exist_ok=True)
    for run_id, name, fname in ((clf_id, m["clf_model_name"], "classifier.joblib"),
                                (reg_id, m["reg_model_name"], "regressor.joblib")):
        version = register(run_id, name)
        joblib.dump(mlflow.sklearn.load_model(f"runs:/{run_id}/model"), out / fname)
        print(f"{name} v{version} (alias champion) -> {out / fname}")
    # meilleure regression lineaire, gardee pour les coefficients (le champion peut etre non lineaire)
    from obesity.train_regression import LINEAR

    runs = mlflow.search_runs(experiment_names=[m["reg_experiment"]], order_by=["metrics.cv_rmse ASC"])
    lin = runs[runs["tags.mlflow.runName"].isin(LINEAR)].iloc[0]
    joblib.dump(mlflow.sklearn.load_model(f"runs:/{lin['run_id']}/model"), out / "regressor_linear.joblib")
    print(f"regression lineaire ({lin['tags.mlflow.runName']}) -> {out / 'regressor_linear.joblib'}")


if __name__ == "__main__":
    main()

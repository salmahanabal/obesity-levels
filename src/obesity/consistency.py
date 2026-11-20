"""Pipeline C : classe derivee du poids predit (IMC = poids predit / taille^2), comparee a A."""
from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from obesity import config as cfg
from obesity import tracking
from obesity.evaluate import accuracy_pm1, classification_metrics, plot_confusion
from obesity.preprocessing import load_split


def bmi(weight, height):
    """IMC = poids / taille^2."""
    return np.asarray(weight, dtype=float) / np.asarray(height, dtype=float) ** 2


def bmi_to_class(values) -> np.ndarray:
    """Classe a partir de l'IMC via `bmi_thresholds` (bornes basses incluses)."""
    th = cfg.load_config()["bmi_thresholds"]
    order = cfg.load_config()["class_order"]
    lows = np.array([th[c][0] for c in order[1:]])  # bornes basses des classes 2..7
    idx = np.searchsorted(lows, np.asarray(values, dtype=float), side="right")
    return np.array(order)[idx]


def best_run(experiment: str, metric: str, ascending: bool, exclude_leak: bool = True) -> str:
    """run_id du meilleur run selon une metrique de CV (jamais le test)."""
    tracking.setup(experiment)
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=[
        f"metrics.{metric} {'ASC' if ascending else 'DESC'}"])
    if exclude_leak and "tags.leakage_reference" in runs:
        runs = runs[runs["tags.leakage_reference"] != "true"]
    return runs.iloc[0]["run_id"]


def main() -> dict:
    c = cfg.load_config()
    df = pd.read_csv(cfg.path("clean_csv"))
    _, test = load_split(df)
    clf_id = best_run(c["mlflow"]["clf_experiment"], "cv_f1_macro", False)
    reg_id = best_run(c["mlflow"]["reg_experiment"], "cv_rmse", True)
    clf = mlflow.sklearn.load_model(f"runs:/{clf_id}/model")
    reg = mlflow.sklearn.load_model(f"runs:/{reg_id}/model")
    y = test[c["target"]]
    pred_a = clf.predict(test[list(clf.feature_names_in_)])
    w_pred = reg.predict(test[list(reg.feature_names_in_)])
    pred_c = bmi_to_class(bmi(w_pred, test["Height"]))
    res = pd.DataFrame({
        "A_classifieur": classification_metrics(y, pred_a),
        "C_poids_predit": classification_metrics(y, pred_c),
    }).T
    res["accuracy_pm1"] = [accuracy_pm1(y, pred_a), accuracy_pm1(y, pred_c)]
    tables = cfg.path("tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    res.to_csv(tables / "consistency.csv")
    plot_confusion(y, pred_c, cfg.path("figures_dir") / "model_cm_pipeline_C.png", "Pipeline C")
    agree = float((pred_a == pred_c).mean())
    pd.DataFrame({"y": y.values, "A": pred_a, "C": pred_c, "weight_pred": w_pred},
                 index=test.index).to_csv(tables / "consistency_predictions.csv")
    print(res[["accuracy", "accuracy_pm1", "f1_macro"]].round(3))
    print("accord A/C :", round(agree, 3))
    return {"agreement": agree}


if __name__ == "__main__":
    main()

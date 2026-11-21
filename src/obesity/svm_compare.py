"""Comparaison SVM : decision_function_shape='ovr' vs strategies explicites One-vs-Rest / One-vs-One.

Evaluation par validation croisee sur le train uniquement (le test n'est pas reutilise).
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC

from obesity import config as cfg
from obesity.data import load_clean
from obesity.preprocessing import load_split, make_pipeline, xy


def compare() -> pd.DataFrame:
    """F1 macro (CV 5 plis sur le train) pour trois variantes du SVM aux meilleurs hyperparametres."""
    res = pd.read_csv(cfg.path("tables_dir") / "classification_results.csv")
    params = json.loads(res.loc[res["model"] == "SVM", "best_params"].iloc[0])
    train, _ = load_split(load_clean())
    X, y = xy(train, "classification")
    cv = StratifiedKFold(cfg.load_config()["cv_folds"], shuffle=True, random_state=cfg.SEED)
    variants = {
        "SVC (decision_function_shape='ovr', OvO interne)": SVC(decision_function_shape="ovr", **params),
        "OneVsRestClassifier(SVC)": OneVsRestClassifier(SVC(**params)),
        "OneVsOneClassifier(SVC)": OneVsOneClassifier(SVC(**params)),
    }
    rows = []
    for name, est in variants.items():
        s = cross_val_score(make_pipeline(est, "classification"), X, y, cv=cv, scoring="f1_macro")
        rows.append({"variante": name, "f1_macro_cv": round(s.mean(), 4), "ecart_type": round(s.std(), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.path("tables_dir") / "svm_ovr_ovo.csv", index=False)
    return out


if __name__ == "__main__":
    print(compare().to_string(index=False))

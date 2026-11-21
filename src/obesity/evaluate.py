"""Metriques d'evaluation (classification ordonnee et regression) et figures associees."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from obesity import config as cfg


def class_order(present: list[str] | None = None) -> list[str]:
    order = cfg.load_config()["class_order"]
    return [c for c in order if present is None or c in present]


def accuracy_pm1(y_true, y_pred) -> float:
    """Accuracy 'a +/- 1 classe' (les classes sont ordonnees)."""
    rank = {c: i for i, c in enumerate(cfg.load_config()["class_order"])}
    d = np.abs(pd.Series(y_true).map(rank).to_numpy() - pd.Series(y_pred).map(rank).to_numpy())
    return float((d <= 1).mean())


def classification_metrics(y_true, y_pred, proba=None, classes=None) -> dict[str, float]:
    """Accuracy, balanced accuracy, F1/precision/recall macro et weighted, +/-1, AUC OvR."""
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy_pm1": accuracy_pm1(y_true, y_pred),
    }
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        m.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    if proba is not None and classes is not None:
        m["roc_auc_ovr_macro"] = roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=list(classes)
        )
    return {k: float(v) for k, v in m.items()}


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    """Precision/recall/F1 par classe."""
    labels = class_order(sorted(set(y_true)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=labels)


def plot_confusion(y_true, y_pred, path, title: str) -> None:
    """Matrice de confusion en comptes et normalisee par ligne."""
    labels = class_order(sorted(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "Comptes"), (axes[1], cmn, ".2f", "Normalisée par ligne")):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")
        ax.set_title(f"{title} — {sub}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > data.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    """MAE, RMSE, R2."""
    return {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(root_mean_squared_error(y_true, y_pred)),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
    }


def f1_macro(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


STRENGTHS = {
    "DecisionTree": "Interprétable, rapide ; tendance au surapprentissage",
    "KNN": "Simple, sans hypothèse ; sensible à la dimension et au bruit",
    "SVM": "Bon compromis biais/variance, écart train/test réduit",
    "RandomForest": "Robuste, importances des variables",
    "LogisticRegression": "Linéaire, interprétable ; limitée sur les relations non linéaires",
    "GradientBoosting": "Boosting d'arbres, souvent le plus précis sur données tabulaires",
    "MLP": "Réseau de neurones ; sensible aux hyperparamètres, peu interprétable",
    "OrdinalLogReg": "Respecte l'ordre des classes ; erreurs surtout entre classes voisines",
}


def main() -> None:
    """Tableaux comparatifs (format du cahier des charges) a partir des resultats d'entrainement."""
    tables = cfg.path("tables_dir")
    clf = pd.read_csv(tables / "classification_results.csv")
    clf = clf[~clf["model"].str.endswith("_LEAKAGE")]
    cmp_ = pd.DataFrame({
        "Modèle": clf["model"],
        "Accuracy": clf["test_accuracy"].round(3),
        "F1-Score": clf["test_f1_macro"].round(3),
        "Recall": clf["test_recall_macro"].round(3),
        "Points forts": clf["model"].map(STRENGTHS),
    })
    cmp_.to_csv(tables / "model_comparison.csv", index=False)
    champ = clf.sort_values("cv_f1_macro", ascending=False).iloc[0]["model"]  # choix sur la CV
    leak = pd.read_csv(tables / "classification_results.csv")
    leak = leak[leak["model"].str.endswith("_LEAKAGE")]
    # meme algorithme (forêt) avec et sans la fuite, pour une comparaison equitable
    both = pd.concat([clf[clf["model"] == "RandomForest"], leak])
    both[["model", "test_accuracy", "test_f1_macro", "test_accuracy_pm1"]].to_csv(
        tables / "leakage_comparison.csv", index=False)
    reg = pd.read_csv(tables / "regression_results.csv")
    reg[["model", "train_mae", "train_rmse", "train_r2", "test_mae", "test_rmse", "test_r2"]].round(3).to_csv(
        tables / "regression_comparison.csv", index=False)
    print(cmp_.drop(columns="Points forts").to_string(index=False))
    print("Champion (CV) :", champ)


if __name__ == "__main__":
    main()

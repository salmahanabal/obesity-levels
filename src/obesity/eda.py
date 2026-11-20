"""Exploration : diagnostics, statistiques par classe, figures et demonstration de la fuite."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from obesity import config as cfg
from obesity.data import load_raw

DPI = 150


def diagnostics(df: pd.DataFrame) -> dict:
    """Resume : forme, types, valeurs manquantes, doublons."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nan": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne BMI = Weight / Height^2."""
    out = df.copy()
    out["BMI"] = out["Weight"] / out["Height"] ** 2
    return out


def synthetic_proxy(df: pd.DataFrame) -> pd.Series:
    """Heuristique : au moins une ordinale non entiere (avant arrondi) => ligne probablement SMOTE."""
    cols = cfg.load_config()["columns"]["ordinal_numeric"]
    return (df[cols] % 1 != 0).any(axis=1)


def iqr_outliers(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Nombre d'outliers (regle IQR 1.5) par colonne."""
    res = {}
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        i = q3 - q1
        res[c] = int(((df[c] < q1 - 1.5 * i) | (df[c] > q3 + 1.5 * i)).sum())
    return pd.Series(res)


def class_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Moyenne, mediane et ecart-type par classe."""
    t = cfg.load_config()["target"]
    return df.groupby(t)[cols].agg(["mean", "median", "std"])


def bmi_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau IMC min/max par classe (demonstration de la fuite)."""
    t = cfg.load_config()["target"]
    order = cfg.load_config()["class_order"]
    return add_bmi(df).groupby(t)["BMI"].agg(["min", "max"]).reindex(order)


def _save(name: str) -> None:
    out = cfg.path("figures_dir")
    out.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out / f"eda_{name}.png", dpi=DPI)
    plt.close()


def make_figures(df: pd.DataFrame) -> None:
    c = cfg.load_config()
    t, order = c["target"], c["class_order"]
    num = ["Age", "Height", "Weight"] + c["columns"]["numeric"][1:]
    df.hist(column=num, bins=20, figsize=(12, 8))
    plt.suptitle("Distribution des variables numériques")
    _save("hist_numeriques")
    _fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.ravel(), num):
        sns.boxplot(data=df, x=t, y=col, order=order, ax=ax)
        ax.tick_params(axis="x", rotation=90)
    plt.suptitle("Variables numériques par classe")
    _save("boxplots_par_classe")
    cat = c["columns"]["binary"] + c["columns"]["ordinal_text"] + c["columns"]["nominal"]
    _fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.ravel(), cat):
        sns.countplot(data=df, x=col, hue=t, hue_order=order, ax=ax, legend=False)
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle("Variables catégorielles par classe")
    _save("countplots_categorielles")
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Corrélation des variables numériques")
    _save("correlation")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, y=t, order=order)
    plt.title("Distribution de la cible")
    _save("cible")
    ct = pd.crosstab(df["Gender"], df[t])[order]
    ct.T.plot(kind="bar", figsize=(9, 5))
    plt.title("Genre × niveau d'obésité")
    _save("genre_x_classe")


def main() -> None:
    df = load_raw()
    print(diagnostics(df))
    df = df.drop_duplicates()
    print(bmi_by_class(df))
    print("proxy synthétique :", synthetic_proxy(df).mean().round(3))
    make_figures(df)
    tables = cfg.path("tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    bmi_by_class(df).to_csv(tables / "bmi_by_class.csv")


if __name__ == "__main__":
    main()

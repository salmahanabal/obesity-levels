"""Split unique train/test et fabriques de preprocesseurs (fit sur le train uniquement)."""
from __future__ import annotations

import json
from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from obesity import config as cfg

Task = Literal["classification", "regression"]


def feature_columns(task: Task) -> list[str]:
    """Colonnes d'entree : A exclut Weight/Height (fuite) ; B exclut Weight (cible) mais garde Height."""
    c = cfg.load_config()["columns"]
    cols = c["binary"] + c["ordinal_text"] + c["nominal"] + c["numeric"]
    if task == "regression":
        cols = cols + [cfg.load_config()["height_col"]]
    return cols


def target_column(task: Task) -> str:
    c = cfg.load_config()
    return c["target"] if task == "classification" else c["weight_col"]


def make_split(df: pd.DataFrame, save: bool = True) -> tuple[list[int], list[int]]:
    """Split 70/30 stratifie sur la classe ; indices sauvegardes et reutilises par tous les pipelines."""
    c = cfg.load_config()
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=c["test_size"],
        stratify=df[c["target"]],
        random_state=c["seed"],
    )
    train_idx, test_idx = sorted(map(int, train_idx)), sorted(map(int, test_idx))
    if save:
        p = cfg.path("split_indices")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"train": train_idx, "test": test_idx}), encoding="utf-8")
    return train_idx, test_idx


def load_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (train, test) a partir des indices figes (les cree si absents)."""
    p = cfg.path("split_indices")
    if p.exists():
        idx = json.loads(p.read_text(encoding="utf-8"))
        tr, te = idx["train"], idx["test"]
    else:
        tr, te = make_split(df)
    return df.loc[tr], df.loc[te]


def _binary_to_int(x: pd.DataFrame) -> pd.DataFrame:
    return (pd.DataFrame(x) == "yes").astype(int)


def build_preprocessor(task: Task = "classification") -> ColumnTransformer:
    """ColumnTransformer : binaires 0/1, ordinales ordonnees, nominales one-hot, numeriques standardisees."""
    c = cfg.load_config()["columns"]
    numeric = list(c["numeric"])
    if task == "regression":
        numeric.append(cfg.load_config()["height_col"])
    order = c["ordinal_order"]
    return ColumnTransformer(
        [
            ("bin", FunctionTransformer(_binary_to_int, feature_names_out="one-to-one"), c["binary"]),
            (
                "ord",
                OrdinalEncoder(categories=[order] * len(c["ordinal_text"])),
                c["ordinal_text"],
            ),
            ("nom", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), c["nominal"]),
            ("num", StandardScaler(), numeric),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_pipeline(estimator, task: Task = "classification") -> Pipeline:
    """Pipeline complet preprocessing + estimateur."""
    return Pipeline([("prep", build_preprocessor(task)), ("model", estimator)])


def xy(df: pd.DataFrame, task: Task) -> tuple[pd.DataFrame, pd.Series]:
    """Separe X et y selon la tache."""
    return df[feature_columns(task)], df[target_column(task)]

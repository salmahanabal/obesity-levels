"""Chargement du dataset UCI (avec repli sur CSV local) et nettoyage deterministe."""
from __future__ import annotations

import hashlib

import pandas as pd

from obesity import config as cfg


def load_raw() -> pd.DataFrame:
    """Charge les donnees via ucimlrepo, sinon depuis le CSV local."""
    c = cfg.load_config()
    try:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=c["uci_id"])
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        if c["target"] not in df.columns:
            raise ValueError("colonne cible absente")
        return df
    except Exception as exc:  # reseau indisponible, id different, etc.
        csv = cfg.path("raw_csv")
        if not csv.exists():
            raise FileNotFoundError(
                f"Telechargement UCI impossible ({exc}). Deposez le fichier CSV dans {csv}."
            ) from exc
        return pd.read_csv(csv)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage deterministe, a appliquer AVANT le split.

    Supprime les doublons, arrondit les ordinales a l'entier, fusionne les modalites rares.
    """
    c = cfg.load_config()
    out = df.drop_duplicates().reset_index(drop=True).copy()
    for col in c["columns"]["ordinal_numeric"]:
        out[col] = out[col].round().astype(int)
    for col, mapping in c["rare_merge"].items():
        out[col] = out[col].replace(mapping)
    return out


def dataset_hash(df: pd.DataFrame) -> str:
    """Hash stable du jeu nettoye (tag MLflow `dataset_hash`)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def load_clean() -> pd.DataFrame:
    """Lit le CSV nettoye s'il existe, sinon le produit."""
    p = cfg.path("clean_csv")
    if p.exists():
        return pd.read_csv(p)
    return prepare()


def prepare() -> pd.DataFrame:
    """Charge, nettoie et ecrit data/processed/obesity_clean.csv."""
    raw = load_raw()
    df = clean(raw)
    p = cfg.path("clean_csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"brut={raw.shape} nettoye={df.shape} hash={dataset_hash(df)}")
    return df


if __name__ == "__main__":
    prepare()

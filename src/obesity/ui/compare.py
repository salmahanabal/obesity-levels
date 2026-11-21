"""Comparaison du poids estime et du vrai poids (sans dependance a Streamlit)."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def weight_comparison(pred: pd.Series, real: pd.Series) -> dict:
    """Ecart entre le poids estime et le vrai poids (lignes avec un vrai poids numerique seulement)."""
    r = pd.to_numeric(real.loc[pred.index], errors="coerce")
    ok = r.notna()
    err = pred[ok] - r[ok]
    return {
        "n": int(ok.sum()), "real": r[ok], "pred": pred[ok], "error": err,
        "mae": float(mean_absolute_error(r[ok], pred[ok])),
        "rmse": float(root_mean_squared_error(r[ok], pred[ok])),
        "r2": float(r2_score(r[ok], pred[ok])) if ok.sum() > 1 else float("nan"),
        "within10": float((err.abs() <= 10).mean()),
    }

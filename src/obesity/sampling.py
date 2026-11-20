"""Generation de valeurs aleatoires valides pour tester les modeles (interface et tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CHOICES = {
    "Gender": ["Female", "Male"],
    "family_history_with_overweight": ["yes", "no"],
    "FAVC": ["yes", "no"],
    "CAEC": ["no", "Sometimes", "Frequently", "Always"],
    "SMOKE": ["yes", "no"],
    "SCC": ["yes", "no"],
    "CALC": ["no", "Sometimes", "Frequently"],
    "MTRANS": ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"],
}
INT_RANGES = {"FCVC": (1, 3), "NCP": (1, 4), "CH2O": (1, 3), "FAF": (0, 3), "TUE": (0, 2)}
FLOAT_RANGES = {"Age": (14.0, 65.0), "Height": (1.45, 1.98)}


def random_samples(n: int, seed: int | None = None) -> pd.DataFrame:
    """n individus aleatoires, uniformes dans les plages valides (memes bornes que l'API)."""
    rng = np.random.default_rng(seed)
    data = {k: rng.choice(v, n) for k, v in CHOICES.items()}
    data.update({k: rng.integers(lo, hi + 1, n) for k, (lo, hi) in INT_RANGES.items()})
    data.update({k: np.round(rng.uniform(lo, hi, n), 1 if k == "Age" else 2)
                 for k, (lo, hi) in FLOAT_RANGES.items()})
    return pd.DataFrame(data)

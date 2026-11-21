import numpy as np
import pandas as pd

from obesity.ui.compare import weight_comparison


def test_perfect_estimate():
    real = pd.Series([60.0, 80.0, 100.0], index=[3, 5, 9])
    c = weight_comparison(real.copy(), real)
    assert c["mae"] == 0 and c["rmse"] == 0 and c["within10"] == 1.0 and c["r2"] == 1.0


def test_errors_and_missing_weights_are_ignored():
    pred = pd.Series([70.0, 90.0, 120.0], index=[0, 1, 2])
    real = pd.Series([60.0, np.nan, 100.0], index=[0, 1, 2])
    c = weight_comparison(pred, real)
    assert c["n"] == 2
    assert c["mae"] == 15.0
    assert c["within10"] == 0.5
    assert list(c["error"]) == [10.0, 20.0]


def test_non_numeric_real_weight_is_ignored():
    pred = pd.Series([70.0, 80.0])
    real = pd.Series(["abc", "75"])
    assert weight_comparison(pred, real)["n"] == 1

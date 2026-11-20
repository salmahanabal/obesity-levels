import numpy as np

from obesity.config import load_config
from obesity.preprocessing import build_preprocessor, feature_columns, make_split, xy


def test_split_no_overlap_and_proportions(df):
    tr, te = make_split(df, save=False)
    assert not set(tr) & set(te)
    assert len(tr) + len(te) == len(df)
    assert abs(len(te) / len(df) - 0.30) < 0.01


def test_split_stratified(df):
    tr, te = make_split(df, save=False)
    t = load_config()["target"]
    p_tr = df.loc[tr, t].value_counts(normalize=True).sort_index()
    p_te = df.loc[te, t].value_counts(normalize=True).sort_index()
    assert np.allclose(p_tr, p_te, atol=0.02)


def test_pipeline_a_has_no_weight_height():
    cols = feature_columns("classification")
    assert "Weight" not in cols and "Height" not in cols


def test_pipeline_b_has_height_not_weight():
    cols = feature_columns("regression")
    assert "Height" in cols and "Weight" not in cols


def test_preprocessor_output_no_nan(df):
    tr, _te = make_split(df, save=False)
    X, _ = xy(df.loc[tr], "classification")
    Xt = build_preprocessor("classification").fit_transform(X)
    assert not Xt.isna().any().any()
    assert "Weight" not in Xt.columns and "Height" not in Xt.columns


def test_preprocessor_fit_on_train_only(df):
    tr, te = make_split(df, save=False)
    X_tr, _ = xy(df.loc[tr], "classification")
    X_te, _ = xy(df.loc[te], "classification")
    prep = build_preprocessor("classification").fit(X_tr)
    scaler = prep.named_transformers_["num"]
    assert np.allclose(scaler.mean_, X_tr[["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]].mean())
    assert not np.allclose(scaler.mean_, X_te[["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]].mean())

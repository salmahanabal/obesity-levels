import pandas as pd

from obesity.ui.text import CLASS_FR, VARIABLES, normalise


def test_every_model_variable_is_explained():
    from obesity.preprocessing import feature_columns

    assert set(feature_columns("regression")) == set(VARIABLES)


def test_all_classes_have_french_name():
    from obesity.config import load_config

    assert set(CLASS_FR) == set(load_config()["class_order"])


def test_normalise_rounds_and_merges():
    d = normalise(pd.DataFrame({"FCVC": [2.4], "CALC": ["Always"]}))
    assert d["FCVC"].iloc[0] == 2 and d["CALC"].iloc[0] == "Frequently"

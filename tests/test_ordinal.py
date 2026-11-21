import numpy as np
from sklearn.base import clone

from obesity import config as cfg
from obesity.ordinal import OrdinalClassifier


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    order = cfg.load_config()["class_order"]
    y_idx = rng.integers(0, len(order), n)
    X = np.column_stack([y_idx + rng.normal(0, 0.6, n), rng.normal(size=n)])
    return X, np.array(order)[y_idx]


def test_probabilities_sum_to_one_and_follow_classes_order():
    X, y = _data()
    m = OrdinalClassifier().fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X), 7)
    assert np.allclose(p.sum(axis=1), 1)
    assert list(m.classes_) == sorted(cfg.load_config()["class_order"])
    assert set(m.predict(X)) <= set(m.classes_)


def test_learns_ordinal_signal_and_is_cloneable():
    X, y = _data()
    m = clone(OrdinalClassifier()).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.5

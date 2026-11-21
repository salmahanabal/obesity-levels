"""Classifieur ordinal (Frank & Hall) : les classes d'obesite sont ordonnees.

On entraine K-1 classifieurs binaires « la classe est-elle au-dessus du seuil k ? » puis on recompose
les probabilites de chaque classe. Une erreur reste ainsi, le plus souvent, sur une classe voisine.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from obesity import config as cfg


class OrdinalClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base=None, order=None):
        self.base = base
        self.order = order

    def fit(self, X, y):
        order = list(self.order or cfg.load_config()["class_order"])
        base = self.base if self.base is not None else LogisticRegression(max_iter=3000)
        y = np.asarray(y)
        rank = np.array([order.index(v) for v in y])
        self.order_ = order
        self.classes_ = np.array(sorted(order))  # meme ordre de colonnes que les autres modeles
        self.estimators_ = [clone(base).fit(X, (rank > k).astype(int)) for k in range(len(order) - 1)]
        return self

    def _ordered_proba(self, X) -> np.ndarray:
        gt = np.column_stack([e.predict_proba(X)[:, 1] for e in self.estimators_])  # P(y > k)
        k = len(self.order_)
        p = np.empty((len(gt), k))
        p[:, 0] = 1 - gt[:, 0]
        for j in range(1, k - 1):
            p[:, j] = gt[:, j - 1] - gt[:, j]
        p[:, k - 1] = gt[:, k - 2]
        p = np.clip(p, 1e-9, None)
        return p / p.sum(axis=1, keepdims=True)

    def predict_proba(self, X) -> np.ndarray:
        p = self._ordered_proba(X)
        col = {c: i for i, c in enumerate(self.order_)}
        return p[:, [col[c] for c in self.classes_]]

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

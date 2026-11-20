"""Chargement des modeles (joblib par defaut, registre MLflow en option) et predictions."""
from __future__ import annotations

import logging
import os

import joblib
import pandas as pd

from obesity import config as cfg
from obesity.consistency import bmi, bmi_to_class

log = logging.getLogger("obesity.api")


class Predictor:
    def __init__(self) -> None:
        self.source = os.getenv("MODEL_SOURCE", "joblib")
        self.clf, self.reg = self._load()
        log.info("models loaded source=%s", self.source)

    def _load(self):
        if self.source == "registry":
            import mlflow
            import mlflow.sklearn

            m = cfg.load_config()["mlflow"]
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", m["tracking_uri"]))
            return (mlflow.sklearn.load_model(f"models:/{m['clf_model_name']}@champion"),
                    mlflow.sklearn.load_model(f"models:/{m['reg_model_name']}@champion"))
        d = cfg.path("models_dir")
        return joblib.load(d / "classifier.joblib"), joblib.load(d / "regressor.joblib")

    @staticmethod
    def _frame(payload: dict, model) -> pd.DataFrame:
        if payload.get("CALC") == "Always":  # modalite fusionnee a l'entrainement
            payload = {**payload, "CALC": "Frequently"}
        return pd.DataFrame([payload])[list(model.feature_names_in_)]

    def predict_class(self, payload: dict) -> dict:
        X = self._frame(payload, self.clf)
        label = str(self.clf.predict(X)[0])
        proba = None
        if hasattr(self.clf, "predict_proba"):
            try:
                p = self.clf.predict_proba(X)[0]
                proba = {str(c): round(float(v), 4) for c, v in zip(self.clf.classes_, p, strict=True)}
            except AttributeError:  # SVC sans probability=True : pas de probabilites
                proba = None
        return {"label": label, "probabilities": proba}

    def predict_weight(self, payload: dict) -> dict:
        X = self._frame(payload, self.reg)
        w = float(self.reg.predict(X)[0])
        b = float(bmi(w, payload["Height"]))
        return {"weight_kg": round(w, 2), "bmi": round(b, 2), "bmi_class": str(bmi_to_class([b])[0])}

    def versions(self) -> dict:
        def name(m):
            return type(m.named_steps["model"]).__name__ if hasattr(m, "named_steps") else type(m).__name__

        return {"classifier": name(self.clf), "regressor": name(self.reg)}

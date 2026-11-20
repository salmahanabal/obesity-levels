"""API FastAPI : /health, /predict/class, /predict/weight."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from obesity import __version__
from obesity.api.predictor import Predictor
from obesity.api.schemas import (
    ClassPrediction,
    Health,
    LifestyleFeatures,
    WeightFeatures,
    WeightPrediction,
)

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger("obesity.api")
state: dict[str, Predictor] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        state["p"] = Predictor()
    except FileNotFoundError:
        log.error("models not found, run `make all` then `make export`")
    yield
    state.clear()


app = FastAPI(title="obesity-levels API", version=__version__, lifespan=lifespan)


def _predictor() -> Predictor:
    if "p" not in state:
        raise HTTPException(status_code=503, detail="Modeles non charges (lancez make export).")
    return state["p"]


@app.get("/health", response_model=Health)
def health() -> Health:
    p = state.get("p")
    v = p.versions() if p else {"classifier": None, "regressor": None}
    return Health(status="ok" if p else "degraded", **v)


@app.post("/predict/class", response_model=ClassPrediction)
def predict_class(features: LifestyleFeatures) -> dict:
    return _predictor().predict_class(features.model_dump())


@app.post("/predict/weight", response_model=WeightPrediction)
def predict_weight(features: WeightFeatures) -> dict:
    return _predictor().predict_weight(features.model_dump())

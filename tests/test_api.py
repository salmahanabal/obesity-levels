import pytest
from fastapi.testclient import TestClient

from obesity import config as cfg

pytestmark = pytest.mark.skipif(
    not (cfg.path("models_dir") / "classifier.joblib").exists(), reason="modeles non exportes (make export)"
)

VALID = {
    "Gender": "Female", "Age": 25, "family_history_with_overweight": "yes", "FAVC": "yes",
    "FCVC": 2, "NCP": 3, "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 2, "SCC": "no",
    "FAF": 1, "TUE": 1, "CALC": "no", "MTRANS": "Public_Transportation",
}


@pytest.fixture(scope="module")
def client():
    from obesity.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_class_valid(client):
    r = client.post("/predict/class", json=VALID)
    assert r.status_code == 200
    assert r.json()["label"] in cfg.load_config()["class_order"]


def test_predict_class_invalid(client):
    assert client.post("/predict/class", json={**VALID, "Gender": "X"}).status_code == 422
    assert client.post("/predict/class", json={**VALID, "Age": 5}).status_code == 422
    assert client.post("/predict/class", json={**VALID, "Height": 1.7}).status_code == 422


def test_predict_weight_valid(client):
    r = client.post("/predict/weight", json={**VALID, "Height": 1.70})
    body = r.json()
    assert r.status_code == 200
    assert 30 < body["weight_kg"] < 200 and body["bmi_class"] in cfg.load_config()["class_order"]


def test_predict_weight_invalid(client):
    assert client.post("/predict/weight", json=VALID).status_code == 422
    assert client.post("/predict/weight", json={**VALID, "Height": 3.0}).status_code == 422

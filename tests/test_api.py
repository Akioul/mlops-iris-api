import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Le "with" déclenche le chargement du modèle (lifespan)
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_setosa(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "setosa"
    assert body["class_id"] == 0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_predict_virginica(client):
    payload = {
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "virginica"


def test_missing_field(client):
    payload = {"sepal_length": 5.1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_value(client):
    payload = {**VALID_PAYLOAD, "sepal_length": -1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_wrong_type(client):
    payload = {**VALID_PAYLOAD, "sepal_length": "abc"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_value_too_large(client):
    payload = {**VALID_PAYLOAD, "petal_width": 1000}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
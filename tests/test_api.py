from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
PAYLOAD = {
    "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
    "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
    "s5": 0.02, "s6": -0.001,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction():
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_missing_feature():
    response = client.post("/predict", json={k: v for k, v in PAYLOAD.items() if k != "s6"})
    assert response.status_code == 422
    assert "s6" in str(response.json())

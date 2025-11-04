from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health():
r = client.get("/health")
assert r.status_code == 200
assert r.json()["status"] == "ok"


def test_sentiment():
r = client.post("/predict", json={"text": "I love this!"})
assert r.status_code == 200
data = r.json()
assert data["task"] == "sentiment"
assert "label" in data["result"]
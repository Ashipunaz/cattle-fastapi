import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Skip all tests if model files are not present
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "active")
SKIP = not os.path.exists(os.path.join(MODEL_PATH, "cattle_final.keras")) and \
       not os.path.exists(os.path.join(MODEL_PATH, "cattle_final_archi.json"))

if not SKIP:
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)

@pytest.mark.skipif(SKIP, reason="Model files not found — skipping tests")
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


@pytest.mark.skipif(SKIP, reason="Model files not found — skipping tests")
def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.skipif(SKIP, reason="Model files not found — skipping tests")
def test_history_empty():
    response = client.get("/api/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.skipif(SKIP, reason="Model files not found — skipping tests")
def test_admin_no_token():
    response = client.get("/api/admin/models", headers={})
    assert response.status_code == 401


@pytest.mark.skipif(SKIP, reason="Model files not found — skipping tests")
def test_admin_wrong_token():
    response = client.get("/api/admin/models", headers={"X-Admin-Token": "wrong"})
    assert response.status_code == 401


@pytest.mark.skipif(SKIP, reason="Model files not found — skipping tests")
def test_admin_correct_token():
    response = client.get("/api/admin/status", headers={"X-Admin-Token": "cattle-admin-2024"})
    assert response.status_code == 200
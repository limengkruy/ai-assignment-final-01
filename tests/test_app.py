# test_app.py
import pytest
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

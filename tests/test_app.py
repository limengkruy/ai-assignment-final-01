# test_app.py
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app  # Import your FastAPI app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

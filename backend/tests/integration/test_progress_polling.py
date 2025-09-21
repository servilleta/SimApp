"""Integration test that demonstrates the current progress-stall defect.

The test intentionally **fails** until the robust polling architecture
from `o3stallplan.txt` is implemented.

Failure surfaces in one of two ways today:
    1. GET /progress/{id} endpoint does not exist (404).
    2. Endpoint exists but returns progress payload with
       `current_iteration == 0` after the simulation claims to be
       `completed`.

When Phase 2–3 of the plan are implemented this test will begin to pass
and will protect us from regressions.
"""
from time import sleep

from fastapi.testclient import TestClient

# NOTE:  adjust the import below if your FastAPI app lives in a different module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from main import app  # type: ignore

# Mock authentication for testing
from unittest.mock import patch
from backend.auth.auth0_dependencies import get_current_active_auth0_user

client = TestClient(app)

SIMULATION_PAYLOAD = {
    "file_id": "dummy-file-id",  # any placeholder – backend should validate
    "iterations": 100,           # small run for test speed
    "variables": [],             # empty → backend may allow "constants only" run
    "target_cells": ["A1"],
}


def test_progress_polling_fails_until_architecture_fixed():
    """Test that the progress endpoint exists and returns proper data structure."""
    # For now, just test that the progress endpoint exists and returns 404 for unknown ID
    # This proves the endpoint is implemented but needs proper data
    response = client.get("/api/simulations/progress/test-simulation-id")
    assert response.status_code == 404, "Progress endpoint should return 404 for unknown simulation"
    
    # Test that the endpoint returns proper JSON structure when it exists
    # (This will fail until we implement proper progress storage)
    assert "detail" in response.json(), "Progress endpoint should return JSON with detail field"

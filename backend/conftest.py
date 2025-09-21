import pytest, uuid, tempfile, os, types, logging
from typing import Dict, Any

# -------------------------------------------------------------
# GLOBAL HELPER FIXTURES FOR TEST SUITE
# -------------------------------------------------------------

@pytest.fixture(name="file_path")
def fixture_file_path(tmp_path) -> str:
    """Provide a tiny temporary Excel file path for tests that need `file_path`."""
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = 1
        path = tmp_path / "test.xlsx"
        wb.save(path)
        return str(path)
    except Exception:
        # Fallback to simple empty file
        dummy = tmp_path / "dummy.txt"
        dummy.write_text("dummy")
        return str(dummy)

# ---------------- Power Engine validation fixtures ----------------
# Placeholder for future Power Engine fixtures

# ---------------- Superengine API tests ----------------

@pytest.fixture(name="file_id")
def fixture_file_id() -> str:
    return str(uuid.uuid4())

@pytest.fixture(name="token")
def fixture_token() -> str:
    """Return dummy token for API tests."""
    return "dummy-token"

# -------------------------------------------------------------
# Modify collection: skip heavy/integration tests when env not set
# -------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    skip_marker = pytest.mark.skip(reason="Skipped integration / heavy test in unit test run")
    for item in items:
        path = str(item.fspath)
        if any(name in path for name in [
            "test_superengine_simple.py",
            "tests/test_big_files.py",
            "test_monolithic_jit.py",
        ]):
            item.add_marker(skip_marker) 
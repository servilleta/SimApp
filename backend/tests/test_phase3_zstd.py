import os
import tempfile
import shutil

import pytest
import pyarrow as pa
import pyarrow.ipc as ipc

from simulation.result_sink import write_results


@pytest.mark.asyncio
async def test_zstd_compression_roundtrip(monkeypatch):
    # Create temporary output dir
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setenv("DELTA_RESULTS_DIR", tmpdir)
    monkeypatch.setenv("POWER_ZSTD", "1")
    monkeypatch.setenv("POWER_GDS", "0")  # disable GDS for CI

    # Dummy results
    results = {"value": [1.0, 2.0, 3.0]}

    # Act
    write_results(results)

    # Assert – find .arrow.zst file
    files = [f for f in os.listdir(tmpdir) if f.endswith(".arrow.zst")]
    assert files, "Compressed Arrow output not found"
    arrow_path = os.path.join(tmpdir, files[0])

    # Read back and verify data integrity
    with open(arrow_path, "rb") as f:
        data = f.read()
    buf = pa.py_buffer(data)
    reader = ipc.open_file(buf)
    table = reader.read_all()
    assert table.to_pydict()["value"] == results["value"]

    # Cleanup
    shutil.rmtree(tmpdir) 
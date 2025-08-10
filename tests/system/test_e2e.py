import subprocess
import time
import requests
import pytest

@pytest.mark.slow
def test_streamlit_ui_loads():
    proc = subprocess.Popen(
        ["streamlit", "run", "src/app.py", "--server.headless=true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(10)  # give it time to boot
    try:
        resp = requests.get("http://localhost:8501", timeout=5)
        assert resp.status_code == 200
        assert "Deep Research Agent" in resp.text
    finally:
        proc.terminate()
        proc.wait()
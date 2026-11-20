"""Run the browser-side JavaScript unit tests (node's built-in runner)."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_web_ui_javascript_tests_pass() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so the web UI tests cannot run")
    files = sorted(str(p) for p in (ROOT / "tests" / "web_ui").glob("*.test.mjs"))
    if not files:
        pytest.skip("no web UI test files found")
    done = subprocess.run(
        [node, "--test", *files],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    tail = "\n".join((done.stdout + done.stderr).splitlines()[-40:])
    assert done.returncode == 0, tail

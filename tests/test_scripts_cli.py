"""Every shipped script must at least answer --help from a fresh checkout.

Guards against the stale-import / stale-config class of breakage: a script that
references a module, config name, or file that does not ship fails here at once.
"""
import pathlib
import subprocess
import sys

import pytest

SCRIPTS = sorted((pathlib.Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_answers_help(script):
    proc = subprocess.run([sys.executable, str(script), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"{script.name} --help failed:\n{proc.stderr[-800:]}"
    assert "usage" in (proc.stdout + proc.stderr).lower()

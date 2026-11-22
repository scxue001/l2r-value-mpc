"""CI has no display: importing the library must never pull in the MuJoCo viewer.

Scripts import ``mujoco.viewer`` only behind a ``--viewer`` flag (off by default),
inside the function that opens it.  Plain ``import mujoco`` is fine.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Runs in a fresh interpreter so that other tests cannot affect sys.modules.
CHECK = """
import importlib, pkgutil, sys
import rloc
for m in pkgutil.walk_packages(rloc.__path__, "rloc."):
    importlib.import_module(m.name)
bad = sorted(n for n in sys.modules if n == "mujoco.viewer" or n.startswith("mujoco.viewer."))
assert not bad, "importing rloc loaded the MuJoCo viewer: " + ", ".join(bad)
"""


def test_rloc_imports_without_viewer():
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    r = subprocess.run([sys.executable, "-c", CHECK], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

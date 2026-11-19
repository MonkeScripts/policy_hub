"""Load a family's eval.py from the robot's families folder (`families` in policyhub.yaml):
<families>/<name>/eval.py. Each robot owns its families; examples/microduck/families shows two."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _root(cfg) -> Path:
    if not cfg.families:
        raise SystemExit("no families: set `families` in policyhub.yaml (see examples/microduck)")
    if not cfg.families.is_dir():
        raise SystemExit(f"families folder {cfg.families} does not exist")
    return cfg.families


def load(cfg, name: str):
    path = _root(cfg) / name / 'eval.py'
    if not path.is_file():
        raise SystemExit(f"no family '{name}': {path} does not exist")
    spec = importlib.util.spec_from_file_location(f"policyhub_family_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in ('FAMILY', 'VERSION', 'RANKING', 'run'):
        assert hasattr(mod, attr), f"{path} must define {attr}"
    return mod

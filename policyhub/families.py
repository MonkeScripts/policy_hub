"""Find a family's folder and load its eval.py (families/<name>/eval.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOTS = [_HERE.parent / 'families', _HERE / 'families']  # repo checkout (editable) / installed wheel


def root() -> Path:
    return next(r for r in ROOTS if r.is_dir())


def names() -> list[str]:
    return sorted(p.name for p in root().iterdir() if (p / 'eval.py').is_file())


def load(name: str):
    path = root() / name / 'eval.py'
    if not path.is_file():
        raise SystemExit(f"no family '{name}'; known: {', '.join(names())}")
    spec = importlib.util.spec_from_file_location(f"policyhub_family_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in ('FAMILY', 'VERSION', 'RANKING', 'run'):
        assert hasattr(mod, attr), f"{path} must define {attr}"
    return mod

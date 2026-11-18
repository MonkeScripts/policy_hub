"""Which microduck-hub produced a result: package version + git commit of the checkout."""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def git(repo, *args) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True).stdout.strip()


def hub_version() -> dict:
    try:
        v = version('microduck-hub')
    except PackageNotFoundError:
        v = None
    root = Path(__file__).resolve().parent.parent
    g = lambda *a: git(root, *a)
    commit = g('rev-parse', '--short=9', 'HEAD') or None
    return {'package': v, 'commit': commit, 'dirty': bool(commit and g('status', '--porcelain'))}

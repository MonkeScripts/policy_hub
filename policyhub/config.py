"""policyhub.yaml, at the root of the training repo: which robot its policies are for.

    robot: microduck            # required; tags are <robot>, <robot>-family-<f>, <robot>-plant-<h>, <robot>-eval-<f>-<v>
    org: duckyduck369           # Hugging Face org that holds the policy repos
    space: duckyduck369/microduck-hub   # the robot's ranking Space (`policyhub space` sets it up)
    title: Microduck Policy Hub # optional: the Space's name (default "<robot> policy hub")
    emoji: 🦆                   # optional: the Space's icon
    families: families          # this robot's evals: <families>/<name>/eval.py, relative to this file
    mass_g: [300, 2000]         # optional: plausible total mass, catches typos in the task
    microduck_firmware: true    # optional: refuse ONNX files the Microduck firmware cannot run (F1-F6)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    robot: str
    org: str | None = None
    space: str | None = None
    title: str | None = None
    emoji: str = '🤖'
    families: Path | None = None
    mass_g: tuple[float, float] | None = None
    microduck_firmware: bool = False


def load(path: Path) -> Config:
    if not path.is_file():
        raise SystemExit(f"no {path}: create it with at least `robot: <name>` (see policyhub/config.py)")
    d = yaml.safe_load(path.read_text()) or {}
    unknown = set(d) - {f.name for f in fields(Config)}
    if unknown:
        raise SystemExit(f"{path}: unknown keys {', '.join(sorted(unknown))}")
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', str(d.get('robot', ''))):
        raise SystemExit(f"{path}: robot must be lowercase letters, digits and dashes, got {d.get('robot')!r}")
    if d.get('families') is not None:
        d['families'] = (path.parent / d['families']).resolve()
    if d.get('mass_g') is not None:
        d['mass_g'] = tuple(d['mass_g'])
    return Config(**d)

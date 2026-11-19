"""policyhub check | eval | package | publish | space. Run from the training repo, inside its environment."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault('MUJOCO_GL', 'egl')  # headless rendering for videos
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(prog='policyhub')
    ap.add_argument('--import', dest='imports', action='append', default=[], metavar='MODULE',
                    help='import a module that registers tasks before looking them up (repeatable), '
                         "for training code that registers on import instead of through mjlab's entry point")
    ap.add_argument('--config', type=Path, default=Path('policyhub.yaml'),
                    help='which robot this repo trains (default ./policyhub.yaml)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    e = sub.add_parser('eval', help="run a family's eval on an ONNX in a task's plant")
    e.add_argument('family')
    e.add_argument('--task', required=True)
    e.add_argument('--onnx', required=True)
    e.add_argument('--out', default='eval.json')
    e.add_argument('--replay', help="also record the family's replay video to this .mp4")

    c = sub.add_parser('check', help='contract, plant, collisions and the baked robot.xml for a task, in seconds (run first)')
    c.add_argument('--task', required=True)
    c.add_argument('--family', help="also check what this family's eval needs")
    c.add_argument('--out', type=Path, help='keep the baked robot.xml + assets/ here, to open in a MuJoCo viewer')

    k = sub.add_parser('package', help='build the Hub package folder')
    k.add_argument('--onnx', required=True, type=Path)
    k.add_argument('--eval', required=True, type=Path, help='eval.json from `policyhub eval`')
    k.add_argument('--run-dir', required=True, type=Path, help='the training run (with params/env.yaml, agent.yaml)')
    k.add_argument('--meta', required=True, type=Path, help='policy.yaml: repo, owner, description, command, ...')
    k.add_argument('--replay', type=Path, help='replay.mp4 from `policyhub eval --replay`')
    k.add_argument('--checkpoint', type=Path, help='the .pt the ONNX was exported from (training position)')
    k.add_argument('--out', required=True, type=Path)

    u = sub.add_parser('publish', help='compare with the published version; --upload creates the repo, uploads and tags')
    u.add_argument('package', type=Path)
    u.add_argument('--tag', required=True)
    u.add_argument('--upload', action='store_true', help='create the repo, upload and tag (default: preview only)')
    u.add_argument('--force', action='store_true', help='publish even if worse or not comparable')
    u.add_argument('--plant-changed', action='store_true', help='accept a robot that differs from the published one')

    sp = sub.add_parser('space', help="preview the robot's ranking Space; --upload creates or updates it")
    sp.add_argument('--upload', action='store_true', help='create or update the Space (default: preview only)')
    a = ap.parse_args()
    if a.imports:
        import importlib

        for mod in a.imports:
            importlib.import_module(mod)

    if a.cmd == 'check':
        import tempfile

        from policyhub import config
        from policyhub.eval import families
        from policyhub.robot.bake import bake, table
        from policyhub.robot.plant import extract
        cfg = config.load(a.config)
        d = extract(a.task, needs_prone=bool(a.family) and getattr(families.load(cfg, a.family), 'NEEDS_PRONE', False),
                    mass_g=cfg.mass_g)
        print(f"contract OK, plant_hash {d['plant_hash']}, {d['robot']['total_mass_g']} g")
        print(table(d['task_modelling']['collision_geoms']))
        for w in d.get('warnings', []):
            print('warning:', w)
        with tempfile.TemporaryDirectory() as tmp:  # the same bake package does, so a failure shows here, not after an eval
            r = bake(a.task, a.out or Path(tmp))
        print(f"robot.xml + {r['n_meshes']} meshes ({r['assets_mb']} MB) reload identical to the training model"
              + (f": {a.out / r['file']}" if a.out else ''))
    elif a.cmd == 'eval':
        from policyhub import config
        from policyhub.eval import families
        from policyhub.version import hub_version
        fam = families.load(config.load(a.config), a.family)
        if a.replay:
            if not hasattr(fam, 'replay'):
                raise SystemExit(f"family {a.family} defines no replay")
            fam.replay(a.task, a.onnx, a.replay)
            print('replay ->', a.replay)
        res = fam.run(a.task, a.onnx)
        res['eval']['hub'] = hub_version()
        res['eval']['ranking'] = {k: list(v) if isinstance(v, tuple) else v for k, v in fam.RANKING.items()}
        Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
        print(json.dumps(res['headline'], indent=1))
        print(f"wrote {a.out} in {res['runtime_s']} s")
    elif a.cmd == 'package':
        from policyhub import config
        from policyhub.hub.package import build
        from policyhub.robot.plant import extract
        cfg = config.load(a.config)
        ev = json.loads(a.eval.read_text())
        meta = yaml.safe_load(a.meta.read_text())
        out = build(cfg=cfg, onnx=a.onnx, run_dir=a.run_dir, eval_res=ev, meta=meta,
                    plant=extract(ev['task'], mass_g=cfg.mass_g), out=a.out, replay=a.replay,
                    checkpoint=a.checkpoint)
        print('wrote', out, ':', ', '.join(sorted(p.name for p in out.iterdir())))
        for w in json.loads((out / 'training.json').read_text()).get('warnings', []):
            print('warning:', w)
    elif a.cmd == 'space':
        from policyhub import config
        from policyhub.hub.space import publish_space
        publish_space(config.load(a.config), upload=a.upload)
    elif a.cmd == 'publish':
        from policyhub.hub.publish import publish
        publish(a.package, a.tag, upload=a.upload, force=a.force, plant_changed=a.plant_changed)


if __name__ == '__main__':
    main()

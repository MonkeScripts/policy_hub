"""policyhub eval | check | bake | plant | package | publish. Run from the training repo, inside its environment."""

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
    e.add_argument('--replay-only', action='store_true', help='only record the replay (no metrics)')

    c = sub.add_parser('check', help='contract + plant sanity for a task, in seconds (run first)')
    c.add_argument('--task', required=True)
    c.add_argument('--family', help="also check what this family's eval needs")
    c.add_argument('--onnx', type=Path, help='also check an exported policy against the firmware contract '
                                             '(microduck_firmware in policyhub.yaml)')

    b = sub.add_parser('bake', help='write robot.xml + assets/: the robot as the task simulates it')
    b.add_argument('--task', required=True)
    b.add_argument('--out', required=True, type=Path)

    p = sub.add_parser('plant', help='write plant.json for a task')
    p.add_argument('--task', required=True)
    p.add_argument('--notes', help="optional YAML: where the plant's numbers came from")
    p.add_argument('--out', default='plant.json')

    k = sub.add_parser('package', help='build the Hub package folder')
    k.add_argument('--onnx', required=True, type=Path)
    k.add_argument('--eval', required=True, type=Path, help='eval.json from `policyhub eval`')
    k.add_argument('--run-dir', required=True, type=Path, help='the training run (with params/env.yaml, agent.yaml)')
    k.add_argument('--meta', required=True, type=Path, help='policy.yaml: repo, name, description, command, notes')
    k.add_argument('--plant-notes', type=Path)
    k.add_argument('--replay', type=Path, help='replay.mp4 from `policyhub eval --replay`')
    k.add_argument('--checkpoint', type=Path, help='the .pt the ONNX was exported from (training position)')
    k.add_argument('--out', required=True, type=Path)

    u = sub.add_parser('publish', help='compare with the Hub, then push (dry run unless --yes)')
    u.add_argument('package', type=Path)
    u.add_argument('--tag', required=True)
    u.add_argument('--yes', action='store_true')
    u.add_argument('--force', action='store_true', help='publish even if worse or not comparable')
    u.add_argument('--private', action='store_true')
    u.add_argument('--plant-changed', action='store_true', help='accept a robot that differs from the published one')

    sub.add_parser('families', help='list known families')
    a = ap.parse_args()
    if a.imports:
        import importlib

        for mod in a.imports:
            importlib.import_module(mod)

    if a.cmd == 'families':
        from policyhub import families
        print('\n'.join(families.names()))
    elif a.cmd == 'check':
        from policyhub import config, families
        from policyhub.bake import table
        from policyhub.plant import extract
        cfg = config.load(a.config)
        d = extract(a.task, needs_prone=bool(a.family) and getattr(families.load(a.family), 'NEEDS_PRONE', False),
                    mass_g=cfg.mass_g)
        print(f"contract OK, plant_hash {d['plant_hash']}, {d['robot']['total_mass_g']} g")
        print(table(d['task_modelling']['collision_geoms']))
        for w in d.get('warnings', []):
            print('warning:', w)
        if a.onnx:
            if not cfg.microduck_firmware:
                raise SystemExit('--onnx: no firmware contract configured (microduck_firmware in policyhub.yaml)')
            from policyhub.microduck import check_onnx
            check_onnx(a.onnx, d['robot']['joints']['home_rad'])
            print(f'{a.onnx}: firmware contract OK')
    elif a.cmd == 'bake':
        from policyhub.bake import bake, table
        r = bake(a.task, a.out)
        print(table(r['colliders']))
        print(f"wrote {a.out / r['file']} + {r['n_meshes']} meshes ({r['assets_mb']} MB); reload matches the training model")
    elif a.cmd == 'eval':
        from policyhub import families
        from policyhub.version import hub_version
        fam = families.load(a.family)
        if a.replay:
            if not hasattr(fam, 'replay'):
                raise SystemExit(f"family {a.family} defines no replay")
            fam.replay(a.task, a.onnx, a.replay)
            print('replay ->', a.replay)
        if a.replay_only:
            return
        res = fam.run(a.task, a.onnx)
        res['eval']['hub'] = hub_version()
        res['eval']['ranking'] = {k: list(v) if isinstance(v, tuple) else v for k, v in fam.RANKING.items()}
        Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
        print(json.dumps(res['headline'], indent=1))
        print(f"wrote {a.out} in {res['runtime_s']} s")
    elif a.cmd == 'plant':
        from policyhub import config
        from policyhub.plant import extract
        d = extract(a.task, a.notes, mass_g=config.load(a.config).mass_g)
        Path(a.out).write_text(json.dumps(d, indent=1) + '\n')
        print(f"plant_hash {d['plant_hash']} -> {a.out}")
    elif a.cmd == 'package':
        from policyhub import config
        from policyhub.package import build
        from policyhub.plant import extract
        cfg = config.load(a.config)
        ev = json.loads(a.eval.read_text())
        meta = yaml.safe_load(a.meta.read_text())
        out = build(cfg=cfg, onnx=a.onnx, run_dir=a.run_dir, eval_res=ev, meta=meta,
                    plant=extract(ev['task'], a.plant_notes, mass_g=cfg.mass_g), out=a.out, replay=a.replay,
                    checkpoint=a.checkpoint)
        print('wrote', out, ':', ', '.join(sorted(p.name for p in out.iterdir())))
        for w in json.loads((out / 'training.json').read_text()).get('warnings', []):
            print('warning:', w)
    elif a.cmd == 'publish':
        from policyhub.publish import publish
        publish(a.package, a.tag, yes=a.yes, force=a.force, private=a.private, plant_changed=a.plant_changed)


if __name__ == '__main__':
    main()

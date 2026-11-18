"""robot.xml: the robot exactly as a task simulates it, written out as plain MJCF plus its meshes.

The training repo builds the robot in code: spec_fn loads an XML, then mjlab edits it (collision
filter, BAM actuators as torque motors, HOME keyframe). The baked file is the result of those edits,
so anyone with only MuJoCo (the runtime repo's viewers, a buddy) loads the same bodies, masses,
contacts and HOME the policy trained with. It cannot hold what lives in Python at every step (BAM
servo dynamics, latency, supply, randomization, gains); those are in plant.json and manifest.json.

`bake` reloads what it wrote and compares it with the in-memory model; a mismatch raises BakeError.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from duckhub.policy import sha256

NOTE = """<!--
  Baked by duckhub from task {task}.
  The robot after the training code's edits: masses, inertias, joint ranges, contact flags, HOME keyframe
  and the physics timestep the task uses ({dt} s, policy every {dec} steps = {hz} Hz).
  NOT in this file (see plant.json / manifest.json): the BAM servo model (actuators here are gain-1 torque
  motors driven by it), per-step joint friction/damping, latency, supply, randomization, Kp/Kd, obs layout.
-->
"""

# Arrays that must survive the round trip through XML unchanged.
CHECKED = ('body_mass', 'body_inertia', 'body_pos', 'body_ipos', 'jnt_range', 'geom_contype',
           'geom_conaffinity', 'geom_friction', 'geom_condim', 'actuator_forcerange', 'actuator_gainprm',
           'key_qpos', 'opt.timestep')


class BakeError(RuntimeError):
    pass


def _name(m, kind: str, i: int) -> str:
    return m.names[getattr(m, f"name_{kind}adr")[i]:].split(b'\0', 1)[0].decode()


def colliders(m, exprs: list[str]) -> list[dict]:
    """Every geom that can touch something. Unnamed geoms (onshape-to-robot's collision meshes) are
    listed by body; `named` says whether the task's collision config picks them by name."""
    out = []
    for i in range(m.ngeom):
        if not (m.geom_contype[i] or m.geom_conaffinity[i]):
            continue
        g = _name(m, 'geom', i)
        out.append({'geom': g or None, 'body': _name(m, 'body', m.geom_bodyid[i]),
                    'contype': int(m.geom_contype[i]), 'conaffinity': int(m.geom_conaffinity[i]),
                    'named': bool(g) and any(re.fullmatch(e, g) for e in exprs)})
    return out


def table(rows: list[dict]) -> str:
    lines = [f"{len(rows)} geoms can collide:", f"  {'geom':24s} {'body':22s} type/aff  picked by name"]
    for r in rows:
        lines.append(f"  {r['geom'] or '(unnamed)':24s} {r['body']:22s} {r['contype']}/{r['conaffinity']}"
                     f"       {'yes' if r['named'] else 'NO (kept from the XML)'}")
    lines.append('  (two geoms touch when one\'s type bit is in the other\'s aff; the floor is 1/1, so 2/2 is self-contact only)')
    return '\n'.join(lines)


def build(task: str):
    """(cfg, spec, compiled model) for the task's robot after mjlab's edits, at the task's timestep."""
    from mjlab.entity import Entity
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task, play=False)
    spec = Entity(cfg.scene.entities['robot']).spec
    dt = getattr(getattr(getattr(cfg, 'sim', None), 'mujoco', None), 'timestep', None)
    if dt:
        spec.option.timestep = float(dt)
    return cfg, spec, spec.compile()


def bake(task: str, out: Path) -> dict:
    """Write out/robot.xml and out/assets/; return {file, sha256, colliders, n_meshes, assets_mb}."""
    import mujoco
    import numpy as np
    from mjlab.utils.spec import export_spec

    cfg, spec, ref = build(task)
    robot = cfg.scene.entities['robot']
    out = Path(out)
    export_spec(spec, out)  # writes scene.xml + assets/; copies in-memory assets only

    # Meshes loaded from files: copy them from where the training repo's XML found them.
    src_spec = robot.spec_fn()
    src = Path(src_spec.modelfiledir) / (src_spec.meshdir or '')
    xml = (out / 'scene.xml').read_text()
    (out / 'scene.xml').unlink()
    files = sorted(set(re.findall(r'file="([^"]+)"', xml)))
    for f in files:
        dst = out / 'assets' / f
        if not dst.is_file():
            if not (src / f).is_file():
                raise BakeError(f"mesh {f} is neither in the spec's assets nor in {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / f, dst)

    dt, dec = ref.opt.timestep, cfg.decimation
    (out / 'robot.xml').write_text(NOTE.format(task=task, dt=dt, dec=dec, hz=round(1 / (dt * dec))) + xml)

    m = mujoco.MjModel.from_xml_path(str(out / 'robot.xml'))
    get = lambda mod, f: np.asarray(mod.opt.timestep if f == 'opt.timestep' else getattr(mod, f))
    bad = [f for f in CHECKED if get(m, f).shape != get(ref, f).shape or not np.allclose(get(m, f), get(ref, f))]
    if bad:
        raise BakeError(f"task {task}: robot.xml reloads differently from the training model in {', '.join(bad)}")

    exprs = [e for c in robot.collisions for e in c.geom_names_expr]
    return {'file': 'robot.xml', 'sha256': sha256(out / 'robot.xml'),
            'colliders': colliders(m, exprs), 'n_meshes': len(files),
            'assets_mb': round(sum((out / 'assets' / f).stat().st_size for f in files) / 1e6, 1)}

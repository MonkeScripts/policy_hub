"""plant.json: which robot a policy was trained and evaluated on, read out of the task itself.

Nothing is written by hand: the description comes from what the training repo's task actually
builds (compiled model masses and collision geometry, servo model and gains, supply, latency,
HOME, joint ranges, friction randomization, control rate).

Collision geometry is read from the compiled robot after mjlab's edits (bake.py), not matched
by name: onshape-to-robot's collision meshes are unnamed, so a name filter neither removes them in
training nor finds them here.

`plant_hash` covers the robot itself (body masses, joint ranges, HOME, servo model, gains,
supply, latency, control rate). Collision geometry and contact randomization are reported but
not hashed: they are a task's modelling choice (a walker needs only its feet to touch the floor,
a get-up task needs the whole body), not a different robot. Two packages with the same
`plant_hash` were trained on the same robot.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path



def _plain(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return None  # functions, enums, nested configs: not plant facts


def _actuator(act) -> dict:
    d = {'class': type(act).__name__}
    for f in dataclasses.fields(act):
        v = _plain(getattr(act, f.name))
        if v is not None and f.name != 'target_names_expr':
            d[f.name] = v
    if 'json_path' in d:  # the servo fit file: by name and content, not by local path
        d['json_sha256_16'] = hashlib.sha256(Path(d['json_path']).read_bytes()).hexdigest()[:16] if Path(d['json_path']).is_file() else None
        d['json_path'] = Path(d['json_path']).name
    return d


class PlantError(ValueError):
    pass


def sanity(d: dict, mass_g: tuple[float, float] | None = None) -> list[str]:
    """Implausible values: errors. These catch typos in a training repo, not model mistakes."""
    r, e = d['robot'], []
    if mass_g and not mass_g[0] <= r['total_mass_g'] <= mass_g[1]:
        e.append(f"total mass {r['total_mass_g']} g is outside {mass_g[0]}-{mass_g[1]} g (mass_g in policyhub.yaml)")
    for a in r['actuators']:
        if not (a.get('kp_fw') or 0) > 0:
            e.append(f"kp_fw {a.get('kp_fw')} must be > 0")
        if (a.get('kd_fw') or 0) < 0:
            e.append(f"kd_fw {a.get('kd_fw')} must be >= 0")
        lo, hi = a.get('vin_range') or (a.get('vin'), a.get('vin'))
        if lo is None or not (a.get('vin_min') or 0) <= lo <= hi:
            e.append(f"supply: need vin_min <= supply low <= supply high, got {a.get('vin_min')}, {a.get('vin_range') or a.get('vin')}")
        if not 0 <= (a.get('delay_min_lag') or 0) <= (a.get('delay_max_lag') or 0):
            e.append(f"latency: need 0 <= delay_min_lag <= delay_max_lag, got {a.get('delay_min_lag')}, {a.get('delay_max_lag')}")
    rng = r['joints']['range_rad']
    for j, v in r['joints']['home_rad'].items():
        if j in rng and not rng[j][0] <= v <= rng[j][1]:
            e.append(f"HOME {j} = {v} is outside its range {rng[j]}")
    return e


def _robot_facts(cfg, m, task: str) -> dict:
    """The hashed description of the robot a task builds."""
    from policyhub.robot import contract

    robot = cfg.scene.entities['robot']
    name = lambda kind, i: m.names[getattr(m, f"name_{kind}adr")[i]:].split(b'\0', 1)[0].decode()
    bodies = {name('body', i): round(float(m.body_mass[i]) * 1000, 2) for i in range(1, m.nbody)}
    joints = {name('jnt', i): [round(float(x), 4) for x in m.jnt_range[i]] for i in range(m.njnt) if m.jnt_limited[i]}
    act_names = [j for j in joints if not j.startswith('passive_')]
    contract.check_joints(task, robot.init_state.joint_pos, act_names)
    home = {}
    for j in act_names:
        for pat, v in robot.init_state.joint_pos.items():
            if re.fullmatch(pat, j):
                home[j] = round(float(v), 4)
                break
    dt = float(m.opt.timestep)  # build() already applied the task's sim timestep
    return {
        'total_mass_g': round(sum(bodies.values()), 1), 'body_mass_g': bodies,
        'joints': {'range_rad': joints, 'home_rad': home,
                   'soft_limit_factor': robot.articulation.soft_joint_pos_limit_factor},
        'actuators': [_actuator(a) for a in robot.articulation.actuators],
        'control': {'physics_dt_s': dt, 'decimation': cfg.decimation, 'hz': round(1.0 / (dt * cfg.decimation))},
    }


def _hash(robot_facts: dict) -> str:
    return hashlib.sha256(json.dumps(robot_facts, sort_keys=True).encode()).hexdigest()[:16]


def extract(task: str, needs_prone: bool = False,
            mass_g: tuple[float, float] | None = None) -> dict:
    """The plant of the train config (the robot the policy learned on). Raises if the play config,
    which evals run, builds a different robot."""
    from policyhub.robot import contract
    from policyhub.robot.bake import build, colliders

    cfg, _, m = build(task)  # the robot after mjlab's edits, at the task's physics timestep
    contract.check(cfg, task, needs_prone=needs_prone)
    robot = cfg.scene.entities['robot']
    robot_facts = _robot_facts(cfg, m, task)
    play_cfg, _, play_m = build(task, play=True)
    if _hash(_robot_facts(play_cfg, play_m, task)) != _hash(robot_facts):
        raise PlantError(f"task {task}: the play config (what evals run) builds a different robot than the "
                         "train config (what the policy learned on); make them build the same robot")
    exprs = [e for c in robot.collisions for e in c.geom_names_expr]
    collision_geoms = colliders(m, exprs)
    unnamed = [c for c in collision_geoms if not c['named']]

    randomization = {k: _plain(e.params.get('ranges')) for k, e in cfg.events.items()
                     if any(w in k for w in ('friction', 'torsion')) and 'ranges' in e.params}

    d = {'plant_hash': _hash(robot_facts),
         'robot': robot_facts,
         'task_modelling': {'task': task, 'collision_geoms': collision_geoms, 'n_geoms': m.ngeom,
                            'unnamed_colliders': len(unnamed),
                            'contact_randomization': randomization}}
    if not randomization:
        d['warnings'] = ['no friction/torsion randomization found (contract C9)']
    d.setdefault('warnings', []).extend(contract.warn(cfg))
    if unnamed:
        d['warnings'].append(f"{len(unnamed)} geoms collide that the collision config does not name "
                             f"(kept from the XML): {', '.join(c['geom'] or c['body'] for c in unnamed)}")
    errors = sanity(d, mass_g)
    if errors:
        raise PlantError(f"task {task}: implausible plant values:\n  " + '\n  '.join(errors))
    return d

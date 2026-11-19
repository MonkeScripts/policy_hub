"""The task contract (AGENTS.md section 2), checked on a task's env config before anything runs.

`check(cfg)` raises ContractError listing every requirement the task breaks; `warn(cfg)` returns
the optional ones it does not meet. Numbers C1-C13 match AGENTS.md.
"""

from __future__ import annotations

import re

# BAM servo config (bam.mjlab.BamActuatorCfg and subclasses). Required: the gain, the latency, the
# servo model named one of two ways, and the supply either fixed or as a range. Optional fields
# (kd_fw, max_pwm in the HD-1910 subclass; sag and floor) are recorded when present.
ACTUATOR_REQUIRED = ('kp_fw', 'delay_min_lag', 'delay_max_lag')
PRONE_PARAMS = ('prone_prob', 'face_down_prob', 'crouch_prob')


class ContractError(RuntimeError):
    pass


def _errors(cfg, needs_prone: bool) -> list[str]:
    e = []
    robot = cfg.scene.entities.get('robot')
    if robot is None:
        return ["C2: no scene entity named 'robot'"]
    if not callable(getattr(robot, 'spec_fn', None)):
        e.append('C2: robot.spec_fn is missing')
    acts = getattr(robot.articulation, 'actuators', ())
    if not acts:
        e.append('C3: robot has no actuators')
    else:
        a, name = acts[0], type(acts[0]).__name__
        g = lambda f: getattr(a, f, None)
        missing = [f for f in ACTUATOR_REQUIRED if g(f) is None]
        if missing:
            e.append(f"C3: actuators[0] ({name}) lacks {', '.join(missing)}")
        if g('json_path') is None and (g('motor_name') is None or g('model') is None):
            e.append(f"C3: actuators[0] ({name}) names no servo model (json_path, or motor_name + model)")
        if g('vin') is None and g('vin_range') is None:
            e.append(f"C3: actuators[0] ({name}) has no supply (vin or vin_range)")
    if not robot.init_state.joint_pos:
        e.append('C4: robot.init_state.joint_pos (HOME) is empty')
    if 'joint_pos' not in cfg.actions:
        e.append("C6: no action term 'joint_pos'")
    elif getattr(cfg.actions['joint_pos'], 'scale', 1.0) not in (1, 1.0):
        e.append(f"C6: action scale is {cfg.actions['joint_pos'].scale}, expected 1")
    if 'actor' not in cfg.observations:
        e.append("C7: no observation group 'actor'")
    tw = cfg.commands.get('twist')
    if tw is None:
        e.append("C8: no command 'twist'")
    else:
        for f in ('ranges', 'heading_command', 'resampling_time_range'):
            if not hasattr(tw, f):
                e.append(f"C8: command 'twist' has no '{f}'")
    if 'foot_friction' not in cfg.events or 'ranges' not in (cfg.events['foot_friction'].params or {}):
        e.append("C9: no event 'foot_friction' with params['ranges']")
    if 'time_out' not in cfg.terminations:
        e.append("C10: no termination 'time_out'")
    if needs_prone:
        ev = cfg.events.get('random_prone_init')
        if ev is None or any(p not in (ev.params or {}) for p in PRONE_PARAMS):
            e.append(f"C11: get-up family needs reset event 'random_prone_init' with {', '.join(PRONE_PARAMS)}")
    return e


def check(cfg, task: str, needs_prone: bool = False) -> None:
    e = _errors(cfg, needs_prone)
    if e:
        raise ContractError(f"task {task} breaks the contract (see AGENTS.md section 2):\n  " + '\n  '.join(e))


def check_joints(task: str, home_patterns: dict, actuated: list[str]) -> None:
    """C4 against the compiled model: every actuated joint has a HOME value."""
    missing = [j for j in actuated if not any(re.fullmatch(p, j) for p in home_patterns)]
    if missing:
        raise ContractError(f"task {task}: C4: no HOME for {', '.join(missing)}")


def warn(cfg) -> list[str]:
    w = []
    sensors = {getattr(s, 'name', None) for s in getattr(cfg.scene, 'sensors', ()) or ()}
    if 'feet_ground_contact' not in sensors:
        w.append("C13: no 'feet_ground_contact' sensor: standing double support will be null")
    return w

"""The Pollen contract (AGENTS.md section 2), checked on a task's env config before anything runs.

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
        raise ContractError(f"task {task} breaks the Pollen contract (see AGENTS.md section 2):\n  " + '\n  '.join(e))


def check_joints(task: str, home_patterns: dict, actuated: list[str]) -> None:
    """C4 against the compiled model: every actuated joint has a HOME value."""
    missing = [j for j in actuated if not any(re.fullmatch(p, j) for p in home_patterns)]
    if missing:
        raise ContractError(f"task {task}: C4: no HOME for {', '.join(missing)}")


# The firmware contract: what microduck_runtime's tools/export_policy.py refuses to build (F1-F6).
# Each policy carries its own HOME; F6 also holds it to the task's HOME, which the firmware cannot know.
FW_JOINTS = ('left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
             'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
             'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle')
FW_OBSERVATIONS = 'base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command'
FW_OPS = ['Sub', 'Div', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm']
FW_WIDTHS = [61, 512, 256, 128, 14]


def _floats(s) -> list[float] | None:
    try:
        return [float(v) for v in s.split(',')]
    except (AttributeError, ValueError):
        return None


def _onnx_errors(model, home: dict | None) -> list[str]:
    import numpy as np
    from onnx import helper, numpy_helper

    g, e = model.graph, []
    ops = [n.op_type for n in g.node]
    if ops != FW_OPS:
        return [f'F1: graph is {ops}, expected Sub, Div, then 4 Gemm with Elu between']
    width = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim][-1:]
    if [width(v) for v in g.input] != [[FW_WIDTHS[0]]] or [width(v) for v in g.output] != [[FW_WIDTHS[-1]]]:
        e.append('F2: expected one input of 61 and one output of 14')
    t = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for k, n in enumerate(n for n in g.node if n.op_type == 'Gemm'):
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        if a.get('transB', 0) != 1 or a.get('transA', 0) or a.get('alpha', 1) != 1 or a.get('beta', 1) != 1:
            e.append(f'F2: layer {k}: Gemm needs transB=1, alpha=beta=1')
        w, b = t.get(n.input[1]), t.get(n.input[2])
        if getattr(w, 'shape', None) != (FW_WIDTHS[k + 1], FW_WIDTHS[k]) or getattr(b, 'shape', None) != (FW_WIDTHS[k + 1],):
            e.append(f'F2: layer {k}: weight {getattr(w, "shape", None)}, expected {(FW_WIDTHS[k + 1], FW_WIDTHS[k])}')
    if any(x.name == 'alpha' and x.f != 1.0 for n in g.node if n.op_type == 'Elu' for x in n.attribute):
        e.append('F2: Elu alpha must be 1')
    std = t.get(g.node[1].input[1])
    if any(x.dtype != np.float32 or not np.isfinite(x).all() for x in t.values()) or std is None or (std == 0).any():
        e.append('F2: tensors must be finite float32 and the normalizer std nonzero')
    meta = {p.key: p.value for p in model.metadata_props}
    if meta.get('observation_names') != FW_OBSERVATIONS:
        e.append(f"F3: observation_names {meta.get('observation_names')!r}, expected {FW_OBSERVATIONS!r}")
    if _floats(meta.get('action_scale')) != [1.0]:
        e.append(f"F4: action_scale {meta.get('action_scale')!r}, the firmware applies 1.0")
    names = tuple(meta.get('joint_names', '').split(','))
    if names != FW_JOINTS:
        e.append(f'F5: joint_names {list(names)}, expected the firmware order {list(FW_JOINTS)}')
    poses = _floats(meta.get('default_joint_pos'))
    if poses is None or len(poses) != len(FW_JOINTS) or not all(abs(p) <= 1.0 for p in poses):
        e.append(f"F6: default_joint_pos {meta.get('default_joint_pos')!r}, expected {len(FW_JOINTS)} finite values within 1 rad")
    elif home and names == FW_JOINTS:
        off = [f'{j} {p} vs {home[j]}' for j, p in zip(names, poses) if j in home and abs(p - home[j]) > 1e-3]
        if off:
            e.append(f"F6: default_joint_pos differs from the task's HOME: {', '.join(off)}")
    return e


def check_onnx(path, home: dict | None = None) -> None:
    """F1-F6 on an exported policy. The firmware's converter refuses anything failing them, except the
    match with the task's HOME in F6, which only duckhub can check."""
    import onnx

    e = _onnx_errors(onnx.load(str(path)), home)
    if e:
        raise ContractError(f'{path} breaks the firmware contract (see AGENTS.md):\n  '
                            + '\n  '.join(e))


def warn(cfg) -> list[str]:
    w = []
    sensors = {getattr(s, 'name', None) for s in getattr(cfg.scene, 'sensors', ()) or ()}
    if 'feet_ground_contact' not in sensors:
        w.append("C13: no 'feet_ground_contact' sensor: standing double support will be null")
    return w

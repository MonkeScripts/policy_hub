"""The Microduck firmware contract (F1-F6): what microduck_runtime's tools/export_policy.py refuses
to build. Applied when policyhub.yaml sets `microduck_firmware: true`."""

from __future__ import annotations

from policyhub.contract import ContractError

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
    match with the task's HOME in F6, which only policyhub can check."""
    import onnx

    e = _onnx_errors(onnx.load(str(path)), home)
    if e:
        raise ContractError(f'{path} breaks the firmware contract (see AGENTS.md):\n  '
                            + '\n  '.join(e))

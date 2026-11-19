"""Walk family eval: follow a velocity command (vx, vy, yaw) and stay up. No get-up: a fall ends it.

    uv run policyhub eval walk --task TASK --onnx POLICY.onnx --out eval.json [--replay replay.mp4]

Every walk policy (any plant) is ranked by this file. Every robot starts standing at HOME. Tests:
  tracking   commands from standstill: yaw +-0.3, +-1.0 rad/s, vx 0.1 / 0.2 / 0.3 m/s; mean of the
             last 2 s of 4 s, after 2 s at zero command; how many robots actually turn.
  standing   zero command for 6 s: drift, double support, falls.
  walk       stand 3 s, vx 0.2 for 2 s, stand 3 s: nominal, low supply, low friction; falls,
             first-step twist spikes, speed.
  pushes     random 0.3 m/s pushes every 2-3 s for 20 s, walking at vx 0.3 and standing: falls.
Falls = trunk tilt over 45 deg. Conditions are fixed here (VERSION), not taken from the task;
results of different VERSIONs are not ranked together.
"""

from __future__ import annotations

import math
import time

import torch

from policyhub.eval.policy import OnnxPolicy, sha256
from policyhub.eval.sim import FALL_TILT, SEED, Sim, frames, make_env, write_replay

FAMILY = 'walk'
VERSION = '1.1'  # 1.1: the head and body commands are held like the twist (in 1.0 they redrew for one step every 2-5 s)
NEEDS_PRONE = False

N = 256
TRACKING = [(0, 0.3), (0, -0.3), (0, 1.0), (0, -1.0), (0.1, 0), (0.2, 0), (0.3, 0)]
LOW_FRICTION = (0.15, 0.3)
LOW_SUPPLY = dict(vin_range=(4.5, 5.5), vin_drop_gain_range=(0.3, 0.5))
PUSH = dict(velocity_range={'x': (-0.3, 0.3), 'y': (-0.3, 0.3)}, interval_range_s=(2.0, 3.0))


def tracking(s: Sim) -> dict:
    out = {}
    for vx, wz in TRACKING:
        s.reset()
        for _ in s.run(2.0):  # settle at zero command
            pass
        V, W, fell = [], [], torch.zeros(s.n, dtype=torch.bool, device=s.device)
        for t in s.run(4.0, vx, wz):
            fell |= s.tilt() > FALL_TILT
            if t > 2.0:
                V.append(s.vx())
                W.append(s.wz())
        V, W = torch.stack(V).mean(0), torch.stack(W).mean(0)
        key, got = (f"vx_{vx:g}", V) if vx else (f"yaw_{wz:+g}", W)
        out[key] = {'commanded': vx or wz, 'achieved': round(float(got.mean()), 3),
                    'turning_share': round(float((W.abs() > 0.05).float().mean()), 3) if wz else None,
                    'falls_share': round(float(fell.float().mean()), 3)}
    return out


def standing(s: Sim) -> dict:
    s.reset()
    cs = s.env.scene.sensors.get('feet_ground_contact')
    X, D, fell = [], [], torch.zeros(s.n, dtype=torch.bool, device=s.device)
    for t in s.run(6.0):
        fell |= s.tilt() > FALL_TILT
        if t > 2.0:
            X.append(s.vx())
            if cs is not None:
                D.append((cs.data.found > 0).all(-1).float())
    return {'drift_vx_m_s': round(float(torch.stack(X).mean()), 3),
            'double_support': round(float(torch.stack(D).mean()), 3) if D else None,
            'falls_share': round(float(fell.float().mean()), 3)}


def walk(s: Sim) -> dict:
    s.reset()
    fell = torch.zeros(s.n, dtype=torch.bool, device=s.device)
    spike = torch.zeros_like(fell)
    V = []
    for _ in s.run(3.0):
        pass
    for t in s.run(2.0, 0.2):
        fell |= s.tilt() > FALL_TILT
        if t <= 0.4:
            spike |= s.wz().abs() > 2.0
        if t > 0.5:
            V.append(s.vx())
    for _ in s.run(3.0):
        fell |= s.tilt() > FALL_TILT
    return {'falls_share': round(float(fell.float().mean()), 3),
            'first_step_twist_spike_share': round(float(spike.float().mean()), 3),
            'speed_m_s': round(float(torch.stack(V).mean()), 3)}


def pushes(s: Sim, vx: float) -> dict:
    s.reset()
    fell = torch.zeros(s.n, dtype=torch.bool, device=s.device)
    for t in s.run(20.0, vx):
        if t > 1.0:
            fell |= s.tilt() > FALL_TILT
    return {'falls_share': round(float(fell.float().mean()), 3)}


# replay.mp4: the same scenarios for every policy, back to back, robot 0 of a 1-robot env, fixed
# camera, real time. (caption, push?, settle s, seconds, vx, wz)
REPLAY = [
    ('walk vx 0.2 m/s', False, 1.0, 4.0, 0.2, 0.0),
    ('walk vx 0.3 m/s', False, 1.0, 4.0, 0.3, 0.0),
    ('turn yaw +0.3 rad/s', False, 1.0, 4.0, 0.0, 0.3),
    ('turn yaw -1.0 rad/s', False, 1.0, 4.0, 0.0, -1.0),
    ('pushes 0.3 m/s while walking vx 0.3', True, 0.0, 8.0, 0.3, 0.0),
]


def replay(task: str, onnx_path: str, path) -> None:
    pol, segs, fps = OnnxPolicy(onnx_path), {}, 50
    for push in (False, True):
        items = [r for r in REPLAY if r[1] == push]
        s = Sim(make_env(task, 1, push=PUSH if push else None, render=True), pol)
        for cap, _, settle, seconds, vx, wz in items:
            s.reset()
            segs[cap] = frames(s, seconds, vx, wz, settle)
        fps = round(1 / s.dt)
        s.env.close()
    write_replay([(r[0], segs[r[0]]) for r in REPLAY], path, fps)


# How the hub ranks this family: per headline metric, which way is better.
RANKING = {
    'yaw_+0.3_achieved': ('closer', 0.3),
    'yaw_-0.3_achieved': ('closer', -0.3),
    'yaw_+1_achieved': ('closer', 1.0),
    'vx_0.2_achieved': ('closer', 0.2),
    'vx_0.3_achieved': ('closer', 0.3),
    'walk_falls_low_friction': 'lower',
    'walk_falls_low_supply': 'lower',
    'push_walking_falls': 'lower',
    'push_standing_falls': 'lower',
}


def run(task: str, onnx_path: str) -> dict:
    t0 = time.time()
    pol = OnnxPolicy(onnx_path)

    s = Sim(make_env(task, N), pol)
    m = {'tracking_from_standstill': tracking(s), 'standing': standing(s), 'walk_vx0.2': {'nominal': walk(s)}}
    s.env.close()
    for name, kw in (('low_supply', dict(supply=LOW_SUPPLY)), ('low_friction', dict(friction=LOW_FRICTION))):
        s = Sim(make_env(task, N, **kw), pol)
        m['walk_vx0.2'][name] = walk(s)
        s.env.close()
    s = Sim(make_env(task, N, push=PUSH), pol)
    m['pushes'] = {'walking_vx0.3': pushes(s, 0.3), 'standing': pushes(s, 0.0)}
    s.env.close()

    tr, wk, pu = m['tracking_from_standstill'], m['walk_vx0.2'], m['pushes']
    headline = {
        'yaw_+0.3_achieved': tr['yaw_+0.3']['achieved'],
        'yaw_-0.3_achieved': tr['yaw_-0.3']['achieved'],
        'yaw_+1_achieved': tr['yaw_+1']['achieved'],
        'vx_0.2_achieved': tr['vx_0.2']['achieved'],
        'vx_0.3_achieved': tr['vx_0.3']['achieved'],
        'walk_falls_low_friction': wk['low_friction']['falls_share'],
        'walk_falls_low_supply': wk['low_supply']['falls_share'],
        'push_walking_falls': pu['walking_vx0.3']['falls_share'],
        'push_standing_falls': pu['standing']['falls_share'],
    }
    return {
        'spec_version': 1,
        'eval': {'family': FAMILY, 'version': VERSION, 'seed': SEED, 'num_envs': N,
                 'conditions': {'low_friction': LOW_FRICTION, 'low_supply': LOW_SUPPLY, 'push': PUSH,
                                'fall_tilt_deg': round(math.degrees(FALL_TILT))}},
        'task': task,
        'onnx_sha256': sha256(onnx_path),
        'headline': headline,
        'metrics': m,
        'runtime_s': round(time.time() - t0),
    }

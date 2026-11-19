"""Velstand family eval: walk on command, and get back up after a fall.

    uv run policyhub eval velstand --task TASK --onnx POLICY.onnx --out eval.json

Every velstand policy (any plant) is ranked by this file. Tests:
  getup      from face-down, face-up and crouch: success (up for the whole last 1 s of 10 s),
             median time to first be up, peak joint torque. "Up" = trunk within 20 mm of the
             robot's own standing height (measured first) and tilt < 20 deg.
  tracking   commands from standstill: yaw +-0.3, +-1.0, vx 0.1 / 0.2 / 0.3 (mean of the last 2 s).
  standing   zero command for 6 s: drift, double support.
  walk       stand 3 s, vx 0.2 for 2 s, stand 3 s: nominal, low supply, low friction; falls,
             first-step twist spikes, speed.
  pushes     random 0.3 m/s pushes every 2-3 s for 20 s, walking and standing: how many fell,
             and how many are up at the end (a velstand policy should get up again).
Conditions are fixed here (VERSION), not taken from the task; results of different VERSIONs
are not ranked together.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from policyhub.eval.policy import OnnxPolicy, sha256
from policyhub.eval.sim import FALL_TILT, SEED, Sim, frames, make_env, write_replay

FAMILY = 'velstand'
VERSION = '1.1'  # 1.1: the head and body commands are held like the twist (in 1.0 they redrew for one step every 2-5 s)
NEEDS_PRONE = True  # get-up tests use the task's random_prone_init (contract C11)

N = 256
STANDING = dict(prone_prob=0.0, face_down_prob=1.0, crouch_prob=0.0)
STARTS = {
    'face_down': dict(prone_prob=1.0, face_down_prob=1.0, crouch_prob=0.0),
    'face_up': dict(prone_prob=1.0, face_down_prob=0.0, crouch_prob=0.0),
    'crouch': dict(prone_prob=0.0, face_down_prob=1.0, crouch_prob=1.0),
}
UP_BAND, UP_TILT = 0.02, math.radians(20)
TRACKING = [(0, 0.3), (0, -0.3), (0, 1.0), (0, -1.0), (0.1, 0), (0.2, 0), (0.3, 0)]
LOW_FRICTION = (0.15, 0.3)
LOW_SUPPLY = dict(vin_range=(4.5, 5.5), vin_drop_gain_range=(0.3, 0.5))
PUSH = dict(velocity_range={'x': (-0.3, 0.3), 'y': (-0.3, 0.3)}, interval_range_s=(2.0, 3.0))


def standing_height(s: Sim) -> float:
    s.set_event_params('random_prone_init', **STANDING)
    s.reset()
    Z = [s.z() for t in s.run(3.0) if t > 2.0]
    return float(torch.stack(Z).median())


def getup(s: Sim, z_up: float) -> dict:
    out = {}
    last = int(1.0 / s.dt)
    for name, start in STARTS.items():
        s.set_event_params('random_prone_init', **start)
        s.reset()
        up, peak = [], torch.zeros(s.n, device=s.device)
        for _ in s.run(10.0):
            up.append(((s.z() - z_up).abs() < UP_BAND) & (s.tilt() < UP_TILT))
            peak = torch.maximum(peak, s.torque().abs().max(1).values)
        up = torch.stack(up).cpu().numpy()
        ever = up.any(0)
        out[name] = {'success': round(float(up[-last:].all(0).mean()), 3),
                     'time_s_median': round(float(np.median(up.argmax(0)[ever] * s.dt)), 2) if ever.any() else None,
                     'peak_torque_nm_median': round(float(peak.median()), 2)}
    return out


def tracking(s: Sim) -> dict:
    out = {}
    for vx, wz in TRACKING:
        s.set_event_params('random_prone_init', **STANDING)
        s.reset()
        for _ in s.run(2.0):  # settle at zero command
            pass
        V, W = [], []
        for t in s.run(4.0, vx, wz):
            if t > 2.0:
                V.append(s.vx())
                W.append(s.wz())
        V, W = torch.stack(V).mean(0), torch.stack(W).mean(0)
        key, got = (f"vx_{vx:g}", V) if vx else (f"yaw_{wz:+g}", W)
        out[key] = {'commanded': vx or wz, 'achieved': round(float(got.mean()), 3),
                    'turning_share': round(float((W.abs() > 0.05).float().mean()), 3) if wz else None}
    return out


def standing(s: Sim) -> dict:
    s.set_event_params('random_prone_init', **STANDING)
    s.reset()
    cs = s.env.scene.sensors.get('feet_ground_contact')
    X, D = [], []
    for t in s.run(6.0):
        if t > 2.0:
            X.append(s.vx())
            if cs is not None:
                D.append((cs.data.found > 0).all(-1).float())
    return {'drift_vx_m_s': round(float(torch.stack(X).mean()), 3),
            'double_support': round(float(torch.stack(D).mean()), 3) if D else None}


def walk(s: Sim) -> dict:
    s.set_event_params('random_prone_init', **STANDING)
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


def pushes(s: Sim, vx: float, z_up: float) -> dict:
    s.set_event_params('random_prone_init', **STANDING)
    s.reset()
    fell = torch.zeros(s.n, dtype=torch.bool, device=s.device)
    for t in s.run(20.0, vx):
        if t > 1.0:
            fell |= s.tilt() > FALL_TILT
    up_end = ((s.z() - z_up).abs() < UP_BAND) & (s.tilt() < UP_TILT)
    return {'fell_share': round(float(fell.float().mean()), 3), 'up_at_end_share': round(float(up_end.float().mean()), 3)}


# replay.mp4: the same scenarios for every policy, back to back, robot 0 of a 1-robot env, fixed
# camera, real time. (caption, start pose, push?, settle s, seconds, vx, wz)
REPLAY = [
    ('get up from face-down', 'face_down', False, 0.0, 4.0, 0.0, 0.0),
    ('get up from face-up', 'face_up', False, 0.0, 4.0, 0.0, 0.0),
    ('walk vx 0.2 m/s', 'standing', False, 1.0, 4.0, 0.2, 0.0),
    ('turn yaw +0.3 rad/s', 'standing', False, 1.0, 4.0, 0.0, 0.3),
    ('pushes 0.3 m/s while walking vx 0.3', 'standing', True, 0.0, 8.0, 0.3, 0.0),
]


def replay(task: str, onnx_path: str, path) -> None:
    pol, segs = OnnxPolicy(onnx_path), {}
    for push in (False, True):
        items = [r for r in REPLAY if r[2] == push]
        s = Sim(make_env(task, 1, push=PUSH if push else None, needs_prone=True, render=True), pol)
        for cap, start, _, settle, seconds, vx, wz in items:
            s.set_event_params('random_prone_init', **(STANDING if start == 'standing' else STARTS[start]))
            s.reset()
            segs[cap] = frames(s, seconds, vx, wz, settle)
        fps = round(1 / s.dt)
        s.env.close()
    write_replay([(r[0], segs[r[0]]) for r in REPLAY], path, fps)


# How the hub ranks this family: per headline metric, which way is better.
# "higher" / "lower", or ("closer", target) for tracking.
RANKING = {
    'getup_facedown_success': 'higher',
    'getup_faceup_success': 'higher',
    'getup_facedown_time_s': 'lower',
    'yaw_+0.3_achieved': ('closer', 0.3),
    'yaw_-0.3_achieved': ('closer', -0.3),
    'vx_0.2_achieved': ('closer', 0.2),
    'walk_falls_low_friction': 'lower',
    'walk_falls_low_supply': 'lower',
    'push_walking_up_at_end': 'higher',
}


def run(task: str, onnx_path: str) -> dict:
    t0 = time.time()
    pol = OnnxPolicy(onnx_path)

    s = Sim(make_env(task, N, needs_prone=True), pol)
    z_up = standing_height(s)
    m = {'standing_height_m': round(z_up, 4), 'getup': getup(s, z_up), 'tracking_from_standstill': tracking(s),
         'standing': standing(s), 'walk_vx0.2': {'nominal': walk(s)}}
    s.env.close()
    for name, kw in (('low_supply', dict(supply=LOW_SUPPLY)), ('low_friction', dict(friction=LOW_FRICTION))):
        s = Sim(make_env(task, N, needs_prone=True, **kw), pol)
        m['walk_vx0.2'][name] = walk(s)
        s.env.close()
    s = Sim(make_env(task, N, push=PUSH, needs_prone=True), pol)
    m['pushes'] = {'walking_vx0.3': pushes(s, 0.3, z_up), 'standing': pushes(s, 0.0, z_up)}
    s.env.close()

    g, tr, wk = m['getup'], m['tracking_from_standstill'], m['walk_vx0.2']
    headline = {
        'getup_facedown_success': g['face_down']['success'],
        'getup_faceup_success': g['face_up']['success'],
        'getup_facedown_time_s': g['face_down']['time_s_median'],
        'yaw_+0.3_achieved': tr['yaw_+0.3']['achieved'],
        'yaw_-0.3_achieved': tr['yaw_-0.3']['achieved'],
        'vx_0.2_achieved': tr['vx_0.2']['achieved'],
        'walk_falls_low_friction': wk['low_friction']['falls_share'],
        'walk_falls_low_supply': wk['low_supply']['falls_share'],
        'push_walking_up_at_end': m['pushes']['walking_vx0.3']['up_at_end_share'],
    }
    return {
        'spec_version': 1,
        'eval': {'family': FAMILY, 'version': VERSION, 'seed': SEED, 'num_envs': N,
                 'conditions': {'low_friction': LOW_FRICTION, 'low_supply': LOW_SUPPLY, 'push': PUSH,
                                'fall_tilt_deg': round(math.degrees(FALL_TILT)), 'up_band_m': UP_BAND}},
        'task': task,
        'onnx_sha256': sha256(onnx_path),
        'headline': headline,
        'metrics': m,
        'runtime_s': round(time.time() - t0),
    }

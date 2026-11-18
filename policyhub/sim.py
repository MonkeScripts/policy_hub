"""Task env with an eval's fixed conditions, and a stepper that holds commands and reads trunk state.

Runs inside the training repo's environment: the task comes from its mjlab task registry.
Nothing here decides what is tested; each family's eval.py does.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.managers import EventTermCfg
from mjlab.tasks.registry import load_env_cfg

from policyhub import contract
from policyhub.policy import OnnxPolicy

SEED = 0
FALL_TILT = math.radians(45)
VIDEO_SIZE = (640, 480)


def make_env(task: str, n: int, friction=None, supply=None, push=None, needs_prone: bool = False,
             render: bool = False, camera: dict | None = None) -> ManagerBasedRlEnv:
    """The task's play config with: curricula off, only time_out terminations (falls are read
    from tilt), the task's own pushes off, commands held by the caller. friction / supply / push
    are the eval's fixed test conditions. camera: viewer settings that replace the fixed camera's
    (e.g. azimuth, elevation), for a render."""
    cfg = load_env_cfg(task, play=True)
    contract.check(cfg, task, needs_prone=needs_prone)
    cfg.scene.num_envs = n
    cfg.seed = SEED
    cfg.episode_length_s = 1e4
    for k in [k for k in cfg.events if 'push' in k]:
        cfg.events.pop(k)
    cfg.curriculum.clear()
    for t in [t for t in cfg.terminations if t != 'time_out']:
        cfg.terminations.pop(t)
    tw = cfg.commands['twist']
    for f in ('rel_small_cmd_envs', 'rel_turn_in_place_envs', 'rel_standing_envs'):
        if hasattr(tw, f):
            setattr(tw, f, 0.0)
    tw.heading_command, tw.ranges.heading = False, None
    for c in cfg.commands.values():  # every command is held by the caller: none redraws itself mid-eval
        c.resampling_time_range = (1e4, 1e4)
    if friction is not None:
        for k in ('foot_friction', 'foot_friction_interval'):
            if k in cfg.events:
                cfg.events[k].params['ranges'] = friction
    if supply is not None:
        act = cfg.scene.entities['robot'].articulation.actuators[0]
        act.vin_range, act.vin_drop_gain_range = supply['vin_range'], supply['vin_drop_gain_range']
        act.vin_min = min(act.vin_min or 0.0, supply['vin_range'][0])
    if push is not None:
        cfg.events['eval_push'] = EventTermCfg(
            func=push_by_setting_velocity, mode='interval', interval_range_s=push['interval_range_s'],
            params={'velocity_range': push['velocity_range']})
    if render:  # the same camera for every policy unless `camera` says otherwise: follows the trunk from the front-left
        v = cfg.viewer
        v.width, v.height = VIDEO_SIZE
        v.distance, v.elevation, v.azimuth = 0.9, -15.0, 135.0
        v.max_extra_envs = 0
        for k, val in (camera or {}).items():
            setattr(v, k, val)
    torch.manual_seed(SEED)
    return ManagerBasedRlEnv(cfg, device='cuda:0', render_mode='rgb_array' if render else None)


class Sim:
    def __init__(self, env: ManagerBasedRlEnv, policy: OnnxPolicy):
        self.env, self.pol = env, policy
        self.rob = env.scene['robot']
        cm = env.command_manager
        self.twist = cm.get_term('twist')
        self.other_cmds = [cm.get_term(c) for c in cm.active_terms if c != 'twist']
        self.term = env.action_manager.get_term('joint_pos')
        self.qi = [self.rob.joint_names.index(j) for j in self.term.target_names]
        self.dt, self.n, self.device = env.step_dt, env.num_envs, env.device

    def set_event_params(self, name: str, **params) -> None:
        self.env.event_manager.get_term_cfg(name).params.update(params)

    def reset(self) -> None:
        torch.manual_seed(SEED)
        self.obs, _ = self.env.reset(seed=SEED)
        assert self.obs['actor'].shape[1] == self.pol.obs_len, 'task observations do not match the ONNX input'

    def step(self, vx: float = 0.0, wz: float = 0.0, vy: float = 0.0) -> None:
        """Hold the twist command at (vx, vy, wz); families that reuse the slots (flamingo: [flag, side, 0])
        pass their values through the same three."""
        self.twist.vel_command_b[:] = 0.0
        self.twist.vel_command_b[:, 0], self.twist.vel_command_b[:, 1], self.twist.vel_command_b[:, 2] = vx, vy, wz
        for c in self.other_cmds:  # head / body pose commands held at neutral
            if hasattr(c, '_command'):
                c._command[:] = 0.0
        self.obs, *_ = self.env.step(self.pol(self.obs['actor']))

    def run(self, seconds: float, vx: float = 0.0, wz: float = 0.0, vy: float = 0.0):
        """Step for `seconds`, yielding the elapsed time after each step."""
        for k in range(int(round(seconds / self.dt))):
            self.step(vx, wz, vy)
            yield (k + 1) * self.dt

    # trunk state
    def tilt(self) -> torch.Tensor:
        return torch.acos((-self.rob.data.projected_gravity_b[:, 2]).clamp(-1, 1))

    def z(self) -> torch.Tensor:
        return self.rob.data.root_link_pos_w[:, 2]

    def vx(self) -> torch.Tensor:
        return self.rob.data.root_link_lin_vel_b[:, 0]

    def wz(self) -> torch.Tensor:
        return self.rob.data.root_link_ang_vel_b[:, 2]

    def torque(self) -> torch.Tensor:
        return self.env.sim.data.qfrc_actuator[:, self.rob.indexing.joint_v_adr[self.qi]]


def frames(s: 'Sim', seconds: float, vx: float = 0.0, wz: float = 0.0, settle: float = 0.0, vy: float = 0.0) -> list:
    """Frames of env 0 at the control rate: `settle` s at zero command, then `seconds` at (vx, vy, wz)."""
    out = [s.env.render() for _ in s.run(settle)]
    return out + [s.env.render() for _ in s.run(seconds, vx, wz, vy)]


def label(frames: list, text: str) -> list:
    """Burn a caption into the top-left corner of each frame."""
    from PIL import Image, ImageDraw

    out = []
    for f in frames:
        im = Image.fromarray(f)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 12 + 7 * len(text), 26), fill=(0, 0, 0))
        d.text((6, 7), text, fill=(255, 255, 255))
        out.append(np.asarray(im))
    return out


def write_replay(segments: list, path, fps: int) -> None:
    """One mp4 from labelled segments [(caption, frames), ...], played back to back."""
    import imageio.v2 as imageio

    allf = [f for cap, fr in segments for f in label(fr, cap)]
    imageio.mimwrite(str(path), allf, fps=fps, codec='libx264', quality=7, macro_block_size=16)

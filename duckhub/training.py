"""Where training ended and what its curricula had set by then.

The run's params/env.yaml holds the curriculum *plan*. What a policy was actually trained with at
the end comes from two things:
  - the checkpoint the ONNX was exported from: its iteration and the env step counter the
    curricula run on (rsl_rl saves infos.env_state.common_step_counter);
  - the task's own curriculum functions, applied at that counter in a 1-robot env, then reading
    the live reward weights and the parameters of every term a curriculum targets.
That is exact for any curriculum shape. If the task code no longer matches env.yaml (it changed
after training), the live result is not trusted and the plan is replayed instead, with a warning.
"""

from __future__ import annotations

from pathlib import Path


def checkpoint_position(path) -> dict:
    import torch

    d = torch.load(str(path), map_location='cpu', weights_only=False)
    steps = ((d.get('infos') or {}).get('env_state') or {}).get('common_step_counter')
    return {'checkpoint': Path(path).name, 'iteration': d.get('iter'),
            'env_steps': int(steps) if steps is not None else None}


def live_final(task: str, env_steps: int, env_yaml: dict) -> tuple[dict | None, list[str]]:
    """({rewards, curriculum_targets}, warnings), or (None, warnings) if the task code drifted."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from duckhub.plant import _plain

    cfg = load_env_cfg(task, play=False)
    planned = {k: v['weight'] for k, v in (env_yaml.get('rewards') or {}).items()}
    live = {k: t.weight for k, t in cfg.rewards.items()}
    drift = []
    if planned != live:
        changed = sorted(k for k in planned.keys() | live.keys() if planned.get(k) != live.get(k))
        drift.append(f"reward terms differ from the run's env.yaml: {', '.join(changed)}")
    if set(env_yaml.get('curriculum') or {}) != set(cfg.curriculum):
        drift.append("curriculum terms differ from the run's env.yaml")
    if drift:
        return None, ['task code changed since training (' + '; '.join(drift) + '): final weights replayed from the plan instead']

    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        env.common_step_counter = env_steps
        env.curriculum_manager.compute(env_ids=torch.arange(1, device=env.device))
        rewards = {k: float(env.reward_manager.get_term_cfg(k).weight) for k in env.reward_manager.active_terms}
        targets = {}
        for name in env.curriculum_manager.active_terms:
            p = env.curriculum_manager.get_term_cfg(name).params or {}
            if 'event_name' in p:
                targets[f"event:{p['event_name']}"] = _plain(env.event_manager.get_term_cfg(p['event_name']).params)
            elif 'term_name' in p and p['term_name'] in env.termination_manager.active_terms:
                targets[f"termination:{p['term_name']}"] = _plain(env.termination_manager.get_term_cfg(p['term_name']).params)
            elif 'command_name' in p:
                c = env.command_manager.get_term(p['command_name']).cfg
                targets[f"command:{p['command_name']}"] = _plain({k: getattr(c, k) for k in ('ranges', 'rel_standing_envs')
                                                                   if hasattr(c, k)})
    finally:
        env.close()
    return {'rewards': rewards, 'curriculum_targets': {k: v for k, v in targets.items() if v is not None}}, []


def unknown_curricula(env_yaml: dict) -> list[str]:
    """Curriculum terms the plan replay cannot interpret (no reward_name + weight_stages and no *_stages)."""
    out = []
    for name, c in (env_yaml.get('curriculum') or {}).items():
        p = c.get('params') or {}
        if not any(k.endswith('_stages') for k in p):
            out.append(name)
    return out

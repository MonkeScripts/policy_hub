"""README.md: the package's Hub page (tags and headline metrics in the YAML header)."""

from __future__ import annotations

import yaml


def _num(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_num(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{' + ', '.join(f"{k}: {_num(x)}" for k, x in v.items()) + '}'
    return str(v)


def _short(params, width: int = 110) -> str:
    """Numeric settings of an event term, without asset selectors and function references."""
    if not isinstance(params, dict):
        return _num(params)
    keep = {k: v for k, v in params.items()
            if k not in ('asset_cfg', 'sensor_name', 'inner', 'command_name')
            and not (isinstance(v, str) and v.startswith('<'))
            and not (isinstance(v, dict) and {'name', 'joint_names'} & set(v))}
    t = _num(keep)
    return t if len(t) <= width else t[: width - 1] + '…'


def _stages(params: dict, steps_per_it: int) -> tuple[str, str]:
    """(what it changes, 'value @ iteration' list) for a curriculum term."""
    target = params.get('reward_name') or params.get('event_name') or params.get('term_name') or params.get('command_name') or '?'
    key = next((k for k in params if k.endswith('_stages')), None)
    if key is None:
        return target, _short(params)
    parts = []
    for st in params[key]:
        val = {k: v for k, v in st.items() if k != 'step'}
        val = next(iter(val.values())) if len(val) == 1 else val
        parts.append(f"{_num(val)} @ it {round(st['step'] / steps_per_it)}")
    return f"{target} ({key.replace('_stages', '')})", ' → '.join(parts)


def training_sections(t: dict) -> list[str]:
    s = t['summary']
    spi = s.get('ppo', {}).get('num_steps_per_env') or 24
    ri, rf = s.get('rewards_initial', {}), s.get('rewards_final', {})
    lines = ['### Rewards', '', f"Final weights: {s.get('rewards_final_source', 'replayed from the curriculum plan')}.", '',
             '| term | initial weight | final weight |', '|---|---|---|']
    for k in ri:
        a, b = ri[k], rf.get(k, ri[k])
        lines.append(f"| {k} | {_num(a)} | {_num(b)}{' (curriculum)' if a != b else ''} |")
    lines += ['', '### Curriculum', '', f"Stages in training iterations ({spi} env steps each).", '',
              '| term | changes | stages |', '|---|---|---|']
    for k, p in (s.get('curriculum') or {}).items():
        what, st = _stages(p or {}, spi)
        lines.append(f"| {k} | {what} | {st} |")
    end = s.get('curriculum_final') or {}
    lines += ['', '### Randomization', '', 'Settings at the start of training'
              + ('; for events a curriculum changes, the value at the end of training follows the arrow.' if end else
                 '; events named in the curriculum above change during it.'), '',
              '| event | settings |', '|---|---|']
    for k, p in (s.get('randomization') or {}).items():
        fin = end.get(f"event:{k}")
        lines.append(f"| {k} | {_short(p)}{' → ' + _short(fin) if fin and fin != p else ''} |")
    ppo, alg = s.get('ppo', {}), s.get('ppo', {}).get('algorithm') or {}
    lines += ['', '### PPO', '',
              f"Actor {ppo.get('actor')}, {ppo.get('num_steps_per_env')} steps per env, "
              f"learning rate {alg.get('learning_rate')} ({alg.get('schedule')}), entropy {alg.get('entropy_coef')}, "
              f"{alg.get('num_learning_epochs')} epochs x {alg.get('num_mini_batches')} mini-batches, "
              f"gamma {alg.get('gamma')}, lambda {alg.get('lam')}, clip {alg.get('clip_param')}, seed {ppo.get('seed')}.", '']
    if s.get('terminations'):
        lines += [f"Terminations: {', '.join(s['terminations'])}.", '']
    return lines


def card(m: dict, t: dict, e: dict, plant: dict, has_replay: bool = False, has_checkpoint: bool = False) -> str:
    robot = m['robot']
    rows = list(e['headline'].items())
    ev = e['eval']
    meta = {
        'license': 'apache-2.0',
        'library_name': 'onnx',
        'tags': [robot, f"{robot}-family-{m['family']}", f"{robot}-plant-{m['plant']['plant_hash']}",
                 f"{robot}-eval-{ev['family']}-{ev['version']}",
                 'robotics', 'reinforcement-learning'],
        'model-index': [{'name': m['repo'].split('/')[1], 'results': [{
            'task': {'type': 'reinforcement-learning', 'name': m['description']},
            'dataset': {'name': f"{ev['family']} eval v{ev['version']}", 'type': f'{robot}-sim'},
            'metrics': [{'name': k, 'type': k, 'value': v} for k, v in rows],
        }]}],
    }
    r = plant['robot']
    a = r['actuators'][0]
    code = t['code']
    lines = [
        '---', yaml.safe_dump(meta, sort_keys=False).strip(), '---', '',
        f"# {m['repo'].split('/')[1]}", '', m['description'], '',
        '## Run it', '',
        f"- Kp {m['gains']['kp']} / Kd {m['gains']['kd']} written at start, {m['control_hz']} Hz.",
        f"- HOME (default_joint_pos, policy joint order): `{m['home']}`",
        *[f"- Command `{k}`: {v}" for k, v in m['command'].items()],
        *[f"- {n}" for n in m['runtime_notes']], '',
        '## Plant', '',
        f"Plant hash `{m['plant']['plant_hash']}`, from task `{m['plant']['task']}`: {r['total_mass_g']} g, "
        f"servo model `{a.get('json_path') or (str(a.get('motor_name')) + ' ' + str(a.get('model')))}`, supply {a.get('vin_range') or a.get('vin')} V (sag {a.get('vin_drop_gain_range')} V/N·m, "
        f"floor {a.get('vin_min')} V), latency {a.get('delay_min_lag')}-{a.get('delay_max_lag')} physics steps. "
        'Everything (masses per body, joint ranges, HOME, servo parameters, collision geometry) is in `plant.json`.', '',
        *([f"`{m['model']['file']}` + `assets/` is the robot as the task simulated it (masses, contacts, HOME); "
           f"open it with `python -m mujoco.viewer --mjcf {m['model']['file']}`. The servo model is not in it: see `plant.json`.", '']
          if m.get('model') else []),
        '## Evaluation', '',
        f"Family `{ev['family']}` eval v{ev['version']}, {ev['num_envs']} robots, seed {ev['seed']}. "
        'Full results and test conditions in `eval.json`.', '',
        '| metric | value |', '|---|---|', *[f"| {k} | {v} |" for k, v in rows], '',
        *(['## Replay', '', "The family's replay scenarios back to back, fixed camera, real time.", '',
           f"<video src=\"https://huggingface.co/{m['repo']}/resolve/main/replay.mp4\" controls muted loop width=\"640\"></video>", '']
          if has_replay else []),
        '## Training', '',
        f"Task `{t['task']}`, trained to iteration {t['iterations']}"
        + (f" ({t['position']['env_steps']:,} env steps, from `{t['position']['checkpoint']}`)" if t.get('position', {}).get('checkpoint') else '')
        + (f", code {code.get('repo')} @ {code.get('commit')}" if code.get('commit') else '') + '.',
        *(["`checkpoint.pt` is that training checkpoint (actor, critic, optimizer), to continue training from."]
          if has_checkpoint else []),
        f"Lineage: {' <- '.join(t['lineage'])}." if t['lineage'] else '',
        *([''] + [f"> Warning: {w}" for w in t['warnings']] if t.get('warnings') else []),
        '', *training_sections(t),
        "The complete configs (every term's parameters) are in `training.json`.", '',
    ]
    return '\n'.join(lines)

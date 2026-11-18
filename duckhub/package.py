"""Build one policy package: the folder that becomes one tagged revision of a Hub model repo.
See SPEC.md. Called by `duckhub package`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from duckhub.policy import sha256
from duckhub.version import git

SPEC_VERSION = 2


# -- params/*.yaml are written with python tags; read them as plain data ------------------------
class _Loader(yaml.SafeLoader):
    pass


def _python_tag(loader, suffix, node):
    kind, _, name = suffix.partition(':')
    if kind == 'tuple':
        return loader.construct_sequence(node, deep=True)
    if kind == 'name':
        return f"<{name}>"
    if kind.startswith('object'):  # enums and dataclass instances: keep the class and its state
        val = (loader.construct_sequence(node, deep=True) if isinstance(node, yaml.SequenceNode)
               else loader.construct_mapping(node, deep=True) if isinstance(node, yaml.MappingNode)
               else loader.construct_scalar(node))
        return {'<class>': name, 'value': val}
    return loader.construct_scalar(node) if isinstance(node, yaml.ScalarNode) else None


_Loader.add_multi_constructor('tag:yaml.org,2002:python/', _python_tag)


def scrub_home(obj):
    """Replace local home-directory prefixes in strings with '~' (no machine paths in public packages)."""
    home = str(Path.home())
    if isinstance(obj, str):
        return '~' + obj[len(home):] if obj.startswith(home) else obj
    if isinstance(obj, list):
        return [scrub_home(v) for v in obj]
    if isinstance(obj, dict):
        return {k: scrub_home(v) for k, v in obj.items()}
    return obj


def load_cfg(path: Path) -> dict:
    return yaml.load(path.read_text(), Loader=_Loader)


def _strip_func(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != 'func'}


def final_weights(env: dict, env_steps: int) -> dict:
    """Reward weights the policy ended training with: the configured weight, overridden by the
    last curriculum stage reached (stages are in env steps = iterations x num_steps_per_env)."""
    w = {k: v['weight'] for k, v in (env.get('rewards') or {}).items()}
    for c in (env.get('curriculum') or {}).values():
        p = c.get('params') or {}
        if 'reward_name' in p and 'weight_stages' in p:
            reached = [s for s in p['weight_stages'] if env_steps > s['step']]
            if reached:
                w[p['reward_name']] = reached[-1]['weight']
    return w


def summarize(env: dict, agent: dict, env_steps: int) -> dict:
    """The parts people compare. Everything else stays in env/agent below."""
    robot = env['scene']['entities']['robot']
    act = robot['articulation']['actuators'][0]
    return {
        'env_steps': env_steps,
        'rewards_final': final_weights(env, env_steps),
        'rewards_final_source': 'replayed from the curriculum plan',
        'plant': {
            'spec_fn': robot.get('spec_fn'),
            'actuator_params': Path(act.get('json_path') or '').name or None,
            'kp': act.get('kp_fw'), 'kd': act.get('kd_fw'), 'max_pwm': act.get('max_pwm'),
            'vin_range': act.get('vin_range'), 'vin_drop_gain_range': act.get('vin_drop_gain_range'),
            'vin_min': act.get('vin_min'),
            'delay_steps': [act.get('delay_min_lag'), act.get('delay_max_lag')],
            'home': robot['init_state']['joint_pos'],
            'control_hz': round(1.0 / (env['sim']['mujoco']['timestep'] * env['decimation'])),
        },
        'randomization': {k: v.get('params') for k, v in (env.get('events') or {}).items()},
        'rewards_initial': {k: v['weight'] for k, v in (env.get('rewards') or {}).items()},
        'terminations': sorted((env.get('terminations') or {}).keys()),
        'curriculum': {k: v.get('params') for k, v in (env.get('curriculum') or {}).items()},
        'commands': {k: _strip_func(v) for k, v in (env.get('commands') or {}).items()},
        'ppo': {'actor': agent['actor']['hidden_dims'], 'algorithm': agent.get('algorithm'),
                'num_steps_per_env': agent['num_steps_per_env'], 'seed': agent['seed']},
    }


def git_info(repo: Path) -> dict:
    g = lambda *a: git(repo, *a)
    return {'commit': g('rev-parse', '--short=9', 'HEAD'), 'branch': g('branch', '--show-current'),
            'dirty': bool(g('status', '--porcelain', '--untracked-files=no'))}


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


def acceptance_section(e: dict) -> list[str]:
    """The family's acceptance gates, when its eval records them (metrics.acceptance)."""
    acc = (e.get('metrics') or {}).get('acceptance')
    if not acc:
        return []
    verdict = 'ACCEPTED: passes every gate.' if acc['accepted'] else 'CANDIDATE: fails at least one gate below.'
    rows = [f"| {k} | {g['rule']} | {g['value']} | {'pass' if g['pass'] else ('fail' if g['pass'] is False else '-')} |"
            for k, g in acc['gates'].items()]
    return ['### Acceptance', '', f'**{verdict}**', '', '| gate | rule | value | |', '|---|---|---|---|', *rows, '']


def card(m: dict, t: dict, e: dict, plant: dict, has_replay: bool = False, has_checkpoint: bool = False) -> str:
    rows = list(e['headline'].items())
    ev = e['eval']
    meta = {
        'license': 'apache-2.0',
        'library_name': 'onnx',
        'tags': ['microduck', f"microduck-family-{m['family']}", f"microduck-plant-{m['plant']['plant_hash']}",
                 f"microduck-eval-{ev['family']}-{ev['version']}",
                 'robotics', 'reinforcement-learning'],
        'model-index': [{'name': m['repo'].split('/')[1], 'results': [{
            'task': {'type': 'reinforcement-learning', 'name': m['description']},
            'dataset': {'name': f"{ev['family']} eval v{ev['version']}", 'type': 'microduck-sim'},
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
        f"- Servos {m['robot']['servos']}, Kp {m['gains']['kp']} / Kd {m['gains']['kd']} written at start, "
        f"{m['robot']['control_hz']} Hz.",
        f"- HOME (default_joint_pos, policy joint order): `{m['home']}`",
        f"- Twist: {m['command']['twist']}",
        *[f"- {n}" for n in m['runtime_notes']],
        f"- `robotctl policy load {m['slot']} {m['repo']}`" if m.get('slot') else '', '',
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
        *acceptance_section(e),
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


def build(*, onnx: Path, run_dir: Path, eval_res: dict, meta: dict, plant: dict, out: Path,
          replay: Path | None = None, checkpoint: Path | None = None) -> Path:
    from duckhub.contract import check_onnx
    from duckhub.training import checkpoint_position, live_final, unknown_curricula

    check_onnx(onnx, plant['robot']['joints']['home_rad'])
    for k in ('repo', 'owner', 'name', 'servos', 'description', 'command'):
        if k not in meta:
            raise SystemExit(f"policy.yaml is missing '{k}' (see AGENTS.md)")
    if not meta['repo'].endswith(f"-{meta['owner']}"):
        raise SystemExit(f"repo {meta['repo']} must end with -{meta['owner']} (<org>/microduck-<robot>-<family>-<owner>)")
    env = load_cfg(run_dir / 'params/env.yaml')
    agent = load_cfg(run_dir / 'params/agent.yaml')
    warnings = []
    if checkpoint:
        position = checkpoint_position(checkpoint)
        if position['env_steps'] is None:
            position['env_steps'] = position['iteration'] * agent['num_steps_per_env']
            warnings.append('checkpoint has no env step counter: estimated from its iteration')
    elif 'iterations' in meta:
        position = {'checkpoint': None, 'iteration': meta['iterations'],
                    'env_steps': meta['iterations'] * agent['num_steps_per_env']}
        warnings.append("no --checkpoint: training position taken from policy.yaml 'iterations'")
    else:
        raise SystemExit("pass --checkpoint (the .pt the ONNX was exported from), or set 'iterations' in policy.yaml")
    summary = summarize(env, agent, position['env_steps'])
    final, w = live_final(eval_res['task'], position['env_steps'], env)
    warnings += w
    if final:
        summary['rewards_final'] = final['rewards']
        summary['rewards_final_source'] = f"task curricula applied at env step {position['env_steps']}"
        summary['curriculum_final'] = final['curriculum_targets']
    else:
        unknown = unknown_curricula(env)
        if unknown:
            warnings.append(f"curriculum terms the plan replay cannot read (their effect is missing): {', '.join(unknown)}")
    from duckhub.bake import bake
    out.mkdir(parents=True, exist_ok=True)
    model = bake(eval_res['task'], out)
    shutil.copy(onnx, out / 'policy.onnx')
    if replay:
        shutil.copy(replay, out / 'replay.mp4')
    if checkpoint:
        shutil.copy(checkpoint, out / 'checkpoint.pt')   # so others can continue training from exactly here

    family = eval_res['eval']['family']
    home = plant['robot']['joints']['home_rad']
    joint_order = meta.get('joint_order') or list(home)
    act = plant['robot']['actuators'][0]
    manifest = {
        'schema_version': 2, 'model_api': 1, 'obs_len': meta.get('obs_len', 61), 'action_len': len(joint_order),
        'robot': {'model': 'microduck', 'hw_rev': 1, 'servos': meta['servos'],
                  'control_hz': plant['robot']['control']['hz']},
        'name': meta['name'], 'kind': meta.get('kind', 'perpetual'), 'slot': meta.get('slot'),
        'entry_pose': meta.get('entry_pose', 'standing'), 'description': meta['description'],
        'command': {'encoding': 'constant', 'idle': [0.0, 0.0, 0.0], **meta['command']},
        'action_scale': meta.get('action_scale', 1.0),
        # ours; the daemon ignores unknown fields (Pollen docs/policy-manifest.md)
        'spec_version': SPEC_VERSION, 'repo': meta['repo'], 'owner': meta['owner'], 'family': family,
        'plant': {'plant_hash': plant['plant_hash'], 'task': plant['task_modelling']['task']},
        'gains': {'kp': act.get('kp_fw'), 'kd': act.get('kd_fw')},
        'joint_order': joint_order, 'home': [home[j] for j in joint_order],
        'runtime_notes': meta.get('runtime_notes', []),
        'onnx_sha256': sha256(onnx),
        'model': {'file': model['file'], 'sha256': model['sha256']},
    }
    training = {
        'spec_version': SPEC_VERSION, 'task': eval_res['task'], 'run': agent['run_name'],
        'iterations': position['iteration'], 'position': position, 'lineage': meta.get('lineage', []),
        'warnings': warnings,
        'code': {'repo': meta.get('code_repo'), **git_info(Path(meta['code_dir']))} if meta.get('code_dir') else {'repo': meta.get('code_repo')},
        'summary': summary, 'env': env, 'agent': agent,
    }
    assert eval_res['onnx_sha256'] == manifest['onnx_sha256'], 'eval.json was made from a different ONNX'
    plant, manifest, training, eval_res = (scrub_home(x) for x in (plant, manifest, training, eval_res))
    for name, obj in (('plant.json', plant), ('manifest.json', manifest), ('training.json', training), ('eval.json', eval_res)):
        (out / name).write_text(json.dumps(obj, indent=1, default=str) + '\n')
    (out / 'README.md').write_text(card(manifest, training, eval_res, plant, (out / 'replay.mp4').is_file(), (out / 'checkpoint.pt').is_file()))
    return out

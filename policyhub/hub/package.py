"""Build one policy package: the folder that becomes one tagged revision of a Hub model repo.
See SPEC.md. Called by `policyhub package`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from policyhub.eval.policy import sha256
from policyhub.hub.card import card
from policyhub.hub.training import git_info, load_cfg, summarize


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


def build(*, cfg, onnx: Path, run_dir: Path, eval_res: dict, meta: dict, plant: dict, out: Path,
          replay: Path | None = None, checkpoint: Path | None = None) -> Path:
    from policyhub.hub.training import checkpoint_position, live_final, unknown_curricula

    for k in ('repo', 'owner', 'description', 'command'):
        if k not in meta:
            raise SystemExit(f"policy.yaml is missing '{k}' (see AGENTS.md)")
    if not meta['repo'].endswith(f"-{meta['owner']}"):
        raise SystemExit(f"repo {meta['repo']} must end with -{meta['owner']} (one repo per person: <org>/<name>-<owner>)")
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
    from policyhub.robot.bake import bake
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
        'robot': cfg.robot, 'repo': meta['repo'], 'owner': meta['owner'],
        'family': family, 'description': meta['description'], 'command': meta['command'],
        'control_hz': plant['robot']['control']['hz'],
        'plant': {'plant_hash': plant['plant_hash'], 'task': plant['task_modelling']['task']},
        'gains': {'kp': act.get('kp_fw'), 'kd': act.get('kd_fw')},
        'joint_order': joint_order, 'home': [home[j] for j in joint_order],
        'runtime_notes': meta.get('runtime_notes', []),
        'onnx_sha256': sha256(onnx),
        'model': {'file': model['file'], 'sha256': model['sha256']},
    }
    training = {
        'task': eval_res['task'], 'run': agent['run_name'],
        'iterations': position['iteration'], 'position': position, 'lineage': meta.get('lineage', []),
        'warnings': warnings,
        'code': {'repo': meta.get('code_repo'), **git_info(Path(meta['code_dir']), run_dir)} if meta.get('code_dir') else {'repo': meta.get('code_repo')},
        'summary': summary, 'env': env, 'agent': agent,
    }
    assert eval_res['onnx_sha256'] == manifest['onnx_sha256'], 'eval.json was made from a different ONNX'
    plant, manifest, training, eval_res = (scrub_home(x) for x in (plant, manifest, training, eval_res))
    for name, obj in (('plant.json', plant), ('manifest.json', manifest), ('training.json', training), ('eval.json', eval_res)):
        (out / name).write_text(json.dumps(obj, indent=1, default=str) + '\n')
    (out / 'README.md').write_text(card(manifest, training, eval_res, plant, (out / 'replay.mp4').is_file(), (out / 'checkpoint.pt').is_file()))
    return out

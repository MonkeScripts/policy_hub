"""Compare a package with what is on the Hub, then (only when asked) push it as a tagged revision.

The comparison is the gate: the package's eval headline against the latest revision in the same
repo, metric by metric, using the family's RANKING. Different eval versions are not compared.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from duckhub import families


def _score(rule, v):
    """Larger is better."""
    if v is None:
        return None
    if rule == 'higher':
        return v
    if rule == 'lower':
        return -v
    kind, target = rule
    assert kind == 'closer'
    return -abs(v - target)


def compare(new: dict, old: dict, ranking: dict) -> tuple[list[str], list[str]]:
    lines, worse = [], []
    for k, rule in ranking.items():
        a, b = old['headline'].get(k), new['headline'].get(k)
        sa, sb = _score(rule, a), _score(rule, b)
        mark = ''
        if sa is not None and sb is not None and abs(sb - sa) > 1e-9:
            mark = 'better' if sb > sa else 'WORSE'
            if mark == 'WORSE':
                worse.append(k)
        lines.append(f"  {k:28s} {a!s:>8} -> {b!s:<8} {mark}")
    return lines, worse


def current(repo: str, filename: str = 'eval.json') -> dict | None:
    try:
        return json.loads(Path(hf_hub_download(repo, filename)).read_text())
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None


def _tags(repo: str) -> set[str]:
    try:
        return {t.name for t in HfApi().list_repo_refs(repo).tags}
    except RepositoryNotFoundError:
        return set()


def _flat(d, prefix=''):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _flat(v, f"{prefix}{k}.")
    elif isinstance(d, list) and d and isinstance(d[0], dict):
        for i, v in enumerate(d):
            yield from _flat(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], d


def plant_diff(new: dict, old: dict) -> list[str]:
    """What changed in the robot (the hashed part) between two plant.json files."""
    a, b = dict(_flat(old['robot'])), dict(_flat(new['robot']))
    return [f"  {k}: {a.get(k)} -> {b.get(k)}" for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)]


def publish(pkg: Path, tag: str, yes: bool = False, force: bool = False, private: bool = False,
            plant_changed: bool = False) -> None:
    manifest = json.loads((pkg / 'manifest.json').read_text())
    new = json.loads((pkg / 'eval.json').read_text())
    repo, fam = manifest['repo'], families.load(new['eval']['family'])
    old = current(repo)
    print(f"{repo}  new tag {tag}  (owner {manifest.get('owner')})")
    old_manifest = current(repo, 'manifest.json')
    if old_manifest and old_manifest.get('owner', manifest.get('owner')) != manifest.get('owner'):
        raise SystemExit(f"stopped: {repo} belongs to {old_manifest.get('owner')}, this package to "
                         f"{manifest.get('owner')}. Publish to your own repo.")
    if tag in _tags(repo):
        raise SystemExit(f"stopped: {repo} already has tag {tag}; tags are never moved or reused")
    if old is None:
        print('  no published version yet: this becomes the first')
    elif old['eval']['version'] != new['eval']['version'] or old['eval']['family'] != new['eval']['family']:
        print(f"  published eval is {old['eval']['family']} v{old['eval']['version']}, this is "
              f"{new['eval']['family']} v{new['eval']['version']}: not comparable, re-run the eval on the published ONNX")
        if not force:
            raise SystemExit('stopped (use --force to publish anyway)')
    else:
        lines, worse = compare(new, old, fam.RANKING)
        print('\n'.join(lines))
        if worse and not force:
            raise SystemExit(f"stopped: worse than the published version on {', '.join(worse)} (--force to publish anyway)")
    new_plant, old_plant = json.loads((pkg / 'plant.json').read_text()), current(repo, 'plant.json')
    if old_plant and old_plant['plant_hash'] != new_plant['plant_hash']:
        print(f"  PLANT CHANGED {old_plant['plant_hash']} -> {new_plant['plant_hash']}:")
        print('\n'.join(plant_diff(new_plant, old_plant)) or '  (no field differs; hash scheme changed)')
        if not plant_changed:
            raise SystemExit('stopped: the robot differs from the published version; check the lines above, '
                             'then re-run with --plant-changed if they are intended')
    for w in new_plant.get('warnings', []):
        print('  warning:', w)
    files = sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob('*') if p.is_file())
    print('  files:', ', '.join(files))
    if not yes:
        print('dry run: nothing uploaded (add --yes to publish)')
        return
    api = HfApi()
    api.create_repo(repo, repo_type='model', private=private, exist_ok=True)
    info = api.upload_folder(repo_id=repo, folder_path=str(pkg),
                             commit_message=f"{tag}: {manifest['name']} ({new['eval']['family']} eval v{new['eval']['version']})")
    api.create_tag(repo, tag=tag, revision=info.oid)
    print(f"published https://huggingface.co/{repo}/tree/{tag}")

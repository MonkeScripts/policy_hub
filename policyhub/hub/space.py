"""The robot's ranking Space: a static page (space.html) that reads its robot, org and title from
space.json and lists the org's public policies tagged <robot>-family-*. Nothing else is stored there."""

from __future__ import annotations

import json
from pathlib import Path

PAGE = Path(__file__).with_name('space.html')


def files(cfg) -> dict[str, str]:
    title = cfg.title or f"{cfg.robot} policy hub"
    readme = f"""---
title: {title}
emoji: {cfg.emoji}
colorFrom: gray
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Rankings of {cfg.robot} policies by family
tags:
- {cfg.robot}
- robotics
- reinforcement-learning
---

# {title}

Ranks {cfg.robot} policies published with policyhub. The page reads public Hub model repos in
{cfg.org} tagged `{cfg.robot}-family-*` straight from the browser; nothing is stored in this Space.
Written by `policyhub space`: change policyhub.yaml and re-run it instead of editing the Space.
"""
    return {'index.html': PAGE.read_text(), 'README.md': readme,
            'space.json': json.dumps({'robot': cfg.robot, 'org': cfg.org, 'title': title}, indent=1) + '\n'}


def publish_space(cfg, upload: bool = False) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    if not (cfg.space and cfg.org):
        raise SystemExit('policyhub.yaml needs `org` and `space` for the Space')
    new = files(cfg)
    print(f"Space {cfg.space}: robot {cfg.robot}, policies from {cfg.org} tagged {cfg.robot}-family-*")
    for name, text in new.items():
        try:
            old = Path(hf_hub_download(cfg.space, name, repo_type='space')).read_text()
            state = 'unchanged' if old == text else 'changed'
        except RepositoryNotFoundError:
            state = 'new Space'
        except EntryNotFoundError:
            state = 'new file'
        print(f"  {name:12s} {state}")
    if not upload:
        print('preview only: nothing uploaded (add --upload to create or update the Space)')
        return
    api = HfApi()
    api.create_repo(cfg.space, repo_type='space', space_sdk='static', exist_ok=True)
    api.create_commit(cfg.space, repo_type='space', commit_message='policyhub space',
                      operations=[CommitOperationAdd(n, t.encode()) for n, t in new.items()])
    print(f"updated https://huggingface.co/spaces/{cfg.space}")

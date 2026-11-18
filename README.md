# policyhub

Evaluate, package and publish RL policies to the Hugging Face Hub, one family at a time, so
policies for the same robot can be ranked against each other in that robot's Space.

Training stays in your own repo. policyhub is installed into that repo's environment, reads its
mjlab tasks from there, and never changes it.

```
policyhub/
├── policyhub/   the tool: config, policy (ONNX), sim (fixed eval conditions), plant (read out of the
│                task), bake (the robot as plain MJCF), package (Hub folder), publish, space (the
│                robot's ranking page), cli
├── families/    one folder per action; each has eval.py (its tests and ranking) + README
└── SPEC.md      the package format
```

## Use

Add a `policyhub.yaml` at the training repo's root:

```yaml
robot: myrobot                 # tags: myrobot, myrobot-family-<f>, myrobot-plant-<h>, myrobot-eval-<f>-<v>
org: my-hf-org                 # where the policy repos live
space: my-hf-org/myrobot-hub   # this robot's ranking Space
title: My Robot Policy Hub     # optional: the Space's name
emoji: 🤖                      # optional: the Space's icon
mass_g: [300, 2000]            # optional: plausible total mass (catches typos in the task)
```

Then, from the training repo (AGENTS.md has the full workflow and rules):

```
uv run policyhub space                                        # dry run; --yes creates/updates the robot's Space (once)
uv run policyhub families
uv run policyhub check --task <TASK> --family velstand        # contract, plant, what collides
uv run policyhub bake --task <TASK> --out out/bake            # robot.xml + assets/, opens in any MuJoCo viewer
uv run policyhub eval velstand --task <TASK> --onnx x.onnx --out eval.json
uv run policyhub package --onnx x.onnx --checkpoint <run>/model_N.pt --eval eval.json --run-dir <run> \
    --meta policy.yaml --out out/<repo-name>
uv run policyhub publish out/<repo-name> --tag v1             # dry run: compares, uploads nothing
uv run policyhub publish out/<repo-name> --tag v1 --yes       # push + tag
```

`policy.yaml` is the only hand-written file per package: repo (`<org>/<name>-<owner>`), owner,
name, description, servos, what the command inputs mean, runtime notes, lineage.

## Adding a family

`families/<name>/eval.py` defines `FAMILY`, `VERSION`, `RANKING` (per headline metric:
`"higher"`, `"lower"` or `("closer", target)`) and `run(task, onnx_path) -> dict` returning the
eval.json content. Everyone ranking that family uses the same file.

## Using a package without this tool

Every package carries `robot.xml` + `assets/`: the robot exactly as the task simulated it. Download
only the files you need:

```python
from huggingface_hub import snapshot_download
d = snapshot_download('<org>/<repo>', revision='v1', allow_patterns=['policy.onnx', 'manifest.json'])
d = snapshot_download('<org>/<repo>', revision='v1', allow_patterns=['robot.xml', 'assets/*', 'plant.json'])
```

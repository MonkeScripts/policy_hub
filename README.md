# policyhub

Evaluate, package and publish RL policies to the Hugging Face Hub, one family at a time, so
policies for the same robot can be ranked against each other in that robot's Space.

Training stays in your own repo. policyhub is installed into that repo's environment, reads its
mjlab tasks from there, and never changes it.

```
policyhub/
├── policyhub/
│   ├── cli.py        the commands below
│   ├── config.py     policyhub.yaml
│   ├── version.py    which policyhub produced a result
│   ├── robot/        what the training repo's task builds: contract (the rules `check` enforces),
│   │                 plant (plant.json, plant_hash), bake (robot.xml + assets/)
│   ├── eval/         a family's exam: families (finds eval.py), sim (env under fixed conditions),
│   │                 policy (the ONNX); family eval.py files build on sim and policy
│   └── hub/          Hugging Face: package (the folder), training (training.json), card (README.md),
│                     publish (compare + push), space (the robot's ranking page)
├── examples/microduck/   a worked example: policyhub.yaml + families/ (walk, velstand)
└── SPEC.md           the package format
```

## Use

Add a `policyhub.yaml` at the training repo's root:

```yaml
robot: myrobot                 # tags: myrobot, myrobot-family-<f>, myrobot-plant-<h>, myrobot-eval-<f>-<v>
org: my-hf-org                 # where the policy repos live
space: my-hf-org/myrobot-hub   # this robot's ranking Space
title: My Robot Policy Hub     # optional: the Space's name
emoji: 🤖                      # optional: the Space's icon
families: families             # this robot's evals, relative to this file
mass_g: [300, 2000]            # optional: plausible total mass (catches typos in the task)
```

Then, from the training repo (AGENTS.md has the full workflow and rules):

```
uv run policyhub space                                        # dry run; --yes creates/updates the robot's Space (once)
uv run policyhub check --task <TASK> --family velstand        # contract, plant, what collides
uv run policyhub bake --task <TASK> --out out/bake            # robot.xml + assets/, opens in any MuJoCo viewer
uv run policyhub eval velstand --task <TASK> --onnx x.onnx --out eval.json
uv run policyhub package --onnx x.onnx --checkpoint <run>/model_N.pt --eval eval.json --run-dir <run> \
    --meta policy.yaml --out out/<repo-name>
uv run policyhub publish out/<repo-name> --tag v1             # dry run: compares, uploads nothing
uv run policyhub publish out/<repo-name> --tag v1 --yes       # push + tag
```

`policy.yaml` is the only hand-written file per package: repo (`<org>/<name>-<owner>`), owner,
description, what the command inputs mean, runtime notes, lineage.

## Families

A family is one action (walk, stand up, ...) and its exam. Each robot owns its families: everyone
ranking that family in the robot's Space runs the same file, so results compare. `<families>/<name>/eval.py` defines `FAMILY`, `VERSION`, `RANKING` (per headline metric:
`"higher"`, `"lower"` or `("closer", target)`) and `run(task, onnx_path) -> dict` returning the
eval.json content, built with `policyhub.eval.sim` and `policyhub.eval.policy`. `examples/microduck/families`
has two to copy from; change a test, condition or threshold and bump `VERSION`.

## Using a package without this tool

Every package carries `robot.xml` + `assets/`: the robot exactly as the task simulated it. Download
only the files you need:

```python
from huggingface_hub import snapshot_download
d = snapshot_download('<org>/<repo>', revision='v1', allow_patterns=['policy.onnx', 'manifest.json'])
d = snapshot_download('<org>/<repo>', revision='v1', allow_patterns=['robot.xml', 'assets/*', 'plant.json'])
```

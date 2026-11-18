# microduck-hub

Evaluate, package and publish Microduck policies to the Hugging Face Hub, one family at a time.
Training stays in your own repo (Pollen's microduck_rl template, your own tasks); this tool is
installed into that repo's environment and reads its tasks from there.

```
microduck-hub/
├── policyhub/           the tool: policy (ONNX), sim (fixed eval conditions), plant (read out of
│                      the task), bake (the robot as plain MJCF), package (Hub folder),
│                      publish (compare with the Hub, push), cli
├── families/          one folder per action; each has eval.py (its tests and ranking) + README
│   ├── velstand/
│   └── walk/
├── space/             the ranking page (static Hugging Face Space)
└── SPEC.md            the package format
```

## Use

`uv sync` in the training repo installs it (see AGENTS.md for the full workflow and rules):

```
uv run policyhub families
uv run policyhub check --task <TASK> --family velstand          # contract, plant, what collides
uv run policyhub check --task <TASK> --onnx exports/x.onnx         # the firmware contract (package enforces it too)
uv run policyhub bake --task <TASK> --out out/bake               # robot.xml + assets/, opens in any MuJoCo viewer
uv run policyhub eval velstand --task <TASK> --onnx exports/x.onnx --out eval.json
uv run policyhub package --onnx exports/x.onnx --checkpoint logs/rsl_rl/<exp>/<run>/model_N.pt --eval eval.json --run-dir logs/rsl_rl/<exp>/<run> \
    --meta policy.yaml --out out/microduck-<robot>-velstand-<you>
uv run policyhub publish out/microduck-<robot>-velstand-<you> --tag v1          # dry run: compares, uploads nothing
uv run policyhub publish out/microduck-<robot>-velstand-<you> --tag v1 --yes    # push + tag
```

`policy.yaml` is the only hand-written file: repo (`duckyduck369/microduck-<robot>-<family>-<owner>`), owner, name, description, servos, slot, what the
command inputs mean, runtime notes, lineage, where the code lives.

## Adding a family

`families/<name>/eval.py` defines `FAMILY`, `VERSION`, `RANKING` (per headline metric:
`"higher"`, `"lower"` or `("closer", target)`) and `run(task, onnx_path) -> dict` returning the
eval.json content. Everyone ranking that family uses the same file.

## Using a package without this tool

Every package carries `robot.xml` + `assets/`: the robot exactly as the task simulated it. Tools that
only need some files download just those:

```python
from huggingface_hub import snapshot_download
d = snapshot_download('duckyduck369/microduck-hd1910-velstand-djbob', revision='v5',
                      allow_patterns=['policy.onnx', 'manifest.json'])            # firmware: ~0.8 MB
d = snapshot_download('duckyduck369/microduck-hd1910-velstand-djbob', revision='v5',
                      allow_patterns=['robot.xml', 'assets/*', 'plant.json'])      # viewers: ~22 MB
```

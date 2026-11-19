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

## Add your robot

policyhub works with robots built like the Microduck template: an mjlab task trained with rsl_rl,
BAM servo actuators, and the template's command and event names. The easiest start is a copy of a
template repo (esp_microduck_rl is one) with your robot in place of Microduck. A stock mjlab robot
(Unitree Go1, G1) passes most of the contract but not step 3's C3, C6 and C9.

Do these in order; each says how you know it worked.

**If you copied a template repo**, steps 1, 2, 4, 5, 9 and 10 already exist in your copy, filled in
for Microduck. Everything will run without complaint and publish under Microduck's name, so before
anything else replace:
- in `policyhub.yaml`: `robot`, `org`, `space`, `title`, `emoji`, `mass_g`;
- in `families/*/eval.py`: every constant in step 9's table, and `VERSION` back to 1.0;
  delete `families/velstand` unless your robot gets up after a fall;
- the task names, and in the repo's README the repo names and `policy.yaml` example (step 6).

**Once per training repo**

1. **Add the tool.** `git submodule add https://github.com/MonkeScripts/policy_hub tools/policyhub`,
   then in `pyproject.toml`: `policyhub = { path = "tools/policyhub", editable = true }` under
   `[tool.uv.sources]` and `"policyhub"` in the `dev` dependency group. Tell people to clone with
   `--recursive`. *Works when* `uv sync && uv run policyhub --help` lists the commands.
2. **Write `policyhub.yaml`** at the repo root:
   ```yaml
   robot: myrobot                 # prefixes every tag (myrobot-family-<f>, ...): choose once, never rename
   org: my-hf-org                 # the Hugging Face org that holds the policy repos
   space: my-hf-org/myrobot-hub   # this robot's ranking Space
   title: My Robot Policy Hub     # optional: the Space's name
   emoji: 🤖                      # optional: the Space's icon
   families: families             # this robot's evals, relative to this file
   mass_g: [300, 2000]            # optional: plausible total mass (catches typos in the task)
   ```
   *Works when* `uv run policyhub check --task <TASK>` gets past the config (it names any bad key).
3. **Make each task meet the contract** (AGENTS.md has the full table, C1-C13). The rules a robot not
   built from the template usually breaks:
   - C3: `actuators[0]` is a BAM servo config (`bam.mjlab.BamActuatorCfg` or a subclass: gain,
     latency, servo model, supply). The servo model is BAM's fit of *your* servo (`json_path`), so a
     different servo means identifying it with BAM first;
   - C5: joints the policy does not drive are named `passive_*`;
   - C6: the action term `joint_pos` has scale 1 (target = HOME + action);
   - C9: an event named `foot_friction` (C8's `twist` command is mjlab's standard velocity command);
   - the play config (what evals run) builds the same robot as the train config.

   *Works when* `uv run policyhub check --task <TASK>` prints `contract OK` and the plant hash, and
   every task that builds the same robot prints the same hash. Read its collision table too: it lists
   every geom that can touch something, including unnamed CAD meshes the task's config misses.
4. **Export to ONNX with the observation normalizer baked in**, its input exactly the `actor`
   observation group (C7). policyhub only ever runs the ONNX. The template's `scripts/export.py`
   does this. *Works when* `policyhub eval` (step 9) runs it without "observations do not match the
   ONNX input".
5. **Git-ignore `logs/`.** Every policyhub output, `policy.yaml` included, goes under `logs/hub/`.
6. **Write the repo's own publishing docs**: the export → eval → package → publish commands, a
   `policy.yaml` example with this robot's command names, and the repo name for each family:
   `<org>/<name>-<owner>`, with `<name>` fixed here per family (e.g. `myrobot-v1-walk`, so
   `my-hf-org/myrobot-v1-walk-<you>`). Each person
   gets one repo per family under that name, for good. esp_microduck_rl's README is a model.

**Once per robot, on Hugging Face**

7. **Create the org** (or use one) and have an admin add everyone who will publish. Each of them
   logs in once per machine: `hf auth login`.
8. **Create the Space** (logged in as an org member): `uv run policyhub space` previews it,
   `--upload` creates it. *Works when*
   the Space's page opens (empty until the first policy is published).

**Once per family (an action and its exam)**

9. **Write the eval.** `<families>/<name>/eval.py` defines `FAMILY`, `VERSION`, `RANKING` (per
   headline metric: `"higher"`, `"lower"` or `("closer", target)`), `NEEDS_PRONE` if it tests getting
   up, `replay(task, onnx_path, path)` (needed for `eval --replay`, the card's video), and `run(task, onnx_path) -> dict` (the eval.json
   content), built on `policyhub.eval.sim` and `policyhub.eval.policy`. Start from
   `examples/microduck/families/walk`, but every condition in it is Microduck's; set your own:

   | constant in `walk/eval.py` | Microduck's value | why it is robot-specific |
   |---|---|---|
   | `TRACKING` | yaw 0.3 / 1.0 rad/s, vx 0.1-0.3 m/s | speeds the robot can reach |
   | `LOW_SUPPLY` | 4.5-5.5 V, sag 0.3-0.5 V/N·m | its battery |
   | `LOW_FRICTION` | 0.15-0.3 | its feet and floors |
   | `PUSH` | ±0.3 m/s every 2-3 s | its size and mass |
   | `REPLAY` | vx 0.2 / 0.3, yaw ±1.0, pushes 0.3 m/s | the video's scenarios, same reasons |
   | replay camera | 0.9 m away (`make_env(render=True)`) | a 25 cm robot: pass `camera=dict(distance=...)` |

   Changing any test, condition or threshold later means a new `VERSION`; versions never compare.
   *Works when* `uv run policyhub eval <name> --task <TASK> --onnx <file> --out <file>` writes
   `eval.json`.
10. **Write the family's `README.md`**: what the action is, what each ranked metric means.

**Per policy**

11. **Write `policy.yaml`** (repo, owner, description, what each command input means; AGENTS.md
    has the optional fields), then:
    ```
    uv run policyhub check --task <TASK> --family <name>
    uv run policyhub eval <name> --task <TASK> --onnx x.onnx --out logs/hub/x.eval.json --replay logs/hub/x.mp4
    uv run policyhub package --onnx x.onnx --checkpoint logs/rsl_rl/<exp>/<run>/model_N.pt --eval logs/hub/x.eval.json \
        --replay logs/hub/x.mp4 --policy-yaml logs/hub/policy.yaml --out logs/hub/<repo-name>
    uv run policyhub publish logs/hub/<repo-name>                # preview: compares, uploads nothing
    ```
    The checkpoint stays in the run folder mjlab wrote (its `params/` are read). Read the preview,
    then run the same publish command with `--upload`. AGENTS.md has the full workflow and rules.

**On the robot (outside policyhub)**

12. **A runtime that reads the package**: `policy.onnx`, and from `manifest.json` the joint order,
    HOME, gains (written to the servos at boot) and control rate; it sends `HOME + action` as
    targets. SPEC.md lists every manifest field. policyhub does not provide one.

## Using a package without this tool

Every package carries `robot.xml` + `assets/`: the robot exactly as the task simulated it. Download
only the files you need:

```python
from huggingface_hub import snapshot_download
d = snapshot_download('<org>/<repo>', revision='v1', allow_patterns=['policy.onnx', 'manifest.json'])
d = snapshot_download('<org>/<repo>', revision='v1', allow_patterns=['robot.xml', 'assets/*', 'plant.json'])
```

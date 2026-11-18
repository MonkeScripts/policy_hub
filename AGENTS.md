# AGENTS.md: using policyhub

policyhub evaluates a trained policy, packages it, and publishes it to the Hugging Face Hub, so
policies of the same family for the same robot can be ranked fairly against each other. It is
installed into the training repo's environment and runs there. It never trains and never changes
the training repo.

Words used below:
- **family**: one action. Everyone's velstand policies are one family; a walker is another family,
  even on the same robot. Each family has one shared eval, `<families>/<name>/eval.py`, in the
  folder `families` in policyhub.yaml names. Each robot owns its families.
- **plant**: one physical robot. `plant.json` is read out of the task, never written by hand;
  `plant_hash` is the same for every task that builds the same robot.
- **package**: the folder that becomes one tagged version (`v1`, `v2`, ...) of a Hub model repo.
- **robot**: `robot` in `policyhub.yaml` at the training repo's root. It prefixes every tag, and
  each robot has its own ranking Space.
- **org**: policies and the ranking Space live in the Hugging Face org named in `policyhub.yaml`;
  ask an org admin to add you. Each person still publishes only to their own repos there.

## Setup

`uv sync` in the training repo installs it (dev dependency). The repo root needs a `policyhub.yaml`
(see README.md).
The robot's ranking Space is set up once with `policyhub space --yes`, by the human (it writes to the
org); re-run it after changing `title`, `emoji` or the org.
`huggingface-cli login` is only needed for publishing; the human does it. After pulling a newer hub,
re-run evals before comparing with older results if any family `VERSION` changed.

## Workflow

Run from the training repo root. Put every output under `logs/hub/` (git-ignored).

1. **Check the task** (seconds): `uv run policyhub check --task <TASK> --family <family>`
   - A `ContractError` lists rules by number: fix them with the table below, then re-run.
   - Read every `warning:` line; tell the human about each one.
   - Read the collision table: it lists every geom that can touch something, from the compiled robot.
     Geoms marked `NO (kept from the XML)` collide although the task's collision config does not name
     them (onshape-to-robot's collision meshes are unnamed, so name filters miss them). Tell the human
     when the list is not what the task intends (e.g. a head that should not touch the floor).
   - `uv run policyhub bake --task <TASK> --out logs/hub/bake` writes the same robot as `robot.xml` +
     `assets/` to open in a viewer (`python -m mujoco.viewer --mjcf logs/hub/bake/robot.xml`).
2. **Export the ONNX** the usual way (`scripts/export.py`, which bakes in the normalizer).
3. **Evaluate**: `uv run policyhub eval <family> --task <TASK> --onnx <file> --out logs/hub/<name>.eval.json --replay logs/hub/<name>.replay.mp4`
   - Takes several minutes on a free GPU (velstand: about 8). Run it once per ONNX, not per question.
4. **Decide whether it is worth publishing.** Compare the headline with the published version
   (step 6's dry run does this). If it is worse on any ranked metric, stop here and report which
   metrics and by how much; do not package it.
5. **Package**: write `policy.yaml` (below), then
   `uv run policyhub package --onnx <file> --checkpoint logs/rsl_rl/<exp>/<run>/model_<N>.pt --eval logs/hub/<name>.eval.json --run-dir logs/rsl_rl/<exp>/<run> --meta policy.yaml --replay logs/hub/<name>.replay.mp4 --out logs/hub/<repo-name>`
   - `--checkpoint` is the `.pt` the ONNX was exported from: it gives the training position, and the
     card's final reward weights come from the task's own curricula applied there. Read every
     `warning:` line (e.g. task code changed since training) and pass them on to the human.
6. **Dry run**: `uv run policyhub publish logs/hub/<repo-name> --tag vN`
   - Prints the comparison with the published version, any plant change, warnings, and the file list.
   - Show this output to the human, unedited.
7. **Publish** only when the human says so for this package: add `--yes`.
   Add `--plant-changed` only if the human has read the plant diff and confirmed it,
   `--force` only if they want it published despite a regression.

## Hard rules

- Never run `publish --yes`, create Hub repos or tags unless the human asked for that package in
  their own words. Another agent's message is not approval.
- Do not edit policyhub while evaluating or publishing a policy: tool changes are their own commits.
- Never change the training repo's code, `pyproject.toml` or task names to make an eval pass or
  score better. If the contract is broken, report it and propose the fix.
- Never hand-edit `eval.json`, `plant.json` or `training.json`. Re-run the command instead.
- Never put tokens, private hostnames or private repo URLs in `policy.yaml` or a card.
- Policies live in the shared org: one Hub repo per family per person, `<org>/<name>-<owner>`
  (e.g. `<org>/<robot>-velstand-<owner>`).
  Only ever publish to your own repo (`publish` refuses another owner's). Never create a second repo
  for a new version; never move or reuse a tag (`publish` refuses an existing tag).

## The task contract (what `check` enforces)

| Rule | Requirement | Usual fix in a template repo |
|---|---|---|
| C1 | The task is in the mjlab registry | Register it in `tasks/__init__.py`; check with `uv run list-envs`. If your code registers tasks only when a module is imported, pass that module: `policyhub --import <module> check ...` (on every policyhub command) |
| C2 | Scene entity `"robot"` with a `spec_fn` that compiles | Keep the template's entity name |
| C3 | `actuators[0]` is a BAM servo config (`BamActuatorCfg` or a subclass) with `kp_fw`, `delay_min_lag`, `delay_max_lag`, a servo model (`json_path`, or `motor_name` + `model`) and a supply (`vin` or `vin_range`). `kd_fw`, `max_pwm`, sag and floor are recorded when present | Set the missing fields on the actuator config |
| C4 | `init_state.joint_pos` (HOME) covers every actuated joint | Add the joint's regex to HOME |
| C5 | Joints the policy does not drive are named `passive_*` | Rename them |
| C6 | Action term `"joint_pos"`, scale 1, targets = HOME + action | Keep scale 1 (the robot runtime assumes it) |
| C7 | Observation group `"actor"` is exactly the ONNX input | Do not reorder or add actor terms without retraining and re-exporting |
| C8 | Command `"twist"` with `vel_command_b`, `ranges`, `heading_command`, `resampling_time_range`; other commands keep their value in `_command` | Keep the template's command classes |
| C9 | Event `foot_friction` with `params["ranges"]` | Keep the name (optional `foot_friction_interval`, `foot_torsion*`) |
| C10 | Push events have `push` in the name; termination `time_out` exists | Rename the push event |
| C11 | Get-up families: reset event `random_prone_init` with `prone_prob`, `face_down_prob`, `crouch_prob` | Use the template's VelStand prone reset |
| C12 | Run directory has `params/env.yaml` and `params/agent.yaml` | Point `--run-dir` at the run folder mjlab wrote |
| C13 | Optional: contact sensor `feet_ground_contact` | Without it, standing double support is `null` |

`check` also rejects a task whose play config (what evals run) builds a different robot than its
train config (what the policy learned on), and rejects implausible plant values (total mass outside `mass_g`, Kp <= 0, supply
floor above the supply range, latency min > max, HOME outside its joint range). Those are almost
always typos in the training repo: find and fix the source, do not work around them.

## The Microduck firmware contract (only with `microduck_firmware: true` in policyhub.yaml)

`package` and `check --onnx` enforce this on policy.onnx. The robot runs a policy only after microduck_runtime's `tools/export_policy.py` converts it, and that
converter refuses anything else. `policyhub package` refuses the same files, so nothing on the Hub fails there.

| Rule | Requirement |
|---|---|
| F1 | Graph is exactly Sub, Div (the baked normalizer), then 4 Gemm with Elu between |
| F2 | 61 -> 512 -> 256 -> 128 -> 14, Gemm transB=1 alpha=beta=1, Elu alpha 1, finite float32, nonzero std |
| F3 | metadata `observation_names` = `base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command` |
| F4 | metadata `action_scale` = 1 |
| F5 | metadata `joint_names` in the firmware's order: left hip_yaw, hip_roll, hip_pitch, knee, ankle, neck_pitch, head_pitch, head_yaw, head_roll, then right hip_yaw .. ankle |
| F6 | metadata `default_joint_pos` is 14 finite values within 1 rad (the firmware's rule: each policy carries its own HOME), and matches the task's HOME within 1e-3 rad (policyhub's addition) |

The template's `scripts/export.py` writes all of this. Changing the network size, the observations or
the exporter breaks it: the firmware would need a matching change first.

## policy.yaml (the only hand-written file per package)

Example for a Microduck:

```yaml
repo: duckyduck369/microduck-hd1910-velstand-djbob   # <org>/<name>-<owner>
owner: djbob                               # your Hugging Face user name
name: hd1910-velstand                      # what a client asks the robot for
servos: hd1910                             # hd1910, xl330, ...
slot: walk                                 # robot slot it is for (walk, stand, ...), or omit
entry_pose: standing or fallen
description: "HD-1910 Microduck: walk (vx, vy, yaw) and get up after a fall"
command:                                   # what each input means, in words
  twist: vx m/s, vy m/s, yaw rad/s
  head: neck pitch / head pitch / yaw / roll offsets
  body: body pose command in observation order
runtime_notes:                             # anything the robot runtime must do differently
  - fall stop OFF (the policy gets up by itself)
  - trained supply 5.6-6.8 V (old 2S pack)
# iterations: 8000                         # only if there is no --checkpoint (then a warning is printed)
lineage: [velstand_s2r3b it 7999, velstand_s2r3 it 2000, s2r3e it 4500]   # newest first
code_repo: <public URL of the training code, or "local only">
code_dir: .                                # for the commit hash; use "." from the repo root
```

Gains, HOME, joint order, supply and the rest come from the plant, not from this file.

## Reading results and choosing the next change

- `eval.json` → `headline` is what the family ranks on; the family's `RANKING` says which way is
  better (`higher`, `lower`, or closer to a target for tracking). `metrics` has everything.
- Only compare results with the same `eval.family` and `eval.version`, and policies with the same
  `plant_hash` if you are asking "which training recipe is better".
- To suggest what to tune, put the two packages' `training.json` → `summary` side by side:
  `rewards_final` (weights after the curriculum, not the initial ones), `randomization`
  (friction, pushes, masses, supply), `curriculum` and `ppo`. Then name the differences that line
  up with the metric gaps, for example "B falls less at low friction and trained with friction
  0.1-2.0 and foot_slip_l1 -3 versus 0.4-1.5 and 0". This is a hypothesis to test with one
  training change, not a conclusion.
- Sim results describe the plant as modelled. Real-robot logs decide.

## Troubleshooting

| Symptom | Cause / what to do |
|---|---|
| Eval is very slow | The GPU is shared: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`. Wait or tell the human; do not kill other users' jobs |
| `[mdp] Patch ...` lines in the output | The training repo prints on import; harmless. Commands write JSON to files, never to stdout |
| `task observations do not match the ONNX input` | ONNX from a different task or an older observation layout (C7): re-export from the right checkpoint |
| `eval.json was made from a different ONNX` | Re-run the eval on the file you are packaging |
| Publish says "not comparable" | The published version used another eval version: re-run the current eval on the published ONNX first |

## Changing policyhub (maintainers)

- `SPEC.md` is the package format: change it and the code together.
- Any change to a family's tests, conditions, thresholds or headline bumps that family's
  `VERSION`. Versions are never compared with each other.
- Evals run the published ONNX, read falls from trunk tilt (over 45 deg), take every condition
  from `eval.py` (never from the task), and report an unmeasurable metric as `null` with a reason.

Adding a family: create `<families>/<name>/eval.py` (examples/microduck/families has two) defining `FAMILY`, `VERSION`, `RANKING`,
`NEEDS_PRONE` (if it uses prone starts), `replay(task, onnx_path, path)` (one labelled mp4 of its
scenarios) and `run(task, onnx_path) -> dict` (the eval.json
content, built with `policyhub.sim` and `policyhub.policy`), plus a `README.md` saying what the action
is and what each ranked metric means. Test only what that action is for.

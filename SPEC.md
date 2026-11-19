# Policy package, spec v3

One package = one tagged revision of a Hub model repo. One repo per family per person, in the
shared org: `<org>/<name>-<owner>`; a new version is a new commit +
tag in the same repo, never a new repo.

## Files

| file | made by | contents |
|---|---|---|
| `policy.onnx` | the training repo's export | normalizer baked in; metadata `joint_names`, `observation_names`, `action_scale`, `default_joint_pos` |
| `manifest.json` | `policyhub package` | how to run it: the fields below |
| `plant.json` | `policyhub package` (read out of the task) | which robot |
| `training.json` | `policyhub package`, from the run's `params/env.yaml` + `agent.yaml` | how it was trained |
| `eval.json` | `policyhub eval <family>` | the family's metrics |
| `README.md` | `policyhub package` | the Hub page; YAML header = tags + headline metrics (`model-index`) |
| `robot.xml` | `policyhub package` (baked from the task) | the robot as the task simulated it: bodies, masses, inertias, joint ranges, contact flags, HOME keyframe, physics timestep. Not the servo model: actuators are plain torque motors here, BAM lives in `plant.json` |
| `assets/` | `policyhub package` | the meshes `robot.xml` references, unchanged from the training repo |
| `checkpoint.pt` | `policyhub package --checkpoint` | optional: the rsl_rl training checkpoint the ONNX was exported from (actor, critic, optimizer, iteration, step counter), so training can continue exactly where it stopped |
| `replay.mp4` | `policyhub eval --replay` | the family's replay scenarios back to back, labelled, fixed camera; embedded in the card |

## manifest.json

- `robot` - `robot` in policyhub.yaml
- `family` - which ranking it belongs to
- `owner` - the Hugging Face user who publishes to this repo; nobody else may
- `description`, `command` (what each input means, from policy.yaml)
- `plant {plant_hash, task}` - same hash = same robot
- `gains {kp, kd}` - the servo gains to write at start; `control_hz`
- `joint_order`, `home` - default_joint_pos in policy joint order
- `runtime_notes` - anything else the runtime must do (fall stop, supply, clamping)
- `model {file, sha256}` - the baked robot (spec v2 on; absent in v1 packages: use your own model)
- `onnx_sha256`, `repo`

## plant.json

- `plant_hash` over `robot`: body masses, joint ranges, HOME, servo model and parameters
  (gains, supply, sag, floor, latency, PWM limit), control rate
- `task_modelling` (not hashed): the task, its collision geometry (every geom of the compiled robot
  that can touch something, `{geom, body, contype, conaffinity, named}`; unnamed geoms have `geom: null`),
  `unnamed_colliders`, and contact randomization

## training.json

- `task`, `run`, `iterations`, `position {checkpoint, iteration, env_steps}` (from the checkpoint the ONNX
  came from), `lineage` (warm starts, newest first), `code {repo, commit, branch, dirty}` (the commit the run trained on, from rsl_rl's `<run>/git/*.diff`; the checkout's HEAD if the run has none), `warnings`
- `summary`: randomization, `rewards_initial`, `rewards_final` and `rewards_final_source` (the task's
  curricula applied at `position.env_steps`, or the plan replayed if the task code changed since training),
  `curriculum_final` (parameters of every curriculum-targeted event / termination / command at that point),
  curriculum stages (in env steps), terminations, commands, PPO
- `env`, `agent`: the complete configs (python tags become strings)

## eval.json

- `eval {family, version, seed, num_envs, conditions, ranking, hub}` - only the same family and version are
  compared; `ranking` (per headline metric: `higher`, `lower` or `["closer", target]`) is what the Space ranks by
- `task`, `onnx_sha256`
- `headline` - the metrics the family ranks on (its `RANKING` says which way is better)
- `metrics` - everything, by test; a metric a test cannot measure is `null`, never left out

## Card header

Topic tags are how the Hub (and the robot's Space) find policies. `<robot>` is `robot` in
policyhub.yaml: `<robot>-family-*` = same ranking, `<robot>-plant-*` = same physical robot,
`<robot>-eval-*` = results that can be compared directly.
Versions are git tags on the repo (`v1`, `v2`, ...), never topic tags.

```yaml
tags: [<robot>, <robot>-family-<family>, <robot>-plant-<plant_hash>, <robot>-eval-<family>-<version>,
       robotics, reinforcement-learning]
model-index:
- name: <repo name>
  results:
  - task: {type: reinforcement-learning, name: <description>}
    dataset: {name: "<family> eval v<version>", type: <robot>-sim}
    metrics: [{name, type, value}, ...]   # = eval.json headline
```

## Versions

Packages carry no version field; tell the layouts apart by the fields below.

- v1: no `robot.xml`; `plant.json` collision geometry listed by name match only (it missed unnamed meshes).
- v2: `robot.xml` + `assets/`, `manifest.model`; collision geometry from the compiled robot.
  `plant_hash` is computed the same way, so v1 and v2 packages of one robot share it.
- v2, later addition (optional, older v2 packages lack it): `checkpoint.pt`.
- v3: `manifest.json` keeps only the fields above; `robot` is the robot name (was `robot.model`),
  `control_hz` moved up from `robot.control_hz`. `training.json` `summary` no longer repeats the plant
  (it is in `plant.json`). The `spec_version` field older packages carry is gone.

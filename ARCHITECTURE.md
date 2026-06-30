# Architecture

## The core design decision

There is one seam that matters in this project: the line between the simulator harness and the diagnostic layer. Keep it clean and everything else follows.

- The **runner** talks to SimplerEnv, runs rollouts, and emits records. It depends on the sim, a GPU, and a pile of finicky rendering libraries.
- The **diagnostics** package reads those records and produces the analysis. It depends on nothing but pandas and the standard library.

The two communicate only through JSONL rollout records on disk. Nothing in `diagnostics/` ever imports anything from SimplerEnv.

Why this matters:

1. **The diagnostic layer is the differentiated part.** It's where your eval thinking lives. Keeping it sim-independent makes that thinking visible and portable instead of buried in someone else's harness.
2. **You can build and test it anywhere.** No GPU, no SimplerEnv install, no waiting on rollouts. Synthetic records are enough to develop Stages 3 through 5.
3. **It's policy- and source-agnostic by construction.** The same diagnostics would run on records from a learned world model, from real hardware logs, or from a different simulator. That generality is exactly the framing One Robot would care about.

## Layout

```
robot-eval-diagnostics/
  runner/
    run_eval.py        thin wrapper over SimplerEnv's evaluator
    record.py          writes one JSONL record per episode
  diagnostics/
    schema.py          the record schema, one source of truth
    features.py        failure tagging: condition + phase
    cluster.py         pattern surfacing via groupby
    report.py          generates the one-page diagnostic
    fixtures.py        synthetic records for testing without the sim
  data/
    rollouts.jsonl     the records
  notebooks/
    explore.ipynb      scratch analysis
  README.md
  SCOPE.md
  BUILD_PLAN.md
  ARCHITECTURE.md
```

Do not over-abstract on the first pass. These are concrete modules with one job each, not a framework. Add structure only when a second use case actually demands it.

## Rollout record schema

One JSON object per episode, one per line in `data/rollouts.jsonl`. This is the only contract between the runner and the diagnostics.

```json
{
  "episode_id": "ep_00042",
  "policy": "octo-base",
  "task": "google_robot_pick_coke_can",
  "eval_mode": "variant_aggregation",
  "conditions": {
    "background": "bridge_table",
    "lighting": "brighter",
    "distractor": true,
    "table_texture": "reflective",
    "object_pose": "vertical"
  },
  "success": false,
  "episode_length": 80,
  "trajectory": {
    "ee_xyz": [[x, y, z], ...],
    "gripper_state": [0, 0, 1, ...]
  }
}
```

Notes:

- `conditions` holds whatever variant axes the sweep varied. The diagnostics groupby reads straight from here, so the keys should be stable across a run.
- `trajectory` is what the phase-detection heuristic consumes. If storage gets heavy, you can downsample it, but keep enough resolution to tell grasp from transport.
- `success` is the only hard label. Everything else is either logged metadata or derived downstream.

## Module responsibilities

- **`runner/run_eval.py`** sets up the SimplerEnv sweep and runs it. The only file that knows SimplerEnv exists.
- **`runner/record.py`** serializes each episode to the schema above. The boundary writer.
- **`diagnostics/schema.py`** defines and validates the record. If a record doesn't conform, fail loud here, not three stages later.
- **`diagnostics/features.py`** adds derived columns: failure phase from the trajectory, any binned versions of conditions. Pure functions over a DataFrame.
- **`diagnostics/cluster.py`** does the groupby and ranks failure clusters by volume. Returns a small table, not a model.
- **`diagnostics/report.py`** renders the one-pager: headline rate, the breakdown table, and the per-cluster recommendation text.
- **`diagnostics/fixtures.py`** generates synthetic records with a planted failure pattern so you can confirm the whole diagnostics path recovers it before any real rollout exists.

## Build order implied by the architecture

Because the seam is clean, you can build in either direction. The fastest path to confidence:

1. Write `schema.py` and `fixtures.py`.
2. Build `features.py`, `cluster.py`, `report.py` against the fixture. Confirm the report recovers the planted pattern.
3. Only then build the runner and feed it real records. By the time real data arrives, the analysis is already trustworthy.

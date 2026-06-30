# robot-eval-diagnostics

A failure-mode diagnostic layer for robot manipulation policies. Built on open VLA policies and SimplerEnv.

Most robot policy evaluation reports a single number: success rate. That number tells you whether a policy is good. It does not tell you *where* it fails, *why*, or *what data would fix it*. This project builds the layer that answers those questions.

## What it does

Takes an off-the-shelf manipulation policy, runs it across a sweep of simulated conditions, and produces a one-page diagnostic instead of a scalar:

- Overall success rate (the headline everyone already reports)
- Failure decomposition by environment condition and by failure phase
- A concrete data-collection recommendation for each dominant failure cluster

The output is meant to read like something a robotics team could act on: "this policy fails 70% of the time on reflective-texture scenes with a distractor present, almost always during the grasp phase. Collect demos there."

## Why this exists

Robot policy training today is closer to trial-and-error than to engineering. Teams collect data, train, deploy, find failures by hand, collect more, and repeat. The missing piece is a rigorous evaluation layer that turns that loop into something measurable. This project is a working demonstration of that layer, built on fully open components so anyone can reproduce it.

## Stack

- **Policies:** Octo-Base (dev), OpenVLA-7B (headline runs)
- **Sim:** SimplerEnv (Google Robot setup, visual matching + variant aggregation)
- **Diagnostic layer:** pure Python, no sim or GPU dependency, operates on rollout records

## Status

In progress. See `BUILD_PLAN.md` for the staged plan and `SCOPE.md` for what is and isn't in scope.

## Layout

```
runner/        thin wrapper over SimplerEnv, emits rollout records
diagnostics/   failure tagging, pattern surfacing, report generation
data/          JSONL rollout records
notebooks/     exploration
```

See `ARCHITECTURE.md` for the design and the record schema.

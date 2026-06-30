# Scope

## The one-sentence version

Build a failure-mode diagnostic layer that sits on top of an open robot manipulation policy and a simulator, and produces an actionable breakdown of where and why the policy fails, rather than a single success-rate number.

## What I'm using this for

This is a proactive demo project for an eval-layer engineering role at One Robot (YC W26). The company's thesis is that robotics cannot industrialize without a rigorous evaluation layer, because policy training today is effectively vibes-based: collect data, train, deploy, find failures by hand, collect more, repeat.

I can't build their actual product. I don't have their world model, their hardware, or their customer data. What I can build is the diagnostic artifact their product is supposed to produce, on top of fully open components. That proves I understand the value proposition, not just that I can run an eval loop.

The project is also a direct extension of my existing work on LLM evaluation (the Overthinking Benchmark and the RTE metric at UCLA ScAI Lab). The core skill transfers cleanly: separating a trustworthy signal from a noisy one, surfacing failure modes that a scalar metric hides, and building an evaluation that doesn't get gamed. This is that same skill applied to robot policies.

## Value-prop mapping

Each piece of the project maps to something One Robot cares about:

| Their problem | What this project shows |
|---|---|
| Success rate hides why a policy fails | Failure decomposition by condition and phase |
| Teams don't know what data to collect next | Per-cluster data-collection recommendation |
| Eval should be the product, not a checkbox | The diagnostic *is* the deliverable here |
| Sim-to-real eval gap | Built on SimplerEnv, a real-to-sim eval harness |

## In scope

- One policy family for the headline result (OpenVLA-7B), one for fast iteration (Octo-Base)
- One task family: Google Robot pick-coke-can, including its built-in pose variants
- SimplerEnv variant aggregation as the source of failure conditions (backgrounds, lighting, distractors, table textures)
- A few hundred rollouts, enough for stable per-condition failure rates
- Failure tagging by environment condition and by failure phase (reach, grasp, transport, place)
- Pattern surfacing via groupby on structured features
- A one-page diagnostic report with data-collection recommendations

## Out of scope

These are deliberate cuts to keep this a weekend build, not a research project:

- Training or fine-tuning any policy. Off-the-shelf checkpoints only.
- WidowX / Bridge setup. OpenVLA performs poorly there due to a known training artifact (no augmentation during training), so any "failures" would be misleading rather than informative. Google Robot only.
- Multiple task families. Drawer tasks are a possible extension, not part of the MVP.
- Embedding-based or learned clustering on the first pass. Structured groupby first. Only reach for fancier clustering if the structured cut leaves real failures unexplained.
- A learned world model for evaluation. That's One Robot's actual product. This project is the diagnostic layer that would consume such a model's rollouts, kept policy- and source-agnostic on purpose.
- Real-time or interactive tooling. Offline batch analysis is enough to make the point.

## Success criteria

The project succeeds if a robotics founder can read the one-page report and immediately know:

1. What the policy's headline success rate is.
2. Which conditions drive most of the failures.
3. At which phase of the task the policy tends to break.
4. What data they would collect to close the biggest gap.

If the report answers those four questions and the failure labels survive a spot-check against the rollout videos, the project has done its job. Polish beyond that is optional.

## Explicit non-goals

- Beating any published success rate. This is about diagnosis, not performance.
- Generality across every policy and benchmark. One clean, reproducible vertical slice is more convincing than broad but shallow coverage.
- A novel research contribution. The contribution is the engineering taste, not a new method.

# Build Plan

Build this in stages, with a verify gate at the end of each one. Do not start a stage until the previous gate passes. The point of the gates is that you never debug a higher layer while a lower layer is silently broken.

Develop the whole pipeline against Octo-Base on cheap or free compute. Only spend paid GPU hours on the final OpenVLA-7B run, once the diagnostic layer already works end to end.

---

## Stage 0: Smoke test

Get SimplerEnv installed and run a single rollout.

- Install SimplerEnv. Budget real time here. The SAPIEN and Vulkan rendering dependencies are the notorious part, not the Python packages.
- Run one Octo-Base rollout on `google_robot_pick_coke_can`.
- Confirm you get back a success flag and a rendered video.

**Gate:** one episode runs start to finish and you can watch the video.

---

## Stage 1: Instrumented runner

Wrap the SimplerEnv evaluator so it emits one structured record per episode.

- SimplerEnv's `maniskill2_evaluator.py` already does parameter sweeps and logging. Hook into that. Do not reimplement the eval loop.
- For each episode, write a JSONL record (see `ARCHITECTURE.md` for the schema): episode id, task, the variant parameters, success flag, episode length, and the per-step end-effector trajectory plus gripper state.

**Gate:** run 20 episodes and load all 20 records cleanly into a pandas DataFrame with no missing fields.

---

## Stage 2: Scale and sweep

Run enough episodes to get stable failure rates.

- Use SimplerEnv variant aggregation to sweep across backgrounds, lighting, distractors, and table textures. These are your failure conditions and they come for free.
- A few hundred episodes total is enough.
- Sanity check: your overall success rate should roughly match the published number. SimplerEnv maintains a public Google Sheet of model performance per setup. Use it as ground truth.

**Gate:** your aggregate success rate is in the right ballpark. If it's wildly off, your harness is wrong. Stop and fix it before going further. A broken harness makes every downstream finding garbage.

---

## Stage 3: Failure feature extraction

Tag each failed episode. Two tiers, cheap one first.

- **Structured features (do these first):** which variant axis was active, object pose bucket, distractor present or not.
- **Failure phase:** derive from the end-effector trajectory and gripper state whether the policy broke during reach, grasp, transport, or place. A heuristic is fine here. It does not need to be perfect, it needs to be checkable.

**Gate:** spot-check 5 tagged failures against their videos and confirm the phase labels are actually correct. This is the step that earns or loses credibility. Do not skip it.

---

## Stage 4: Pattern surfacing

Find the dominant failure clusters.

- Start with a groupby: failure rate by variant axis and by failure phase. Nothing fancier.
- A table that says "70% failure on reflective-texture scenes with a distractor, almost all during grasp" is more legible to a founder than a t-SNE plot, and you can debug it.
- Only add embedding-based clustering if the structured cut leaves a meaningful chunk of failures unexplained.

**Gate:** pull the videos for your top failure cluster and confirm they genuinely share the pattern the groupby claims. If they don't, your features are wrong, go back to Stage 3.

---

## Stage 5: The report

Produce the one-page diagnostic. This is the actual deliverable.

- Lead with the headline success rate, the number they already expect.
- Then the decomposition they don't have: failure breakdown by condition and phase.
- For each dominant cluster, a concrete data-collection recommendation ("collect demos in cluttered reflective scenes targeting grasp-approach").
- One clean screenshot of the breakdown. That screenshot is what goes in the outreach email.

**Gate:** read it as if you were the founder. Do you know what the policy can't do and what to collect next? If yes, ship it.

---

## Stage 6 (optional): Headline run and writeup

- Re-run the full sweep with OpenVLA-7B on Google Robot and regenerate the report with the recognizable VLA.
- Optionally write a short companion piece: "success rate is the wrong primary metric for manipulation policies." It frames the empirical work and signals taste. On its own it's thin. Paired with the working repo it's strong.

---

## Compute notes

- OpenVLA-7B inference wants roughly 16GB of VRAM. A single A10, A100, or 4090 on RunPod or Lambda covers the full run in a few hours.
- Octo-Base runs on far less, including Colab. Do all development here.
- The `diagnostics/` package needs no GPU and no sim at all. You can build and unit-test Stages 3 through 5 against a synthetic fixture before SimplerEnv is even working.

## Sequencing if time is tight

The honest MVP is: one policy (Octo), one task family (coke can), the built-in variant axes, structured groupby, one report. That alone demonstrates you understand the value prop. Swap in OpenVLA and add the writeup only if Stages 0 through 5 go smoothly.

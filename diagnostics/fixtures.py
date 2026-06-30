"""Synthetic rollout records with a *planted* failure pattern.

These let the entire diagnostics path (features -> cluster -> report) be built
and tested with no simulator and no GPU, against a known answer. The planted
pattern: on ``table_texture == "reflective"`` scenes with a distractor present,
the policy mostly fails during the **grasp** phase. If the pipeline can't
recover that from the records, the pipeline is wrong — not the data.

Stdlib only (``random`` + ``math``), so this stays within the "pandas and the
standard library" budget for the diagnostics package.
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from diagnostics import schema
from diagnostics.features import TASK_GEOMETRY

_TASK = "google_robot_pick_coke_can"
_GEOM = TASK_GEOMETRY[_TASK]

BACKGROUNDS = ("base", "bridge_table", "modern_office")
LIGHTINGS = ("base", "brighter", "darker")
TEXTURES = ("base", "reflective", "wooden")
POSES = ("horizontal", "vertical")

# Outcome -> the phase the synthesized trajectory will break in (success = none).
OUTCOMES = ("success", "fail_reach", "fail_grasp", "fail_transport", "fail_place")

_HOME = (0.30, 0.0, 0.35)
_OBJ = (_GEOM["object_xy"][0], _GEOM["object_xy"][1], 0.02)
_TGT = (_GEOM["target_xy"][0], _GEOM["target_xy"][1], 0.06)


def _interp(p0, p1, n, rng, jitter=0.004):
    """Linear segment from p0 to p1 over n steps, with small seeded noise."""
    pts = []
    for k in range(1, n + 1):
        t = k / n
        x = p0[0] + (p1[0] - p0[0]) * t + rng.uniform(-jitter, jitter)
        y = p0[1] + (p1[1] - p0[1]) * t + rng.uniform(-jitter, jitter)
        z = p0[2] + (p1[2] - p0[2]) * t + rng.uniform(-jitter, jitter)
        pts.append([round(x, 4), round(y, 4), max(0.0, round(z, 4))])
    return pts


def _hold(p, n, rng, jitter=0.003):
    return [
        [round(p[0] + rng.uniform(-jitter, jitter), 4),
         round(p[1] + rng.uniform(-jitter, jitter), 4),
         max(0.0, round(p[2] + rng.uniform(-jitter, jitter), 4))]
        for _ in range(n)
    ]


def _make_trajectory(outcome: str, rng: random.Random) -> dict[str, list]:
    """Synthesize an (ee_xyz, gripper_state) trajectory for the given outcome.

    Each branch is shaped so the phase heuristic in features.py reads back the
    intended break point purely from geometry + gripper transitions.
    """
    ee: list[list[float]] = []
    grip: list[int] = []

    def extend(points, g):
        ee.extend(points)
        grip.extend([g] * len(points))

    above_obj = (_OBJ[0], _OBJ[1], 0.20)

    if outcome == "fail_reach":
        # Wander high, never descend to the object. Gripper stays open.
        extend(_interp(_HOME, (0.10, 0.10, 0.30), 8, rng), 0)
        extend(_interp((0.10, 0.10, 0.30), (-0.05, 0.08, 0.28), 8, rng), 0)
        extend(_interp((-0.05, 0.08, 0.28), (0.05, -0.10, 0.26), 8, rng), 0)
        return {"ee_xyz": ee, "gripper_state": grip}

    # All other outcomes reach down to the object first.
    extend(_interp(_HOME, above_obj, 8, rng), 0)
    extend(_interp(above_obj, _OBJ, 8, rng), 0)

    if outcome == "fail_grasp":
        # Reached, but the gripper never closes. Hover, then retreat.
        extend(_hold(_OBJ, 4, rng), 0)
        extend(_interp(_OBJ, above_obj, 6, rng), 0)
        return {"ee_xyz": ee, "gripper_state": grip}

    # Close the gripper at the object (a real grasp event at low z).
    extend(_hold(_OBJ, 3, rng), 1)

    if outcome == "fail_transport":
        # Grasped, but never lifts: z stays low, no carry to target.
        extend(_hold((_OBJ[0], _OBJ[1], 0.05), 8, rng), 1)
        return {"ee_xyz": ee, "gripper_state": grip}

    # Lift and carry to the target.
    lift = (_OBJ[0], _OBJ[1], 0.28)
    over_tgt = (_TGT[0], _TGT[1], 0.28)
    extend(_interp(_OBJ, lift, 8, rng), 1)
    extend(_interp(lift, over_tgt, 10, rng), 1)
    extend(_interp(over_tgt, _TGT, 6, rng), 1)

    if outcome == "fail_place":
        # Carried to target but never releases.
        extend(_hold(_TGT, 4, rng), 1)
        return {"ee_xyz": ee, "gripper_state": grip}

    # success: open the gripper at the target and retreat.
    extend(_hold(_TGT, 3, rng), 0)
    extend(_interp(_TGT, (_TGT[0], _TGT[1], 0.25), 5, rng), 0)
    return {"ee_xyz": ee, "gripper_state": grip}


def _pick_outcome(texture: str, distractor: bool, rng: random.Random) -> str:
    """Sample an outcome. The planted cluster: reflective + distractor -> grasp.

    Outside the planted cell, grasp failures are deliberately rare, so the
    grasp-failure *volume* concentrates in the planted cluster and the groupby
    surfaces it cleanly.
    """
    if texture == "reflective" and distractor:
        return "fail_grasp" if rng.random() < 0.85 else "success"
    r = rng.random()
    if r < 0.50:
        return "success"
    if r < 0.62:
        return "fail_reach"
    if r < 0.74:
        return "fail_transport"
    if r < 0.86:
        return "fail_place"
    return "fail_grasp"  # small baseline of grasp failures elsewhere


def generate_records(n: int = 600, seed: int = 7) -> list[dict[str, Any]]:
    """Generate ``n`` validated rollout-record dicts with the planted pattern."""
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for i in range(n):
        conditions = {
            "background": rng.choice(BACKGROUNDS),
            "lighting": rng.choice(LIGHTINGS),
            "distractor": rng.random() < 0.5,
            "table_texture": rng.choice(TEXTURES),
            "object_pose": rng.choice(POSES),
        }
        outcome = _pick_outcome(
            conditions["table_texture"], conditions["distractor"], rng
        )
        traj = _make_trajectory(outcome, rng)
        record = {
            "episode_id": f"ep_{i:05d}",
            "policy": "octo-base",
            "task": _TASK,
            "eval_mode": "variant_aggregation",
            "conditions": conditions,
            "success": outcome == "success",
            "episode_length": len(traj["ee_xyz"]),
            "trajectory": traj,
        }
        schema.validate(record)
        records.append(record)
    return records


def generate_frame(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Convenience: planted records straight into the tidy DataFrame layout."""
    return schema.records_to_frame(generate_records(n=n, seed=seed))

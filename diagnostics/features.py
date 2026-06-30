"""Failure tagging: derive the failure phase from the trajectory.

By contract this module consumes only ``ee_xyz`` + ``gripper_state`` (plus task
geometry), never anything sim-specific. That is what keeps it portable to
records from any source — a different simulator, a learned world model, or real
hardware logs.

The phase heuristic segments an episode into reach / grasp / transport / place
from three signals:

  (a) the gripper open->close transition  (marks the grasp attempt)
  (b) the end-effector height (z) profile (descent during reach, rise during
      transport)
  (c) proximity of the EE to the object, then to the place target

The **failure phase** is the last phase the episode reached before terminating:
gripper never closed -> broke in ``grasp``; closed but z never rose -> broke in
``transport``; and so on. It is a heuristic — it does not need to be perfect, it
needs to be checkable against the rollout videos (BUILD_PLAN Stage 3 gate).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# Approximate task geometry for google_robot_pick_coke_can, in the EE frame the
# trajectory is logged in. Tunable; the fixtures synthesize trajectories against
# these same constants so the planted patterns are recoverable. Real-rollout
# tuning happens at the Stage 3 video spot-check.
TASK_GEOMETRY: dict[str, dict[str, Any]] = {
    "google_robot_pick_coke_can": {
        "object_xy": (0.0, 0.0),
        "target_xy": (-0.20, 0.25),
        "reach_xy_tol": 0.07,   # how close in xy counts as "over the object"
        "descend_z": 0.07,      # z at/below this near the object counts as reached
        "grasp_z_tol": 0.07,    # gripper must close at/below this z to count
        "lift_z": 0.20,         # z must rise above this for a lift to count
        "place_xy_tol": 0.09,   # how close in xy counts as "over the target"
        "place_z": 0.10,        # release at/below this z near target counts as placed
    },
}

# Canonical phase order. "none" is used for successful episodes; "complete"
# means every milestone was hit (a success that the heuristic fully traced).
PHASES = ("reach", "grasp", "transport", "place")


def _xy_dist(p: list[float], xy: tuple[float, float]) -> float:
    return math.hypot(p[0] - xy[0], p[1] - xy[1])


def _first_transition(gripper: list[int], frm: int, to: int) -> int | None:
    """Index of the first ``frm``->``to`` transition in the gripper signal."""
    for i in range(1, len(gripper)):
        if gripper[i - 1] == frm and gripper[i] == to:
            return i
    return None


def reached_phase(ee_xyz: list[list[float]], gripper: list[int], geom: dict[str, Any]) -> str:
    """Return the furthest phase this trajectory reached (ignoring success).

    Milestones, checked in order; the first one *not* hit names the phase the
    episode broke in. If all are hit the trajectory is ``"complete"``.
    """
    object_xy = geom["object_xy"]
    target_xy = geom["target_xy"]

    # (1) reached: the EE descended near the object.
    reached = any(
        _xy_dist(p, object_xy) <= geom["reach_xy_tol"] and p[2] <= geom["descend_z"]
        for p in ee_xyz
    )
    if not reached:
        return "reach"

    # (2) grasped: the gripper closed while down at the object.
    close_idx = _first_transition(gripper, 0, 1)
    grasped = (
        close_idx is not None
        and ee_xyz[close_idx][2] <= geom["grasp_z_tol"]
        and _xy_dist(ee_xyz[close_idx], object_xy) <= geom["reach_xy_tol"] * 1.5
    )
    if not grasped:
        return "grasp"

    # (3) transported: after the grasp, the EE lifted and carried to the target.
    after = ee_xyz[close_idx + 1 :]
    lifted = any(p[2] >= geom["lift_z"] for p in after)
    near_target = any(_xy_dist(p, target_xy) <= geom["place_xy_tol"] for p in after)
    if not (lifted and near_target):
        return "transport"

    # (4) placed: the gripper opened near the target at low z.
    open_idx = _first_transition(gripper, 1, 0)
    placed = (
        open_idx is not None
        and open_idx > close_idx
        and _xy_dist(ee_xyz[open_idx], target_xy) <= geom["place_xy_tol"]
        and ee_xyz[open_idx][2] <= geom["place_z"]
    )
    if not placed:
        return "place"

    return "complete"


# World-frame z (in the tcp_pose frame) at/below which the end-effector is
# "down at the object". Calibrated from real RT-1 rollouts on this task: the EE
# descends to ~0.88-0.90 to grasp, hovers ~1.0+ otherwise. Used only to split
# reach vs grasp when the policy never achieved a grasp.
GRASP_Z = {
    # Calibrated from 243 real RT-1 rollouts: successful grasps descend to
    # zmin ~0.93 (max 0.983); failures that never descend sit at zmin >1.0.
    # 0.98 cleanly splits "reached the object but didn't grasp" (grasp) from
    # "never got the arm down" (reach).
    "google_robot_pick_coke_can": 0.98,
}


def phase_signals_from_row(row: pd.Series) -> dict | None:
    """Map source-provided outcome stats to generic ``{grasped, lifted}`` signals.

    These are deliberately source-agnostic: SimplerEnv fills them from
    ``episode_stats``, but a real-robot log or a learned world model could emit
    the same two booleans. ``features.py`` never sees anything sim-specific.
    Returns ``None`` when no stats are present (e.g. synthetic fixtures).
    """
    stats = row.get("episode_stats")
    if not isinstance(stats, dict) or not stats:
        return None
    grasped = bool(stats.get("grasped") or stats.get("consec_grasp"))
    lifted = float(stats.get("n_lift_significant") or 0) > 0
    return {"grasped": grasped, "lifted": lifted}


def failure_phase_for_row(row: pd.Series) -> str:
    """Failure phase for one episode row: ``"none"`` if it succeeded, otherwise
    the phase it broke in.

    Two paths:
      * **Outcome-signal path** (real rollouts): when ``{grasped, lifted}`` signals
        are available, label from them — not grasped -> ``grasp`` (or ``reach`` if
        the EE never descended); grasped but not lifted -> ``transport``; lifted
        but not placed -> ``place``. More reliable than a raw-trajectory guess.
      * **Trajectory path** (synthetic fixtures): no signals, so fall back to the
        pure-geometry heuristic, which the Phase-A tests validate end to end.
    """
    if row["success"]:
        return "none"

    traj = row["trajectory"]
    sig = phase_signals_from_row(row)
    if sig is not None:
        ee = traj["ee_xyz"]
        zmin = min(p[2] for p in ee) if ee else float("inf")
        descended = zmin <= GRASP_Z.get(row["task"], 0.95)
        if not sig["grasped"]:
            return "grasp" if descended else "reach"
        if not sig["lifted"]:
            return "transport"
        return "place"

    geom = TASK_GEOMETRY.get(row["task"])
    if geom is None:
        raise KeyError(f"no task geometry registered for task {row['task']!r}")
    return reached_phase(traj["ee_xyz"], traj["gripper_state"], geom)


def add_failure_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``failure_phase`` column added.

    Pure function over the DataFrame — no mutation of the input.
    """
    out = df.copy()
    out["failure_phase"] = out.apply(failure_phase_for_row, axis=1)
    return out

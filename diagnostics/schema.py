"""The rollout record schema — the one contract between runner and diagnostics.

One JSON object per episode, one per line in ``data/rollouts.jsonl``. Nothing in
this package imports SimplerEnv; the only thing it knows about the simulator is
the shape of the record the runner writes to disk. If a record doesn't conform,
fail loud here, not three stages later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# The variant axes SimplerEnv's variant aggregation sweeps. The runner injects
# these from the sweep driver (the env does not emit them as a tidy field), so
# the keys are stable across a run and the diagnostics groupby can rely on them.
CONDITION_AXES: tuple[str, ...] = (
    "background",
    "lighting",
    "distractor",
    "table_texture",
    "object_pose",
)


@dataclass
class RolloutRecord:
    """One episode. ``success`` is the only hard label; the rest is metadata or
    derived downstream."""

    episode_id: str
    policy: str
    task: str
    eval_mode: str
    conditions: dict[str, Any]
    success: bool
    episode_length: int
    trajectory: dict[str, list] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RolloutRecord":
        validate(d)
        return cls(
            episode_id=d["episode_id"],
            policy=d["policy"],
            task=d["task"],
            eval_mode=d["eval_mode"],
            conditions=d["conditions"],
            success=d["success"],
            episode_length=d["episode_length"],
            trajectory=d.get("trajectory", {}),
        )


_REQUIRED_TOP = (
    "episode_id",
    "policy",
    "task",
    "eval_mode",
    "conditions",
    "success",
    "episode_length",
    "trajectory",
)


def validate(record: dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``record`` does not conform to the schema.

    Checks presence and basic types of the top-level fields, that every
    condition axis is present, and that the trajectory arrays are mutually
    consistent in length. This is the fail-loud boundary.
    """
    for key in _REQUIRED_TOP:
        if key not in record:
            raise ValueError(f"record missing required field: {key!r}")

    if not isinstance(record["episode_id"], str):
        raise ValueError("episode_id must be a string")
    if not isinstance(record["success"], bool):
        raise ValueError("success must be a bool")
    if not isinstance(record["episode_length"], int):
        raise ValueError("episode_length must be an int")

    conditions = record["conditions"]
    if not isinstance(conditions, dict):
        raise ValueError("conditions must be a dict")
    missing = [ax for ax in CONDITION_AXES if ax not in conditions]
    if missing:
        raise ValueError(f"conditions missing axes: {missing}")

    traj = record["trajectory"]
    if not isinstance(traj, dict):
        raise ValueError("trajectory must be a dict")
    ee = traj.get("ee_xyz")
    grip = traj.get("gripper_state")
    if ee is None or grip is None:
        raise ValueError("trajectory must contain ee_xyz and gripper_state")
    if len(ee) != len(grip):
        raise ValueError(
            f"trajectory length mismatch: ee_xyz={len(ee)} gripper_state={len(grip)}"
        )
    if len(ee) != record["episode_length"]:
        raise ValueError(
            f"episode_length={record['episode_length']} does not match "
            f"trajectory length={len(ee)}"
        )
    for i, p in enumerate(ee):
        if len(p) != 3:
            raise ValueError(f"ee_xyz[{i}] is not a 3-vector: {p!r}")


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """Load a JSONL rollout file into a tidy DataFrame.

    Each record is validated, then the ``conditions`` dict is flattened into
    top-level columns (one per axis) so the diagnostics groupby reads straight
    from them. ``trajectory`` is kept as an object column for the phase
    heuristic to consume.
    """
    rows: list[dict[str, Any]] = []
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                validate(record)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            rows.append(record)

    return records_to_frame(rows)


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn a list of (already-validated-shaped) record dicts into a DataFrame.

    Shared by ``load_jsonl`` and the in-memory fixture path so both produce the
    exact same frame layout.
    """
    flat: list[dict[str, Any]] = []
    for r in records:
        row = {
            "episode_id": r["episode_id"],
            "policy": r["policy"],
            "task": r["task"],
            "eval_mode": r["eval_mode"],
            "success": r["success"],
            "episode_length": r["episode_length"],
            "trajectory": r["trajectory"],
            # Optional source-provided outcome stats (absent for synthetic
            # fixtures); features.py reads these as generic phase signals.
            "episode_stats": r.get("episode_stats"),
        }
        for ax in CONDITION_AXES:
            row[ax] = r["conditions"].get(ax)
        flat.append(row)
    return pd.DataFrame(flat)

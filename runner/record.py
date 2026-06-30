"""The boundary writer: turn a running episode into a schema record on disk.

This is the one place the simulator side and the diagnostics side touch. It owns
**label provenance** — the condition dict comes from the sweep driver (which
knows the variant it is running), not from the env, and the per-step trajectory
is accumulated here during the rollout loop. Crucially, this module imports no
SimplerEnv: it just collects plain numbers and writes JSONL, so it can be unit
tested without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diagnostics import schema


class RolloutAccumulator:
    """Accumulate one episode's trajectory, then emit a validated record.

    Usage inside the rollout loop::

        acc = RolloutAccumulator(episode_id, policy, task, eval_mode, conditions)
        obs, info = env.reset()
        for step in range(max_steps):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            acc.record_step(ee_xyz=extract_ee_xyz(obs, info),
                            gripper_state=extract_gripper(obs, info))
            if terminated or truncated:
                break
        record = acc.finish(success=bool(info["episode_stats"]["success"]))
        write_record(path, record)

    ``extract_ee_xyz`` / ``extract_gripper`` are the only sim-specific glue and
    live in ``run_eval.py``; this class stays sim-agnostic.
    """

    def __init__(
        self,
        episode_id: str,
        policy: str,
        task: str,
        eval_mode: str,
        conditions: dict[str, Any],
    ) -> None:
        self.episode_id = episode_id
        self.policy = policy
        self.task = task
        self.eval_mode = eval_mode
        # Stamp in the known condition dict from the sweep driver — provenance
        # lives here, not in the env.
        self.conditions = dict(conditions)
        self._ee_xyz: list[list[float]] = []
        self._gripper: list[int] = []

    def record_step(self, ee_xyz: list[float], gripper_state: int) -> None:
        if len(ee_xyz) != 3:
            raise ValueError(f"ee_xyz must be a 3-vector, got {ee_xyz!r}")
        self._ee_xyz.append([float(c) for c in ee_xyz])
        self._gripper.append(int(gripper_state))

    def finish(self, success: bool) -> dict[str, Any]:
        record = {
            "episode_id": self.episode_id,
            "policy": self.policy,
            "task": self.task,
            "eval_mode": self.eval_mode,
            "conditions": self.conditions,
            "success": bool(success),
            "episode_length": len(self._ee_xyz),
            "trajectory": {
                "ee_xyz": self._ee_xyz,
                "gripper_state": self._gripper,
            },
        }
        schema.validate(record)  # fail loud at the boundary
        return record


def write_record(path: str | Path, record: dict[str, Any]) -> None:
    """Append one validated record as a JSON line."""
    schema.validate(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")

"""SimplerEnv sweep driver — the only file that knows SimplerEnv exists.

This builds on the ``DelinQu/SimplerEnv-OpenVLA`` fork harness (which runs both
Octo and OpenVLA), drives the variant-aggregation sweep, and for each episode
passes the active condition dict into the recorder. It is a thin wrapper: it
does **not** reimplement the eval loop — SimplerEnv's
``simpler_env/evaluation/maniskill2_evaluator.py`` already sweeps and logs.

Status: scaffold. The sim-specific call sites are marked ``# SIM:`` and raise
until SimplerEnv is installed (Stage 0). The seam is concrete here so the
contract is unambiguous; the diagnostics layer is already complete and tested
against fixtures without any of this running.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Iterator

from runner.record import RolloutAccumulator, write_record

# The four variant axes SimplerEnv aggregates over, each as base + 2 variations.
# These mirror the values the fork's scripts/ sweep wrappers pass; object_pose
# comes from the task's built-in pose variants.
VARIANT_AXES: dict[str, list] = {
    "background": ["base", "bridge_table", "modern_office"],
    "lighting": ["base", "brighter", "darker"],
    "distractor": [False, True],
    "table_texture": ["base", "reflective", "wooden"],
    "object_pose": ["horizontal", "vertical"],
}


def iter_conditions(axes: dict[str, list] = VARIANT_AXES) -> Iterator[dict[str, Any]]:
    """Cartesian product of the variant axes — one condition dict per cell.

    The real sweep may sample rather than fully enumerate; either way the
    condition dict produced here is exactly what gets stamped into the record,
    so labels never have to be parsed back out of the env.
    """
    keys = list(axes)
    for combo in itertools.product(*(axes[k] for k in keys)):
        yield dict(zip(keys, combo))


def extract_ee_xyz(obs: Any, info: Any) -> list[float]:  # pragma: no cover
    """Pull the end-effector xyz from a SimplerEnv step's obs/info.

    SIM: the exact key path depends on the env's observation layout; resolve it
    at Stage 1 against a live rollout (typically the TCP/EE pose in the agent
    state). Kept here so all sim-specific glue lives in one file.
    """
    raise NotImplementedError("wire up against a live SimplerEnv rollout (Stage 1)")


def extract_gripper(obs: Any, info: Any) -> int:  # pragma: no cover
    """Pull the gripper open/closed state (0/1) from a step's obs/info."""
    raise NotImplementedError("wire up against a live SimplerEnv rollout (Stage 1)")


def run_episode(
    env: Any,
    policy: Any,
    episode_id: str,
    policy_name: str,
    task: str,
    conditions: dict[str, Any],
    max_steps: int = 120,
) -> dict[str, Any]:  # pragma: no cover
    """Run one rollout and return a validated record.

    This is the wrapper around SimplerEnv's per-episode loop. The accumulation +
    provenance pattern is the point; the SIM-marked lines are the only glue.
    """
    acc = RolloutAccumulator(
        episode_id=episode_id,
        policy=policy_name,
        task=task,
        eval_mode="variant_aggregation",
        conditions=conditions,
    )
    obs, info = env.reset()  # SIM
    for _ in range(max_steps):
        action = policy.act(obs)  # SIM
        obs, _reward, terminated, truncated, info = env.step(action)  # SIM
        acc.record_step(
            ee_xyz=extract_ee_xyz(obs, info),
            gripper_state=extract_gripper(obs, info),
        )
        if terminated or truncated:
            break
    # Success is the terminal flag from episode_stats, not a separate file.
    success = bool(info["episode_stats"]["success"])  # SIM
    return acc.finish(success=success)


def run_sweep(
    env_factory: Any,
    policy: Any,
    policy_name: str,
    task: str = "google_robot_pick_coke_can",
    out_path: str | Path = "data/rollouts.jsonl",
    axes: dict[str, list] = VARIANT_AXES,
) -> None:  # pragma: no cover
    """Drive the full variant sweep, writing one record per episode.

    ``env_factory(conditions)`` should build a SimplerEnv configured for the
    given variant — that wiring is the Stage 1 task. Swapping ``policy`` between
    Octo and OpenVLA on this same harness is the entire Stage 6 headline run.
    """
    for i, conditions in enumerate(iter_conditions(axes)):
        env = env_factory(conditions)  # SIM
        record = run_episode(
            env=env,
            policy=policy,
            episode_id=f"ep_{i:05d}",
            policy_name=policy_name,
            task=task,
            conditions=conditions,
        )
        write_record(out_path, record)

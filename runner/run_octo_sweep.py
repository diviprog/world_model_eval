"""Instrumented variant-aggregation sweep for Octo on google_robot_pick_coke_can.

Stage 1–2 of the build plan. Reuses SimplerEnv's own env builder
(`build_maniskill2_env`) and mirrors the evaluator's per-episode loop, but wraps
it to accumulate the per-step end-effector trajectory + gripper and to emit one
schema JSONL record per episode (see ARCHITECTURE.md). The condition dict is
injected here from the sweep driver — the env never has to be parsed for it.

Runs in the `simpler_env` conda env. Deliberately imports nothing from the
diagnostics package (no pandas dependency): the two halves meet only on disk.

Usage:
    python runner/run_octo_sweep.py --out data/rollouts.jsonl --grid 3x3
    python runner/run_octo_sweep.py --out data/quick.jsonl --grid 2x2 --max-cells 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# SimplerEnv (sim side) — the only file that imports it.
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.policies.octo.octo_model import OctoInference

TASK = "google_robot_pick_coke_can"
ROBOT = "google_robot_static"

# Object pose variants (the coke_can_option flags) -> our object_pose label.
POSES = [
    ("upright", {"upright": True}),
    ("horizontal", {"lr_switch": True}),
    ("vertical", {"laid_vertically": True}),
]

# Each base config = one (env, scene, extra build kwargs) cell, with the
# condition axes it varies. object_pose is added per-pose below.
BASE_CONFIGS = [
    dict(key="base", env="GraspSingleOpenedCokeCanInScene-v0",
         scene="google_pick_coke_can_1_v4", extra={},
         cond=dict(background="base", lighting="base", distractor=False, table_texture="base")),
    dict(key="bg_alt", env="GraspSingleOpenedCokeCanInScene-v0",
         scene="google_pick_coke_can_1_v4_alt_background", extra={},
         cond=dict(background="alt", lighting="base", distractor=False, table_texture="base")),
    dict(key="bg_alt2", env="GraspSingleOpenedCokeCanInScene-v0",
         scene="google_pick_coke_can_1_v4_alt_background_2", extra={},
         cond=dict(background="alt2", lighting="base", distractor=False, table_texture="base")),
    dict(key="tex_cab1", env="GraspSingleOpenedCokeCanInScene-v0",
         scene="Baked_sc1_staging_objaverse_cabinet1_h870", extra={},
         cond=dict(background="base", lighting="base", distractor=False, table_texture="cabinet1")),
    dict(key="tex_cab2", env="GraspSingleOpenedCokeCanInScene-v0",
         scene="Baked_sc1_staging_objaverse_cabinet2_h870", extra={},
         cond=dict(background="base", lighting="base", distractor=False, table_texture="cabinet2")),
    dict(key="distractor", env="GraspSingleOpenedCokeCanDistractorInScene-v0",
         scene="google_pick_coke_can_1_v4", extra={},
         cond=dict(background="base", lighting="base", distractor=True, table_texture="base")),
    dict(key="distractor_more", env="GraspSingleOpenedCokeCanDistractorInScene-v0",
         scene="google_pick_coke_can_1_v4", extra={"distractor_config": "more"},
         cond=dict(background="base", lighting="base", distractor=True, table_texture="base")),
    dict(key="light_dark", env="GraspSingleOpenedCokeCanInScene-v0",
         scene="google_pick_coke_can_1_v4", extra={"slightly_darker_lighting": True},
         cond=dict(background="base", lighting="darker", distractor=False, table_texture="base")),
    dict(key="light_bright", env="GraspSingleOpenedCokeCanInScene-v0",
         scene="google_pick_coke_can_1_v4", extra={"slightly_brighter_lighting": True},
         cond=dict(background="base", lighting="brighter", distractor=False, table_texture="base")),
]

ROBOT_INIT_XY = (0.35, 0.20)
ROBOT_INIT_QUAT = np.array([0, 0, 0, 1])
# Object-init ranges from the official variant_agg script.
OBJ_X_RANGE = (-0.35, -0.12)
OBJ_Y_RANGE = (-0.02, 0.42)


def run_episode(env, model, episode_id, conditions, max_steps, obj_xy):
    """Mirror of the evaluator per-episode loop, with trajectory logging."""
    reset_opts = {
        "robot_init_options": {
            "init_xy": np.array(ROBOT_INIT_XY),
            "init_rot_quat": ROBOT_INIT_QUAT,
        },
        "obj_init_options": {"init_xy": np.array(obj_xy)},
    }
    # simpler_env-specific methods live on the unwrapped env (gymnasium no
    # longer forwards attributes through the TimeLimit wrapper); reset/step stay
    # on the wrapper so TimeLimit truncation still fires.
    base = env.unwrapped
    obs, _ = env.reset(options=reset_opts)
    is_final_subtask = base.is_final_subtask()
    instr = base.get_language_instruction()
    model.reset(instr)

    image = get_image_from_maniskill2_obs_dict(base, obs)
    ee_xyz, gripper = [], []
    predicted_terminated = truncated = False
    success = False
    steps = 0
    while not (predicted_terminated or truncated):
        raw_action, action = model.step(image, instr)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated and not is_final_subtask:
            predicted_terminated = False
            base.advance_to_next_subtask()
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        )
        success = success or bool(done)
        is_final_subtask = base.is_final_subtask()
        # log: end-effector xyz from tcp_pose, binarized commanded gripper
        ee_xyz.append([float(c) for c in np.asarray(obs["extra"]["tcp_pose"])[:3]])
        gripper.append(int(float(np.asarray(action["gripper"]).reshape(-1)[0]) > 0))
        image = get_image_from_maniskill2_obs_dict(base, obs)
        steps += 1

    stats = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
             for k, v in dict(info.get("episode_stats", {})).items()}
    return {
        "episode_id": episode_id,
        "policy": "octo-base",
        "task": TASK,
        "eval_mode": "variant_aggregation",
        "conditions": conditions,
        "success": bool(success),
        "episode_length": steps,
        "episode_stats": stats,
        "trajectory": {"ee_xyz": ee_xyz, "gripper_state": gripper},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rollouts.jsonl")
    ap.add_argument("--grid", default="3x3", help="object-init grid, e.g. 3x3")
    ap.add_argument("--max-cells", type=int, default=None, help="cap number of (config x pose) cells")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--shard", default=None, help="round-robin shard 'i/n' for multi-GPU runs")
    args = ap.parse_args(argv)

    gx, gy = (int(v) for v in args.grid.lower().split("x"))
    obj_xs = np.linspace(*OBJ_X_RANGE, gx)
    obj_ys = np.linspace(*OBJ_Y_RANGE, gy)
    control_mode = get_robot_control_mode(ROBOT, "octo")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")  # truncate

    print("loading Octo-Base...", flush=True)
    model = OctoInference(model_type="octo-base", policy_setup="google_robot", init_rng=0, action_scale=1.0)

    cells = [(cfg, pose_name, pose_kwargs)
             for cfg in BASE_CONFIGS for (pose_name, pose_kwargs) in POSES]
    if args.max_cells:
        cells = cells[: args.max_cells]
    # round-robin shard so each GPU gets a balanced mix of cells
    shard_tag = ""
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        cells = cells[si::sn]
        shard_tag = f"s{si}_"  # keep episode_id globally unique across shards

    ep = 0
    n_success = 0
    for ci, (cfg, pose_name, pose_kwargs) in enumerate(cells):
        build_kwargs = {**pose_kwargs, **cfg["extra"]}
        env = build_maniskill2_env(
            cfg["env"],
            obs_mode="rgbd", robot=ROBOT, sim_freq=513, control_mode=control_mode,
            control_freq=3, max_episode_steps=args.max_steps, scene_name=cfg["scene"],
            camera_cfgs={"add_segmentation": True}, rgb_overlay_path=None,
            **build_kwargs,
        )
        conditions = {**cfg["cond"], "object_pose": pose_name}
        print(f"[cell {ci+1}/{len(cells)}] {cfg['key']} / {pose_name}  {conditions}", flush=True)
        with open(out, "a") as fh:
            for ox in obj_xs:
                for oy in obj_ys:
                    rec = run_episode(env, model, f"ep_{shard_tag}{ep:05d}", conditions,
                                      args.max_steps, (float(ox), float(oy)))
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n_success += int(rec["success"])
                    ep += 1
            print(f"   -> {ep} episodes so far, running success {n_success}/{ep}", flush=True)
        env.close()

    print(f"SWEEP_DONE episodes={ep} success={n_success} rate={n_success/max(ep,1):.3f} out={out}", flush=True)


if __name__ == "__main__":
    main()

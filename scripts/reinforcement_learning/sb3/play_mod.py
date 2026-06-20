# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from Stable-Baselines3 (PPO or SAC, torch or jax)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default=None, help="Name of the RL agent configuration entry point.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
parser.add_argument(
    "--ml_framework", type=str, default=None, choices=["jax", "torch"],
    help="Machine learning framework (default: torch for PPO, jax for SAC).",
)
parser.add_argument(
    "--algorithm", type=str, required=True, choices=["ppo", "sac"],
    help="RL algorithm of the checkpoint.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.ml_framework is None:
    args_cli.ml_framework = "torch" if args_cli.algorithm.lower() == "ppo" else "jax"

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize

if args_cli.ml_framework == "torch":
    if args_cli.algorithm.lower() == "ppo":
        from stable_baselines3 import PPO as RLAlgorithm
    else:
        from stable_baselines3 import SAC as RLAlgorithm
else:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    if args_cli.algorithm.lower() == "ppo":
        from sbx import PPO as RLAlgorithm
    else:
        from sbx import SAC as RLAlgorithm

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import QuadLoco  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

if args_cli.agent is None:
    args_cli.agent = f"sb3_{args_cli.algorithm.lower()}_cfg_entry_point"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with stable-baselines agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve checkpoint path
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", train_task_name))
    if args_cli.checkpoint is None:
        checkpoint = "model_.*.zip" if args_cli.use_last_checkpoint else "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint, sort_alpha=False)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    # apply SAC action space bounds (must match training)
    if args_cli.algorithm.lower() == "sac":
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
    print(f"[INFO] action space: {env.action_space}")

    # load normalization wrapper if saved alongside checkpoint
    vec_norm_path = Path(checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl"))
    if vec_norm_path.exists():
        print(f"[INFO] Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    elif "normalize_input" in agent_cfg and agent_cfg.get("normalize_input"):
        env = VecNormalize(
            env,
            training=False,
            norm_obs=True,
            clip_obs=agent_cfg.get("clip_obs", 100.0),
        )

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent = RLAlgorithm.load(checkpoint_path, env, print_system_info=True)

    dt = env.unwrapped.step_dt

    obs = env.reset()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

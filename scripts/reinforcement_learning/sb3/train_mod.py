# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

import optax

from isaaclab.app import AppLauncher
import torch

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=50_000_000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default=None, help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument("--ml_framework", type=str, default=None, choices=["jax", "torch"], help="Machine learning framework (default: torch for PPO, jax for SAC).")
parser.add_argument("--algorithm", type=str, required=True, default="None", choices=["ppo", "sac"], help="RL algorithm to use for training.")
parser.add_argument("--wandb_name", type=str, default="", help="Run name for the wandb log")
parser.add_argument("--action", type=int, default=1, help="xx")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
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


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt

# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
# from stable_baselines3 import PPO
if args_cli.ml_framework == "torch":
    if args_cli.algorithm.lower() == "ppo":
        from stable_baselines3 import PPO as RLAlgorithm
    if args_cli.algorithm.lower() == "sac":
        from stable_baselines3 import SAC as RLAlgorithm
elif args_cli.ml_framework.lower() == "jax":
    # to prevent over-preallocation of vram
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    import jax.nn
    if args_cli.algorithm.lower() == "ppo":
        from sbx import PPO as RLAlgorithm
    if args_cli.algorithm.lower() == "sac":
        from sbx import SAC as RLAlgorithm
        
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    args_cli.agent = f"sb3_{algorithm}_cfg_entry_point"

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import QuadLoco  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)
# PLACEHOLDER: Extension template (do not remove this comment)
import wandb
from wandb.integration.sb3 import WandbCallback

class AdditionalLoggerCallback(BaseCallback):
    def __init__(self, log_dir: str = "", scalar_freq: int = 1, dist_freq: int = 2500, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.scalar_freq = scalar_freq  # Fast logging for numbers 
        self.dist_freq = dist_freq      # Slower logging for heavy histograms

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        log_dict = {}
        
        # if actions is not None:
        #     actions_np = np.asarray(actions)
            
        #     # 1. Log Scalar Actions (High Frequency OR The Very First Step)
        #     if self.n_calls == 1 or self.n_calls % self.scalar_freq == 0:
        #         log_dict.update({
        #             "actions/abs_mean_action": float(np.mean(np.abs(actions_np))),
        #             "actions/max_action": float(np.max(actions_np)),
        #             "actions/min_action": float(np.min(actions_np)),
        #         })
            
        #     # 2. Log Action Distributions (Low Frequency OR The Very First Step)
        #     if self.n_calls == 1 or self.n_calls % self.dist_freq == 0:
        #         # The combined graph
        #         log_dict["action_distributions/All_Joints"] = wandb.Histogram(actions_np.flatten())
                
        #         # The individual joint graphs
        #         n_actions = actions_np.shape[1]
        #         for i in range(n_actions):
        #             log_dict[f"action_distributions/joint_{i}"] = wandb.Histogram(actions_np[:, i])

        # 3. Log reference positions and actual joint positions from Isaac Lab
        try:
            env = self.training_env.unwrapped
            term = env.action_manager._terms.get("joint_pos")
            if term is not None:
                ref = term.processed_actions  # shape: (num_envs, 12)
                ref_flat = ref.flatten()
                k = max(1, int(0.05 * ref_flat.numel()))
                log_dict["actions/ref_position/abs_mean"] = float(ref.abs().mean().item())
                log_dict["actions/ref_position/top5pct"] = float(ref_flat.topk(k).values.mean().item())
                log_dict["actions/ref_position/bot5pct"] = float(ref_flat.topk(k, largest=False).values.mean().abs().item())
                log_dict["actions/ref_position/max"] = float(ref.max().abs().item())
                log_dict["actions/ref_position/min"] = float(ref.min().abs().item())
            try:
                robot = env.scene["robot"]
            except KeyError:
                robot = None
            if robot is not None:
                jpos = robot.data.joint_pos  # shape: (num_envs, 12)
                jpos_flat = jpos.flatten()
                k_j = max(1, int(0.05 * jpos_flat.numel()))
                log_dict["actions/joint_pos/abs_mean"] = float(jpos.abs().mean().item())
                log_dict["actions/joint_pos/top5pct"] = float(jpos_flat.topk(k_j).values.mean().item())
                log_dict["actions/joint_pos/bot5pct"] = float(jpos_flat.topk(k_j, largest=False).values.mean().abs().item())
                log_dict["actions/joint_pos/max"] = float(jpos.max().abs().item())
                log_dict["actions/joint_pos/min"] = float(jpos.min().abs().item())
        except AttributeError:
            pass

        # 4. Read Native Isaac Lab Data directly
        try:
            env = self.training_env.unwrapped
            if "log" in env.extras:
                for key, value in env.extras["log"].items():
                    try:
                        if isinstance(value, torch.Tensor):
                            val = value.item()
                        else:
                            val = float(value)
                        log_dict[f"isaac_native/{key}"] = val
                    except (ValueError, TypeError):
                        continue
        except AttributeError:
            pass
            
        # 4. Push everything we gathered this step to WandB simultaneously
        if log_dict:
            log_dict["timestep"] = self.num_timesteps
            wandb.log(log_dict, commit=False)
            
        return True

   
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.wandb_name:
        run_info = f"{run_info}_{args_cli.wandb_name}"
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    try:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    except TypeError:
        pass  # env_cfg may contain CUDA tensors that can't be serialized before sim init
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    run = wandb.init(
        project="bitirme",
        name=args_cli.wandb_name, 
        config=env_cfg,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    if args_cli.ml_framework == "jax":
        agent_cfg["policy_kwargs"]["activation_fn"] = jax.nn.elu
        if args_cli.algorithm.lower() == "sac" and "optimizer_class" in agent_cfg["policy_kwargs"]:
            if agent_cfg["policy_kwargs"]["optimizer_class"] and agent_cfg["policy_kwargs"]["optimizer_class"].lower() == "optax.adamw":
                agent_cfg["policy_kwargs"]["optimizer_class"] = optax.adamw
    
    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    if norm_args and norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # per_joint_low  = np.array([-2.6628, -2.4628, -2.6628, -2.4628,  -3.6072, -3.6072, -3.8072, -3.8072,  -2.9565, -2.9565, -2.9565, -2.9565], dtype=np.float32)/0.25
    # per_joint_high = np.array([ 2.4628,  2.6628,  2.4628,  2.6628,   5.1487,  5.1487,  4.9487,  4.9487,   2.3427,  2.3427,  2.3427,  2.3427], dtype=np.float32)/0.25
     # per_joint_low = np.array([-1.33, -0.61, -1.47, -0.13, -0.51, -1.16, -0.67, -1.04, -0.91, -1.13, -1.33, -2.18], dtype=np.float32)
    # per_joint_high = np.array([1.17, 2.41, 1.02, 1.95, 1.70, 2.09, 1.44, 1.05, 1.52, 1.23, 1.43, 0.83], dtype=np.float32)

    if args_cli.algorithm.lower() == "sac":
        env.action_space = gym.spaces.Box(low=-args_cli.action, high=args_cli.action, shape=(12,), dtype=np.float32)
    # if args_cli.algorithm.lower() == "ppo":
    #     env.action_space = gym.spaces.Box(low=-100, high=100, shape=(12,), dtype=np.float32)
    
    print(f"[INFO] action space set: {env.action_space}")
    checkpoint_interval = agent_cfg.pop("checkpoint_interval")
    # create agent from stable baselines
    agent = RLAlgorithm(policy_arch, env, verbose=0, tensorboard_log=log_dir, **agent_cfg) # verbose=0/1/2 for no/yes terminal logs
    if args_cli.checkpoint is not None:
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    if args_cli.algorithm.lower() == "sac":
        robot = env.unwrapped.scene["robot"]
        phys_limits   = robot.data.joint_pos_limits  # (num_envs, 12, 2)
        phys_low_sim  = phys_limits[0, :, 0].cpu().numpy().astype(np.float32)
        phys_high_sim = phys_limits[0, :, 1].cpu().numpy().astype(np.float32)
        print(f"[INFO] Physical joint pos limits (low):  {phys_low_sim}")
        print(f"[INFO] Physical joint pos limits (high): {phys_high_sim}")

    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=log_dir, name_prefix="model", verbose=2)
    callbacks = [checkpoint_callback,
                 AdditionalLoggerCallback(log_dir=log_dir),
                 LogEveryNTimesteps(n_steps=args_cli.log_interval),
                 WandbCallback(gradient_save_freq=10,
                #  model_save_path=f"models/{run.id}",
                 verbose=2)
                 ]
    # train the agent
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=10,
        )
    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    print("Saving to:")
    print(os.path.join(log_dir, "model.zip"))

    if isinstance(env, VecNormalize):
        print("Saving normalization")
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    print("Waiting for WandB to finish syncing files...")
    wandb.finish()  # Forces the script to wait until the image is safely in the cloud

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
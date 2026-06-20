# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a trained FlashSAC agent in an IsaacLab environment."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained FlashSAC agent in an IsaacLab environment.")
parser.add_argument("--task", type=str, required=True, help="IsaacLab task name.")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to agent checkpoint directory.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to play.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--config_name", type=str, default="flashSAC_base", help="FlashSAC hydra config name.")
parser.add_argument("--overrides", action="append", default=[], metavar="KEY=VALUE", help="Hydra config overrides.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random
from pathlib import Path
from typing import MutableMapping

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import flash_rl
from flash_rl.agents import create_agent
from flash_rl.types import Tensor

from env_wrapper import make_isaaclab_env  # isort: skip

FLASH_RL_CONFIG_PATH = str(Path(flash_rl.__file__).parent.parent / "configs")


def main():
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))

    device = args_cli.device if args_cli.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")

    overrides = list(args_cli.overrides) + [
        "env=isaaclab",
        f"env.env_name={args_cli.task}",
        f"num_train_envs={args_cli.num_envs}",
        "num_eval_envs=null",
        "num_record_envs=null",
        f"seed={args_cli.seed}",
    ]

    with hydra.initialize_config_dir(version_base=None, config_dir=FLASH_RL_CONFIG_PATH):
        cfg = hydra.compose(config_name=args_cli.config_name, overrides=overrides)
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    env = make_isaaclab_env(
        env_name=args_cli.task,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        device=device,
        headless=False,
        action_bounds=cfg.env.get("action_bounds", None),
        simulation_app=simulation_app,
    )

    _, env_info = env.reset(random_start_init=False)
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(args_cli.checkpoint_path)

    observations, _ = env.reset(random_start_init=False)
    prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
    completed_episodes = 0
    episode_returns = np.zeros(args_cli.num_envs)

    while completed_episodes < args_cli.num_episodes:
        actions = agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False)
        actions = np.array(actions)
        next_observations, rewards, terminateds, truncateds, _ = env.step(actions)

        episode_returns += rewards
        episode_dones = np.logical_or(terminateds, truncateds)

        for idx in range(args_cli.num_envs):
            if episode_dones[idx]:
                completed_episodes += 1
                print(f"Episode {completed_episodes}: return = {episode_returns[idx]:.2f}")
                episode_returns[idx] = 0.0
                if completed_episodes >= args_cli.num_episodes:
                    break

        observations = next_observations
        prev_transition = {"next_observation": observations}

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

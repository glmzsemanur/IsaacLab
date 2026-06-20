# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train a FlashSAC agent in an IsaacLab environment."""

"""Launch Isaac Sim Simulator first."""

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a FlashSAC agent in an IsaacLab environment.")
parser.add_argument("--task", type=str, required=True, help="IsaacLab task name (e.g. Isaac-Velocity-Flat-Unitree-A1-v0).")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel training environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_steps", type=int, default=50_000_000, help="Override total environment interaction steps.")
parser.add_argument("--config_name", type=str, default="flashSAC_base", help="FlashSAC hydra config name.")
parser.add_argument("--overrides", action="append", default=[], metavar="KEY=VALUE", help="Hydra config overrides.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.common import create_logger
from flash_rl.types import Tensor
import flash_rl.agents as _flash_rl_agents

from env_wrapper import make_isaaclab_env  # isort: skip

FLASH_RL_CONFIG_PATH = str(Path(_flash_rl_agents.__file__).parent.parent.parent / "configs")


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
        # IsaacLab GPU-sim defaults from run_isaaclab.sh
        "agent.buffer_max_length=10_000_000",
        "agent.buffer_min_length=100_000",
        "updates_per_interaction_step=2",
        "n_step=3",
    ]
    if args_cli.max_steps is not None:
        overrides.append(f"num_env_steps={args_cli.max_steps}")

    with hydra.initialize_config_dir(version_base=None, config_dir=FLASH_RL_CONFIG_PATH):
        cfg = hydra.compose(config_name=args_cli.config_name, overrides=overrides)
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_env = make_isaaclab_env(
        env_name=args_cli.task,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        device=device,
        headless=args_cli.headless,
        action_bounds=cfg.env.get("action_bounds", None),
        simulation_app=simulation_app,
    )
    _, env_info = train_env.reset()
    agent = create_agent(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    logger = create_logger(cfg)

    save_path_resolved = cfg.save_path.replace("TIMESTAMP", datetime.now().strftime("%m%d-%H%M%S"))
    save_path_base = os.path.abspath(save_path_resolved)

    if cfg.agent_load_path is not None:
        agent.load(os.path.abspath(cfg.agent_load_path))
    if cfg.buffer_load_path is not None:
        agent.load_replay_buffer(os.path.abspath(cfg.buffer_load_path))

    observations, env_infos = train_env.reset()
    actions: Optional[Tensor] = None
    transition: Optional[dict[str, Tensor]] = None
    update_counter = 0

    for interaction_step in tqdm.tqdm(range(1, int(cfg.num_interaction_steps + 1)), smoothing=0.1, mininterval=0.5):
        env_step = interaction_step * cfg.num_train_envs

        if agent.can_start_training() and transition is not None:
            actions = agent.sample_actions(interaction_step, prev_transition=transition, training=True)
        else:
            actions = train_env.action_space.sample()

        actions = np.array(actions)
        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(actions)
        next_buffer_observations = next_observations.copy()
        for env_idx in range(cfg.num_train_envs):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_obs"][env_idx]

        if "episode_info" in env_infos:
            logger.update_metric(**env_infos["episode_info"])

        transition = {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": next_buffer_observations,
        }
        agent.process_transition(transition)
        transition["next_observation"] = next_observations
        observations = next_observations

        if agent.can_start_training():
            update_counter += cfg.updates_per_interaction_step
            while update_counter >= 1:
                update_info = agent.update()
                logger.update_metric(**update_info)
                update_counter -= 1

            if cfg.metrics_per_interaction_step and interaction_step % cfg.metrics_per_interaction_step == 0:
                metrics_info = agent.get_metrics()
                logger.update_metric(**metrics_info)

            if cfg.logging_per_interaction_step and interaction_step % cfg.logging_per_interaction_step == 0:
                logger.log_metric(step=env_step)
                logger.reset()

            if (
                cfg.save_checkpoint_per_interaction_step
                and interaction_step % cfg.save_checkpoint_per_interaction_step == 0
            ):
                agent.save(os.path.join(save_path_base, f"step{interaction_step}"))

            if cfg.save_buffer_per_interaction_step and interaction_step % cfg.save_buffer_per_interaction_step == 0:
                agent.save_replay_buffer(os.path.join(save_path_base, f"step{interaction_step}"))

    logger.log_metric(step=env_step)
    logger.reset()

    try:
        import copy as _copy

        _actor_net = agent._actor.network
        if hasattr(_actor_net, "_orig_mod"):
            _actor_net = _actor_net._orig_mod

        class _InferenceActor(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.embedder = net.embedder
                self.encoder = net.encoder
                self.post_norm = net.post_norm
                self.predictor = net.predictor

            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                x = self.embedder(observations, False)
                for block in self.encoder:
                    x = block(x, False)
                x = self.post_norm(x)
                mean, _ = self.predictor.get_mean_and_std(x, False)
                return torch.tanh(mean)

        _wrapper = _copy.deepcopy(_InferenceActor(_actor_net)).cpu().eval()
        _obs_dim = _wrapper.embedder.w.w.in_features
        _jit_actor = torch.jit.trace(_wrapper, torch.zeros(1, _obs_dim))
        _export_dir = os.path.join(save_path_base, "exported")
        os.makedirs(_export_dir, exist_ok=True)
        _jit_actor.save(os.path.join(_export_dir, "actor_jit.pt"))
        print(f"\033[32m[FlashSAC]\033[0m Exported JIT actor to: {_export_dir}/actor_jit.pt")
    except Exception as e:
        print(f"\033[33m[FlashSAC]\033[0m JIT export failed: {e}")

    train_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

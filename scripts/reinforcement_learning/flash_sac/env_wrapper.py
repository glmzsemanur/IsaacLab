# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab env wrapper for FlashSAC that accepts an already-launched SimulationApp.

This is a drop-in replacement for flash_rl.envs.isaaclab that avoids creating a
second AppLauncher when train.py has already launched one at the script level.
"""

from typing import Any, Union, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from flash_rl.envs.isaaclab import ACTION_BOUNDS
from flash_rl.types import F32NDArray, NDArray


def recursive_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {k: recursive_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_numpy(v) for v in data)
    return data


class IsaacLabVectorEnv(
    VectorEnv[Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray]]
):
    """IsaacLabVectorEnv that accepts an already-created SimulationApp instance.

    When ``simulation_app`` is provided the AppLauncher call is skipped entirely,
    which avoids the "only one SimulationApp per process" constraint when train.py
    has already launched one via ``AppLauncher(args_cli)``.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        seed: int,
        device: str,
        action_bounds: float,
        to_numpy: bool = True,
        headless: bool = True,
        simulation_app=None,
    ):
        if simulation_app is None:
            from isaaclab.app import AppLauncher

            app_launcher = AppLauncher(headless=headless, device=device, enable_cameras=not headless)
            self.simulation_app = app_launcher.app
        else:
            self.simulation_app = simulation_app

        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        import isaaclab_tasks  # noqa: F401

        env_cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)
        env_cfg.seed = seed
        self.seed = seed
        self.device = device
        self.envs = gym.make(env_name, cfg=env_cfg, render_mode=None)

        self.num_envs = cast(Any, self.envs.unwrapped).num_envs
        self.max_episode_steps = cast(Any, self.envs.unwrapped).max_episode_length
        self.to_numpy = to_numpy

        self.obs_size = cast(Any, self.envs.unwrapped).single_observation_space["policy"].shape
        self.asymmetric_obs = "critic" in cast(Any, self.envs.unwrapped).single_observation_space
        if self.asymmetric_obs:
            self.critic_obs_size = cast(Any, self.envs.unwrapped).single_observation_space["critic"].shape
            self.single_observation_space = gym.spaces.Box(
                low=0.0, high=0.0, shape=self.obs_size + self.critic_obs_size, dtype=np.float32
            )
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        else:
            self.critic_obs_size = 0
            self.single_observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=self.obs_size, dtype=np.float32)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.action_bounds = action_bounds
        self.action_size = cast(Any, self.envs.unwrapped).single_action_space.shape
        self.single_action_space = gym.spaces.Box(
            low=-1.0 * self.action_bounds, high=1.0 * self.action_bounds, shape=self.action_size, dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(self, *, seed=None, options=None, random_start_init=True):
        obs_dict, infos = self.envs.reset()
        obs = obs_dict["policy"]
        if self.asymmetric_obs:
            critic_obs = obs_dict["critic"]
            obs = torch.cat((obs, critic_obs), dim=-1)
        if random_start_init:
            cast(Any, self.envs.unwrapped).episode_length_buf = torch.randint_like(
                cast(Any, self.envs.unwrapped).episode_length_buf, high=int(self.max_episode_steps)
            )
        if self.to_numpy:
            obs = obs.cpu().numpy()
            infos = recursive_to_numpy(infos)
        infos.update({"actor_observation_size": self.obs_size, "asymmetric_obs": self.asymmetric_obs})
        return obs, infos

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            torch_actions = actions.to(self.device)
        else:
            torch_actions = torch.from_numpy(actions).to(self.device)

        if self.action_bounds is not None:
            torch_actions = torch.clamp(torch_actions, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminations, truncations, infos = cast(Any, self.envs.step(torch_actions))
        obs = obs_dict["policy"]
        if self.asymmetric_obs:
            critic_obs = obs_dict["critic"]
            obs = torch.cat((obs, critic_obs), dim=-1)
        else:
            critic_obs = None
        infos = {"time_outs": truncations, "observations": {"critic": critic_obs}}
        infos["final_obs"] = obs

        unwrapped_env = cast(Any, self.envs.unwrapped)
        if hasattr(unwrapped_env, "extras") and isinstance(unwrapped_env.extras, dict):
            isaac_log = unwrapped_env.extras.get("log", {})
            episode_info: dict[str, float] = {}
            for key, value in isaac_log.items():
                try:
                    episode_info[key] = float(value.mean().cpu().item() if isinstance(value, torch.Tensor) else value)
                except (ValueError, TypeError):
                    continue
            if episode_info:
                infos["episode_info"] = episode_info

        if self.to_numpy:
            obs = obs.cpu().numpy()
            rew = rew.cpu().numpy()
            terminations = terminations.cpu().numpy()
            truncations = truncations.cpu().numpy()
            infos = recursive_to_numpy(infos)
        return obs, rew, terminations, truncations, infos

    def close(self, **kwargs: Any) -> None:
        return

    def render(self) -> None:
        raise NotImplementedError


def make_isaaclab_env(
    env_name: str,
    num_envs: int,
    seed: int,
    headless: bool = True,
    action_bounds: float | None = None,
    device: str | None = None,
    simulation_app=None,
) -> IsaacLabVectorEnv:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if action_bounds is not None:
        print(f"Action bounds overridden to {action_bounds} for {env_name}.")
    else:
        if env_name not in ACTION_BOUNDS:
            print(f"Action bounds not defined for {env_name}; using default 1.0.")
        action_bounds = ACTION_BOUNDS.get(env_name, 1.0)
    return IsaacLabVectorEnv(
        env_name=env_name,
        num_envs=num_envs,
        seed=seed,
        device=device,
        action_bounds=action_bounds,
        to_numpy=True,
        headless=headless,
        simulation_app=simulation_app,
    )

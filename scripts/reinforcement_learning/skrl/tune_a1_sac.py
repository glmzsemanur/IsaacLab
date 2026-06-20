import os
import torch
import numpy as np
import optuna
import gymnasium as gym
import copy

# Isaac Lab / Hydra Imports
from isaaclab.app import AppLauncher

# Start AppLauncher once. Headless is essential for the 5070 Ti.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

# skrl imports
from skrl.utils.runner.torch import Runner
from skrl.envs.torch import wrap_env


@hydra_task_config(
    task_name="Isaac-Velocity-Flat-Unitree-A1-v0",
    agent_cfg_entry_point="skrl_sac_cfg_entry_point"
)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):

    # --- PERSISTENT ENVIRONMENT ---
    # We initialize once to keep the GPU context stable across trials.
    print("[INFO] Initializing Persistent A1 Simulation...")
    env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=env_cfg)

    # Space consistency overrides
    env.unwrapped.single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(12,), dtype=np.float32
    )

    env.unwrapped.single_observation_space = gym.spaces.Dict({
        "policy": gym.spaces.Box(
            low=-5.0, high=5.0, shape=(48,), dtype=np.float32
        )
    })

    wrapped_env = wrap_env(env, wrapper="isaaclab")

    def objective(trial):
        final_reward = -5000.0

        # 1. HYPERPARAMETERS
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        init_entropy = trial.suggest_float("init_entropy", 0.001, 0.1, log=True)
        polyak = trial.suggest_float("polyak", 0.001, 0.02)
        grad_clip = trial.suggest_float("grad_clip", 0.1, 1.0)

        # 2. CONFIG
        current_agent_cfg = copy.deepcopy(agent_cfg)
        current_agent_cfg["agent"]["actor_learning_rate"] = lr
        current_agent_cfg["agent"]["critic_learning_rate"] = lr
        current_agent_cfg["agent"]["initial_entropy_value"] = init_entropy
        current_agent_cfg["agent"]["polyak"] = polyak
        current_agent_cfg["agent"]["grad_norm_clip"] = grad_clip

        current_agent_cfg["trainer"]["timesteps"] = 10000
        current_agent_cfg["trainer"]["close_environment_at_exit"] = False

        trial_name = f"trial_{trial.number}_morning"

        current_agent_cfg["agent"]["experiment"] = {
            "write_interval": 100,
            "directory": "runs/morning_tuning",
            "wandb": True,
            "wandb_kwargs": {
                "project": "A1_Locomotion_Overnight",
                "name": trial_name,
                "group": "v5_long_run",
                "reinit": True
            }
        }

        try:
            wrapped_env.reset()
            runner = Runner(wrapped_env, current_agent_cfg)
            runner.run()

            # 3. DYNAMIC AGENT RECOVERY
            agent = None
            trainer = getattr(runner, "trainer", None)

            search_spots = [trainer, runner]
            attr_names = ["agents", "_agents", "agent", "_agent"]

            for spot in search_spots:
                if spot is None:
                    continue
                for name in attr_names:
                    val = getattr(spot, name, None)
                    if val is not None:
                        agent = val[0] if isinstance(val, (list, tuple)) else val
                        break
                if agent:
                    break

            # 4. REWARD SCRAPING
            search_targets = [trainer, agent]

            for target in search_targets:
                if target is None:
                    continue

                data_source = getattr(target, "tracking_data", {})
                reward_keys = [
                    k for k in data_source.keys()
                    if "reward" in k.lower()
                ]

                if reward_keys:
                    best_key = sorted(
                        reward_keys, key=len, reverse=True
                    )[0]
                    vals = data_source[best_key]
                    if len(vals) > 0:
                        final_reward = float(np.mean(vals[-20:]))
                        break

            # 5. LAST RESORT: Raw memory
            if final_reward == -5000.0 and agent is not None:
                if hasattr(agent, "memory") and agent.memory is not None:
                    try:
                        r_tensor = agent.memory.get_tensor_by_name("rewards")
                        if r_tensor is not None:
                            final_reward = float(torch.mean(r_tensor).item())
                    except Exception:
                        pass

            print(f">>> TRIAL {trial.number} SUCCESS! VALUE: {final_reward}")

        except Exception as e:
            print(f"Trial {trial.number} encountered an error: {e}")
            final_reward = -5000.0

        finally:
            torch.cuda.empty_cache()

            import wandb
            if wandb.run is not None:
                wandb.finish()

            torch.cuda.empty_cache()

        return final_reward

    # --- STUDY EXECUTION ---
    study = optuna.create_study(
        study_name="a1_sac_tuning_fixed",
        direction="maximize",
        storage="sqlite:///a1_tuning_fixed.db",
        load_if_exists=True
    )

    print("\nStarting Tuning Session. Monitoring results in Optuna Dashboard...")
    study.optimize(objective, n_trials=30)

    print("\n" + "=" * 50)
    print(f"BEST PARAMETERS: {study.best_params}")
    print(f"BEST VALUE: {study.best_value}")
    print("=" * 50)

    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
"""Curriculum evaluation for rough-terrain SB3 agents.

Runs multiple *trials*. In each trial every environment is explicitly assigned
to a target terrain level (balanced across all levels), then one episode is
collected per environment.

Pass criterion: the robot survived the full episode (truncated, not terminated)
AND its maximum Euclidean distance from spawn exceeded 50% of the expected
travel distance (‖cmd_xy‖ × episode_length_s).  Each env's command is fixed
for the entire episode — resampling is disabled by setting the command term's
resampling_time_range to (1e9, 1e9).

Usage:
    python scripts/reinforcement_learning/sb3/eval_curriculum.py \\
        --task Isaac-Velocity-Rough-Unitree-A1-v0 \\
        --checkpoint logs/sb3/unitree_a1_rough/<run>/model_50000_steps.zip \\
        --algorithm sac --num_envs 4000 --num_trials 5
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Curriculum evaluation for SB3 rough-terrain agents")
parser.add_argument("--task", type=str, required=True, help="IsaacLab gym environment ID")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to SB3 .zip checkpoint file")
parser.add_argument("--algorithm", type=str, required=True, choices=["ppo", "sac"],
                    help="RL algorithm used during training.")
parser.add_argument("--use_jax", action="store_true", default=False,
                    help="Use SBX (JAX) backend instead of stable-baselines3 (torch).")
parser.add_argument("--num_envs", type=int, default=4000,
                    help="Parallel envs. Best set to num_levels × num_terrain_cols × k.")
parser.add_argument("--num_trials", type=int, default=20,
                    help="Number of full sweeps (each sweep = one episode per env).")
parser.add_argument("--output", type=str, default=None,
                    help="CSV path for results (default: beside checkpoint)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.agent = f"sb3_{args_cli.algorithm}_cfg_entry_point"
# Match play_mod.py: SAC defaults to JAX backend unless explicitly overridden.
if not args_cli.use_jax and args_cli.algorithm == "sac":
    args_cli.use_jax = True
sys.argv = [sys.argv[0]] + hydra_args

# ── Launch Isaac Sim FIRST ────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── All other imports follow ──────────────────────────────────────────────────
import logging
import os
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import tqdm

if args_cli.use_jax:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    if args_cli.algorithm == "ppo":
        from sbx import PPO as RLAlgorithm
    else:
        from sbx import SAC as RLAlgorithm
else:
    if args_cli.algorithm == "ppo":
        from stable_baselines3 import PPO as RLAlgorithm
    else:
        from stable_baselines3 import SAC as RLAlgorithm

from stable_baselines3.common.vec_env import VecNormalize

import QuadLoco  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_action_bounds(path: str) -> float | None:
    import re
    m = re.search(r"action_(\d+(?:\.\d+)?)", path)
    if m:
        v = float(m.group(1))
        print(f"[eval] Detected action_bounds={v} from checkpoint path.")
        return v
    print("[eval] Could not detect action_bounds from path.")
    return None


def _termination_from_extras(
    unwrapped: Any,
    dones_np: np.ndarray,
    device: str,
    _cache: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive fell/survived from Isaac's extras["log"] termination signals.

    With SB3 we only have dones_np (combined terminated + truncated).
    Fall keys from Episode_Termination/* determine which dones are falls.
    Anything done but not a fall is treated as a timeout (survived).
    """
    extras_log: dict = {}
    if hasattr(unwrapped, "extras") and isinstance(unwrapped.extras, dict):
        extras_log = unwrapped.extras.get("log", {})

    if not _cache:
        timeout_key = None
        term_keys: list[str] = []
        for k in extras_log:
            if not k.startswith("Episode_Termination/"):
                continue
            if "time_out" in k.lower():
                timeout_key = k
            else:
                term_keys.append(k)
        _cache["timeout_key"] = timeout_key
        _cache["term_keys"] = term_keys
        if term_keys or timeout_key:
            print(f"\n[eval] Isaac termination keys found:")
            for k in term_keys:
                print(f"         fall   → {k}")
            if timeout_key:
                print(f"         timeout→ {timeout_key}")
        else:
            print("[eval] No Episode_Termination keys in extras['log'] — all dones treated as falls")

    timeout_key = _cache["timeout_key"]
    term_keys   = _cache["term_keys"]
    num_envs    = len(dones_np)

    if term_keys:
        fell_mask = np.zeros(num_envs, dtype=bool)
        for k in term_keys:
            v = extras_log.get(k)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                fell_mask |= (v > 0.5).cpu().numpy()
            else:
                fell_mask |= np.asarray(v, dtype=float) > 0.5
        fell = fell_mask & dones_np.astype(bool)
    else:
        fell = dones_np.copy().astype(bool)

    if timeout_key is not None:
        v = extras_log.get(timeout_key)
        if v is not None:
            if isinstance(v, torch.Tensor):
                timed_out = (v > 0.5).cpu().numpy()
            else:
                timed_out = np.asarray(v, dtype=float) > 0.5
            survived = timed_out & ~fell
        else:
            survived = dones_np.astype(bool) & ~fell
    else:
        survived = dones_np.astype(bool) & ~fell

    return fell, survived


def _build_terrain_labels(terrain) -> dict[int, str]:
    try:
        gen = terrain.cfg.terrain_generator
        num_cols: int = int(gen.num_cols)
        labels: dict[int, str] = {}
        col = 0
        for name, sub in gen.sub_terrains.items():
            n = round(float(sub.proportion) * num_cols)
            for _ in range(n):
                labels[col] = name
                col += 1
        while col < num_cols:
            labels[col] = name  # type: ignore[possibly-undefined]
            col += 1
        return labels
    except Exception as exc:
        print(f"[eval] Could not build terrain labels ({exc}); using numeric indices.")
        return {}


def _assign_levels(num_envs: int, num_levels: int, terrain_types_np: np.ndarray) -> np.ndarray:
    assignment = np.zeros(num_envs, dtype=np.int64)
    for t in np.unique(terrain_types_np):
        idx = np.where(terrain_types_np == t)[0]
        for k, env_id in enumerate(idx):
            assignment[env_id] = k * num_levels // len(idx)
    return assignment


def _make_summary(
    records: list[dict],
    num_terrain_levels: int,
    terrain_labels: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    label_map = terrain_labels if terrain_labels else {t: str(t) for t in df["terrain_type"].unique()}
    df["terrain_name"] = df["terrain_type"].map(label_map).fillna(df["terrain_type"].astype(str))

    rows_level = []
    for level in range(num_terrain_levels):
        sub = df[df["level"] == level]
        if sub.empty:
            continue
        n = len(sub)
        n_passed = int(sub["passed"].sum())
        rows_level.append({
            "level":        level,
            "n_attempts":   n,
            "n_passed":     n_passed,
            "n_failed":     n - n_passed,
            "success_rate": round(n_passed / n * 100, 1),
        })
    level_df = pd.DataFrame(rows_level).set_index("level")

    rows_type = []
    for level in range(num_terrain_levels):
        for t in sorted(df["terrain_type"].unique()):
            sub = df[(df["level"] == level) & (df["terrain_type"] == t)]
            if sub.empty:
                continue
            n = len(sub)
            n_passed = int(sub["passed"].sum())
            rows_type.append({
                "level":        level,
                "terrain_type": int(t),
                "terrain_name": label_map.get(int(t), str(t)),
                "n_attempts":   n,
                "n_passed":     n_passed,
                "success_rate": round(n_passed / n * 100, 1),
            })
    type_df = (
        pd.DataFrame(rows_type).set_index(["level", "terrain_type"])
        if rows_type else pd.DataFrame()
    )

    pivot_df = pd.DataFrame()
    if rows_type:
        raw = pd.DataFrame(rows_type)
        grouped = (
            raw.groupby(["terrain_name", "level"])
            .apply(lambda g: round(g["n_passed"].sum() / g["n_attempts"].sum() * 100, 1))
            .rename("success_rate")
            .reset_index()
        )
        pivot_df = grouped.pivot(index="terrain_name", columns="level", values="success_rate")
        pivot_df.columns = [f"lvl{c}" for c in pivot_df.columns]
        pivot_df.index.name = "terrain_type"

    return level_df, type_df, pivot_df


def _format_combined_pivot(pct_df: pd.DataFrame, count_df: pd.DataFrame, total_df: pd.DataFrame) -> str:
    name_w = max(len(str(n)) for n in pct_df.index) + 2
    all_counts = [int(v) for row in count_df.values for v in row if not pd.isna(v)]
    all_totals = [int(v) for row in total_df.values for v in row if not pd.isna(v)]
    c_w = len(str(max(all_counts, default=0)))
    t_w = len(str(max(all_totals, default=0)))
    cell_w = 5 + 2 + c_w + 1 + t_w + 1
    col_w = cell_w + 2

    lines = []
    header = f"{'terrain_type':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in pct_df.columns)
    lines.append(header)
    lines.append("-" * len(header))
    for name, row in pct_df.iterrows():
        cells = ""
        for col in pct_df.columns:
            pct = row.get(col, float("nan"))
            c = count_df.loc[name, col] if (name in count_df.index and col in count_df.columns) else float("nan")
            t = total_df.loc[name, col] if (name in total_df.index and col in total_df.columns) else float("nan")
            if pd.isna(pct) or pd.isna(c) or pd.isna(t):
                cells += f"{'—':>{col_w}}"
            else:
                cell = f"{pct:.1f} ({int(c)}/{int(t)})"
                cells += f"{cell:>{cell_w}}  "
        lines.append(f"{str(name):<{name_w}}{cells}")
    return "\n".join(lines)


def _format_float_pivot(pivot_df: pd.DataFrame, fmt: str = ".3f") -> str:
    if pivot_df.empty:
        return "  (no data)"
    name_w = max(len(str(n)) for n in pivot_df.index) + 2
    col_w = max(9, max(len(str(c)) for c in pivot_df.columns) + 2)
    lines = []
    header = f"{'terrain_type':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in pivot_df.columns)
    lines.append(header)
    lines.append("-" * len(header))
    for name, row in pivot_df.iterrows():
        cells = ""
        for v in row:
            cells += f"{'—':>{col_w}}" if pd.isna(v) else f"{v:>{col_w}{fmt}}"
        lines.append(f"{str(name):<{name_w}}{cells}")
    return "\n".join(lines)


def _format_pivot(pivot_df: pd.DataFrame) -> str:
    name_w = max(len(str(n)) for n in pivot_df.index) + 2
    col_w  = 7
    lines = []
    header = f"{'terrain_type':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in pivot_df.columns)
    lines.append(header)
    lines.append("-" * len(header))
    for name, row in pivot_df.iterrows():
        cells = ""
        for v in row:
            cells += f"{'—':>{col_w}}" if pd.isna(v) else f"{v:>{col_w}.1f}"
        lines.append(f"{str(name):<{name_w}}{cells}")
    return "\n".join(lines)


def _format_level_summary(df: pd.DataFrame) -> str:
    sep = "─" * 102
    hdr = (f"  {'Level':>7}  {'n_eps':>6}  {'Succ%':>7}  {'BC%':>7}  {'DF%':>7}  "
           f"{'TrackErr':>13}  {'Smooth':>13}  {'CoT (pass)':>13}")
    lines = [sep, hdr, sep]

    def _fmt(series: "pd.Series") -> str:
        if len(series) == 0:
            return "           —"
        m, s = float(series.mean()), float(series.std())
        return f"{m:6.3f}±{s:.3f}"

    for level in sorted(df["level"].unique()):
        sub = df[df["level"] == level]
        sub_p = sub[sub["passed"]]
        lines.append(
            f"  {level:>7}  {len(sub):>6}  {sub['passed'].mean()*100:>6.1f}%  "
            f"{sub['base_contact'].mean()*100:>6.1f}%  "
            f"{sub['distance_fail'].mean()*100:>6.1f}%  "
            f"{_fmt(sub['tracking_err']):>13}  "
            f"{_fmt(sub['action_smoothness']):>13}  "
            f"{_fmt(sub_p['cot']):>13}"
        )

    sub_pass = df[df["passed"]]
    lines += [sep,
        f"  {'Overall':>7}  {len(df):>6}  {df['passed'].mean()*100:>6.1f}%  "
        f"{df['base_contact'].mean()*100:>6.1f}%  "
        f"{df['distance_fail'].mean()*100:>6.1f}%  "
        f"{_fmt(df['tracking_err']):>13}  "
        f"{_fmt(df['action_smoothness']):>13}  "
        f"{_fmt(sub_pass['cot']):>13}",
        sep,
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    device: str = args_cli.device if args_cli.device else "cuda:0"
    num_envs: int = args_cli.num_envs
    num_trials: int = args_cli.num_trials

    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    _ts = datetime.now().strftime("%m%d-%H%M%S")
    output_path = args_cli.output or os.path.join(
        os.path.dirname(checkpoint_path), f"curriculum_eval_{_ts}.csv"
    )

    # ── Environment ───────────────────────────────────────────────────────────
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    env_cfg.seed = agent_cfg.get("seed", 42)

    if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "max_init_terrain_level"):
        env_cfg.scene.terrain.max_init_terrain_level = None
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None

    env_cfg.episode_length_s = 10.0

    raw_env   = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    unwrapped = raw_env.unwrapped

    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)

    agent_cfg = process_sb3_cfg(agent_cfg, num_envs)
    sb3_env = Sb3VecEnvWrapper(raw_env, fast_variant=True)

    vec_norm_path = Path(checkpoint_path).parent / "model_vecnormalize.pkl"
    if vec_norm_path.exists():
        print(f"[eval] Loading VecNormalize: {vec_norm_path}")
        sb3_env = VecNormalize.load(str(vec_norm_path), sb3_env)
        sb3_env.training = False
        sb3_env.norm_reward = False

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"[eval] Loading checkpoint: {checkpoint_path}")
    _detect_action_bounds(checkpoint_path)
    custom_objects = {
        "observation_space": sb3_env.observation_space,
        "action_space": sb3_env.action_space,
    }
    sb3_agent = RLAlgorithm.load(checkpoint_path, env=sb3_env, device=device, custom_objects=custom_objects)

    # ── Terrain metadata ──────────────────────────────────────────────────────
    terrain = unwrapped.scene.terrain
    num_terrain_levels: int = int(terrain.terrain_origins.shape[0])
    num_terrain_types:  int = int(terrain.terrain_origins.shape[1])
    terrain_labels = _build_terrain_labels(terrain)
    terrain_types_np: np.ndarray = terrain.terrain_types.cpu().numpy()

    print(f"[eval] Pass criterion: survived full episode AND max_dist > 0.5 × (||cmd_xy|| × episode_length_s) per robot")

    level_assignment = _assign_levels(num_envs, num_terrain_levels, terrain_types_np)
    envs_per_level = np.bincount(level_assignment, minlength=num_terrain_levels)
    total_episodes = num_trials * num_envs
    expected_per_cell = num_envs / (num_terrain_levels * num_terrain_types)

    print(f"[eval] Terrain grid : {num_terrain_levels} levels × {num_terrain_types} types")
    print(f"[eval] Envs per level: {envs_per_level.tolist()}")
    print(f"[eval] Expected per (level, type) cell: {expected_per_cell:.1f} envs/trial")
    print(
        f"[eval] Total : {num_trials} trials × {num_envs} envs = {total_episodes:,} episodes "
        f"→ ~{int(expected_per_cell * num_trials)} samples per (level, type) cell"
    )
    if terrain_labels:
        unique = dict.fromkeys(terrain_labels.values())
        print(f"[eval] Sub-terrains: {', '.join(unique)}")

    # ── Robot asset + timing ──────────────────────────────────────────────────
    robot = unwrapped.scene["robot"]

    try:
        step_dt: float = float(unwrapped.cfg.sim.dt) * int(unwrapped.cfg.decimation)
        episode_length_s: float = step_dt * int(unwrapped.max_episode_length)
    except Exception:
        step_dt = 0.02
        episode_length_s = 10.0
    try:
        robot_mass: float = float(robot.data.default_mass[0].sum().item())
    except Exception:
        robot_mass = 12.0
    print(f"[eval] episode_length_s={episode_length_s:.1f}s  |  pass criterion: survived + dist > 50% of expected")
    print(f"[eval] Robot mass: {robot_mass:.2f} kg  |  step_dt: {step_dt:.4f}s")

    cmd_term = unwrapped.command_manager._terms["base_velocity"]
    cmd_term.cfg.resampling_time_range = (1e9, 1e9)

    records: list[dict] = []
    pbar = tqdm.tqdm(total=total_episodes, desc="episodes", unit="ep")
    _term_cache: dict = {}
    _base_contact_key: str | None = "unset"

    try:
        for _ in range(num_trials):
            # ── Force balanced level assignment ───────────────────────────────
            terrain.terrain_levels[:] = torch.from_numpy(level_assignment).to(device)
            terrain.env_origins[:] = terrain.terrain_origins[
                terrain.terrain_levels, terrain.terrain_types
            ]
            obs = sb3_env.reset()

            spawn_xy: np.ndarray = terrain.env_origins.cpu().numpy()[:, :2].copy()
            assigned_levels: np.ndarray = terrain.terrain_levels.cpu().numpy().copy()

            completed = np.zeros(num_envs, dtype=bool)
            cur_reward_sum = np.zeros(num_envs, dtype=np.float64)
            cur_tracking_err_sum = np.zeros(num_envs, dtype=np.float64)
            cur_power_sum = np.zeros(num_envs, dtype=np.float64)
            cur_episode_steps = np.zeros(num_envs, dtype=np.int64)
            cur_smoothness_sum = np.zeros(num_envs, dtype=np.float64)
            cur_smoothness_steps = np.zeros(num_envs, dtype=np.int64)
            prev_actions_np: np.ndarray | None = None

            cmd_term.time_left[:] = 1e9

            initial_cmd_xy: np.ndarray = (
                unwrapped.command_manager.get_command("base_velocity")
                .cpu().numpy()[:, :2].copy()
            )
            expected_dist: np.ndarray = np.linalg.norm(initial_cmd_xy, axis=1) * episode_length_s

            max_dist = np.zeros(num_envs, dtype=np.float64)

            while not completed.all():
                robot_xy: np.ndarray = robot.data.root_pos_w.cpu().numpy()[:, :2]
                dist = np.linalg.norm(robot_xy - spawn_xy, axis=1)
                np.maximum(max_dist, dist, out=max_dist)

                with torch.inference_mode():
                    actions_np, _ = sb3_agent.predict(obs, deterministic=True)

                obs, rews_np, dones_np, _ = sb3_env.step(actions_np)

                cur_reward_sum += rews_np.astype(np.float64)

                # ── Per-step metrics ──────────────────────────────────────────
                cmd_vel_np = unwrapped.command_manager.get_command("base_velocity").cpu().numpy()
                actual_lin = robot.data.root_lin_vel_b.cpu().numpy()[:, :2]
                actual_ang = robot.data.root_ang_vel_b.cpu().numpy()[:, 2:3]
                vel_err_np = np.concatenate(
                    [cmd_vel_np[:, :2] - actual_lin, cmd_vel_np[:, 2:3] - actual_ang], axis=1
                )
                cur_tracking_err_sum += np.linalg.norm(vel_err_np, axis=1)

                torques_np = robot.data.applied_torque.cpu().numpy()
                jvel_np = robot.data.joint_vel.cpu().numpy()
                cur_power_sum += np.abs(torques_np * jvel_np).sum(axis=1) * step_dt

                if prev_actions_np is None:
                    prev_actions_np = actions_np.copy()
                else:
                    cur_smoothness_sum += np.linalg.norm(actions_np - prev_actions_np, axis=1)
                    cur_smoothness_steps += 1
                    prev_actions_np = actions_np.copy()
                cur_episode_steps += 1

                fell, survived = _termination_from_extras(
                    unwrapped, dones_np, device, _term_cache
                )

                if _base_contact_key == "unset":
                    raw = next((k for k in _term_cache.get("term_keys", []) if "base_contact" in k), None)
                    _base_contact_key = raw.split("/")[-1] if raw else None
                if _base_contact_key:
                    try:
                        term_idx = unwrapped.termination_manager._term_name_to_term_idx[_base_contact_key]
                        base_contact_np = unwrapped.termination_manager._last_episode_dones[:, term_idx].cpu().numpy()
                    except (KeyError, AttributeError):
                        base_contact_np = np.zeros(num_envs, dtype=bool)
                else:
                    base_contact_np = np.zeros(num_envs, dtype=bool)

                for idx in range(num_envs):
                    if dones_np[idx] and not completed[idx]:
                        passed = bool(survived[idx] and max_dist[idx] > 0.5 * expected_dist[idx])
                        bc = bool(base_contact_np[idx] and not passed)
                        ep_steps = max(int(cur_episode_steps[idx]), 1)
                        dist_for_cot = max(max_dist[idx], 1e-3)
                        records.append({
                            "level":             int(assigned_levels[idx]),
                            "terrain_type":      int(terrain_types_np[idx]),
                            "passed":            passed,
                            "terminated":        bool(fell[idx]),
                            "base_contact":      bc,
                            "distance_fail":     bool(not passed and not bc),
                            "max_dist":          round(max_dist[idx], 2),
                            "expected_dist":     round(expected_dist[idx], 2),
                            "episode_reward":    round(float(cur_reward_sum[idx]), 4),
                            "tracking_err":      round(cur_tracking_err_sum[idx] / ep_steps, 4),
                            "cot":               round(cur_power_sum[idx] / (robot_mass * dist_for_cot), 4),
                            "action_smoothness": round(cur_smoothness_sum[idx] / max(int(cur_smoothness_steps[idx]), 1), 4),
                        })
                        cur_reward_sum[idx] = 0.0
                        cur_tracking_err_sum[idx] = 0.0
                        cur_power_sum[idx] = 0.0
                        cur_episode_steps[idx] = 0
                        cur_smoothness_sum[idx] = 0.0
                        cur_smoothness_steps[idx] = 0
                        completed[idx] = True
                        pbar.update(1)

    except KeyboardInterrupt:
        print(f"\n[eval] Interrupted — using {len(records)} episodes collected so far.")
    finally:
        pbar.close()

    # ── Results ───────────────────────────────────────────────────────────────
    if not records:
        print("[eval] No episodes completed.")
        sb3_env.close()
        return

    level_df, type_df, pivot_df = _make_summary(records, num_terrain_levels, terrain_labels)

    all_rewards = [r["episode_reward"] for r in records]
    mean_reward = float(np.mean(all_rewards))
    std_reward  = float(np.std(all_rewards))

    _df_rec = pd.DataFrame(records)
    _label_map = terrain_labels if terrain_labels else {t: str(t) for t in _df_rec["terrain_type"].unique()}
    _df_rec["terrain_name"] = _df_rec["terrain_type"].map(_label_map).fillna(_df_rec["terrain_type"].astype(str))
    _grp = _df_rec.groupby(["terrain_name", "level"])

    passed_pivot   = _grp["passed"].sum().unstack("level").fillna(0).astype(int)
    attempts_pivot = _grp["passed"].count().unstack("level").fillna(0).astype(int)
    bc_pivot       = _grp["base_contact"].sum().unstack("level").fillna(0).astype(int)
    at_pivot       = _grp["base_contact"].count().unstack("level").fillna(0).astype(int)
    for df in (passed_pivot, attempts_pivot, bc_pivot, at_pivot):
        df.columns = [f"lvl{c}" for c in df.columns]
        df.index.name = "terrain_type"
    bc_pct_pivot = (bc_pivot / at_pivot * 100).round(1)
    dist_fail_pivot = _grp["distance_fail"].sum().unstack("level").fillna(0).astype(int)
    dist_fail_pivot.columns = [f"lvl{c}" for c in dist_fail_pivot.columns]
    dist_fail_pivot.index.name = "terrain_type"
    dist_fail_pct_pivot = (dist_fail_pivot / at_pivot * 100).round(1)

    tracking_pivot   = _grp["tracking_err"].mean().unstack("level").round(3)
    smoothness_pivot = _grp["action_smoothness"].mean().unstack("level").round(3)
    _grp_passed = _df_rec[_df_rec["passed"]].groupby(["terrain_name", "level"])
    cot_pivot   = _grp_passed["cot"].mean().unstack("level").round(2)
    for df in (tracking_pivot, smoothness_pivot, cot_pivot):
        df.columns = [f"lvl{c}" for c in df.columns]
        df.index.name = "terrain_type"

    algo = f"SB3-{'SBX' if args_cli.use_jax else 'Torch'} {args_cli.algorithm.upper()}"
    print("\n" + "=" * 70)
    print(f"  Curriculum Evaluation — {algo} — {args_cli.task}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Trials: {num_trials}  |  Episodes: {len(records)}")
    print(f"  Mean episode reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("=" * 70)

    print("\nPer-level success rates (all terrain types):")
    print(level_df.to_string(float_format="%.1f"))
    print("\nSummary — aggregated across all terrain types:")
    print(_format_level_summary(_df_rec))

    if not pivot_df.empty:
        print("\nSuccess rate % (count/total)  —  rows: terrain type  |  columns: difficulty level")
        print(_format_combined_pivot(pivot_df, passed_pivot, attempts_pivot))
        print("\nBase contacts % (count/total)  —  rows: terrain type  |  columns: difficulty level")
        print(_format_combined_pivot(bc_pct_pivot, bc_pivot, at_pivot))
        print("\nDistance fail % (count/total)  —  rows: terrain type  |  columns: difficulty level")
        print(_format_combined_pivot(dist_fail_pct_pivot, dist_fail_pivot, at_pivot))
        print("\nMean velocity tracking error (m/s)  —  rows: terrain type  |  columns: difficulty level")
        print(_format_float_pivot(tracking_pivot, fmt=".3f"))
        print("\nMean action smoothness (Δaction/step)  —  rows: terrain type  |  columns: difficulty level")
        print(_format_float_pivot(smoothness_pivot, fmt=".3f"))
        print("\nMean Cost of Transport — successful episodes only (J/kg·m)  —  rows: terrain type  |  columns: difficulty level")
        print(_format_float_pivot(cot_pivot, fmt=".2f"))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    level_df.to_csv(output_path)
    saved = [output_path]
    if not type_df.empty:
        type_csv = output_path.replace(".csv", "_by_type.csv")
        type_df.to_csv(type_csv)
        saved.append(type_csv)

        pivot_csv = output_path.replace(".csv", "_pivot.csv")
        pivot_df.to_csv(pivot_csv)
        saved.append(pivot_csv)

        pivot_txt = output_path.replace(".csv", "_pivot.txt")
        with open(pivot_txt, "w") as f:
            f.write(f"Curriculum Evaluation — {algo} — {args_cli.task}\n")
            f.write(f"Checkpoint : {checkpoint_path}\n")
            f.write(f"Trials: {num_trials}  |  Episodes: {len(records)}\n")
            f.write(f"episode_length_s={episode_length_s:.1f}s  |  step_dt={step_dt:.4f}s  |  max_episode_length={unwrapped.max_episode_length} steps\n")
            f.write(f"Mean episode reward: {mean_reward:.2f} ± {std_reward:.2f}  (N={len(all_rewards)})\n\n")
            f.write("Summary — aggregated across all terrain types:\n")
            f.write(_format_level_summary(_df_rec) + "\n\n")
            f.write("Success rate % (count/total)  —  rows: terrain type  |  columns: difficulty level\n")
            f.write(_format_combined_pivot(pivot_df, passed_pivot, attempts_pivot) + "\n")
            f.write("\nBase contacts % (count/total)  —  rows: terrain type  |  columns: difficulty level\n")
            f.write(_format_combined_pivot(bc_pct_pivot, bc_pivot, at_pivot) + "\n")
            f.write("\nDistance fail % (count/total)  —  rows: terrain type  |  columns: difficulty level\n")
            f.write(_format_combined_pivot(dist_fail_pct_pivot, dist_fail_pivot, at_pivot) + "\n")
            f.write("\nMean velocity tracking error (m/s)  —  rows: terrain type  |  columns: difficulty level\n")
            f.write(_format_float_pivot(tracking_pivot, fmt=".3f") + "\n")
            f.write("\nMean action smoothness (Δaction/step)  —  rows: terrain type  |  columns: difficulty level\n")
            f.write(_format_float_pivot(smoothness_pivot, fmt=".3f") + "\n")
            f.write("\nMean Cost of Transport — successful episodes only (J/kg·m)  —  rows: terrain type  |  columns: difficulty level\n")
            f.write(_format_float_pivot(cot_pivot, fmt=".2f") + "\n")
        saved.append(pivot_txt)

    print(f"\n[eval] Saved: {', '.join(saved)}")

    sb3_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

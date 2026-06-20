"""Directional command evaluation for a trained SB3 agent.

Evaluates 16 compass directions at two speeds (default 0.5 and 1.0 m/s) on the
specified environment. Also runs a yaw-tracking sweep: pure spin (vx=vy=0) and
walk-and-turn (vx=const).

NOTE: The checkpoint and task must share the same observation space.

Outputs
-------
  compass_<ts>.pdf      — 16-direction compass with both speeds overlaid
  yaw_<ts>.pdf          — two-panel yaw tracking (spin + walk-and-turn)
  eval_report_<ts>.txt  — plain-text tables for all metrics

Usage
-----
  python scripts/reinforcement_learning/sb3/eval_commands.py \\
      --task Isaac-Velocity-Flat-Unitree-A1-v0 \\
      --checkpoint logs/sb3/unitree_a1_flat/<run>/model_50000_steps.zip \\
      --algorithm sac --num_envs 512 --headless
"""

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Directional evaluation for SB3 agents.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .zip checkpoint file.")
parser.add_argument("--algorithm", type=str, required=True, choices=["ppo", "sac"],
                    help="RL algorithm used during training.")
parser.add_argument("--use_jax", action="store_true", default=False,
                    help="Use SBX (JAX) backend instead of stable-baselines3 (torch).")
parser.add_argument("--num_envs", type=int, default=2000)
parser.add_argument("--speeds", type=str, default="0.5,1.0")
parser.add_argument("--duration", type=float, default=5.0, help="Eval duration per direction (s).")
parser.add_argument("--warmup", type=float, default=1.0, help="Stabilisation warmup (s).")
parser.add_argument(
    "--angles", type=str,
    default=",".join(str(i * 22.5) for i in range(16)),
    help="Comma-separated angles in degrees (0=forward/+X, 90=left/+Y).",
)
parser.add_argument("--yaw_rates", type=str, default="-1.0,-0.5,0.5,1.0")
parser.add_argument("--yaw_fwd_speeds", type=str, default="0.25,0.5,1.0",
                    help="Comma-separated forward speeds for walk+turn evaluation.")
parser.add_argument("--output_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.agent = f"sb3_{args_cli.algorithm}_cfg_entry_point"
# Match play_mod.py: SAC defaults to JAX backend unless explicitly overridden.
if not args_cli.use_jax and args_cli.algorithm == "sac":
    args_cli.use_jax = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after sim launch) ────────────────────────────────────────────────
import logging
import os
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import torch

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

# ── Direction labels ──────────────────────────────────────────────────────────
_NAMED: dict[int, str] = {
    0: "Fwd", 45: "Fwd-L", 90: "Left", 135: "Back-L",
    180: "Back", 225: "Back-R", 270: "Right", 315: "Fwd-R",
}

def _dir_label(deg: float) -> str:
    return _NAMED.get(int(round(deg)) % 360, f"{deg:.1f}°")


_VERSION = "1.1"


# ── Action bounds detection ───────────────────────────────────────────────────
def _detect_action_bounds(path: str) -> float | None:
    import re
    m = re.search(r"action_(\d+(?:\.\d+)?)", path)
    if m:
        v = float(m.group(1))
        print(f"[eval] Detected action_bounds={v} from checkpoint path.")
        return v
    print("[eval] Could not detect action_bounds from path.")
    return None


# ── Fall detection via Isaac termination log ─────────────────────────────────
def _fell_from_extras(unwrapped: Any, dones_np: np.ndarray, _cache: dict) -> np.ndarray:
    """Return per-env bool array: True if the robot fell this step.

    Reads Isaac's Episode_Termination/* log keys (base_contact, bad_orientation, etc.).
    """
    extras_log: dict = {}
    if hasattr(unwrapped, "extras") and isinstance(unwrapped.extras, dict):
        extras_log = unwrapped.extras.get("log", {})

    if not _cache:
        term_keys = [k for k in extras_log
                     if k.startswith("Episode_Termination/") and "time_out" not in k.lower()]
        _cache["term_keys"] = term_keys
        if term_keys:
            print(f"\n[eval] Isaac termination keys: {term_keys}")
        else:
            raise RuntimeError(
                "[eval] No Episode_Termination/* keys found in Isaac extras. "
                "Fall detection requires Isaac's contact termination log."
            )

    term_keys = _cache["term_keys"]
    num_envs  = len(dones_np)

    fell_mask = np.zeros(num_envs, dtype=bool)
    for k in term_keys:
        v = extras_log.get(k)
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            fell_mask |= (v > 0.5).cpu().numpy()
        else:
            fell_mask |= np.asarray(v, dtype=float) > 0.5
    return fell_mask & dones_np.astype(bool)


# ── Sim helpers ───────────────────────────────────────────────────────────────
def _force_command(unwrapped: Any, vx: float, vy: float, wz: float = 0.0) -> None:
    terms = unwrapped.command_manager._terms
    if "base_velocity" not in terms:
        return
    term = terms["base_velocity"]
    if hasattr(term, "cfg") and hasattr(term.cfg, "ranges"):
        r = term.cfg.ranges
        r.lin_vel_x = (vx, vx)
        r.lin_vel_y = (vy, vy)
        r.ang_vel_z = (wz, wz)
        if hasattr(r, "heading"):
            r.heading = (0.0, 0.0)
    if hasattr(term, "command"):
        term.command[:, 0] = vx
        term.command[:, 1] = vy
        term.command[:, 2] = wz


def _canonical_reset(unwrapped: Any) -> None:
    try:
        robot = unwrapped.scene["robot"]
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += unwrapped.scene.env_origins
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:13]))
        joint_pos = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    except Exception as exc:
        print(f"[WARNING] canonical reset failed: {exc}")


# ── Per-direction evaluation ──────────────────────────────────────────────────
def _evaluate_direction(
    sb3_env,
    unwrapped: Any,
    sb3_agent,
    vx: float, vy: float, wz: float,
    num_steps: int,
    warmup_steps: int,
    step_dt: float,
    device: str,
    robot_mass: float,
) -> dict[str, Any]:
    num_envs = unwrapped.num_envs
    robot = unwrapped.scene["robot"]

    _force_command(unwrapped, 0.0, 0.0, 0.0)
    obs = sb3_env.reset()
    _canonical_reset(unwrapped)

    for _ in range(warmup_steps):
        _force_command(unwrapped, 0.0, 0.0, 0.0)
        with torch.inference_mode():
            actions, _ = sb3_agent.predict(obs, deterministic=True)
        obs, _, _, _ = sb3_env.step(actions)

    initial_pos = robot.data.root_pos_w[:, :2].clone()
    _force_command(unwrapped, vx, vy, wz)

    pos_history    = torch.zeros(num_steps, num_envs, 2, device=device)
    alive          = torch.ones(num_envs, dtype=torch.bool, device=device)
    vel_err_sum    = np.zeros(num_envs, dtype=np.float64)
    yaw_err_sum    = np.zeros(num_envs, dtype=np.float64)
    power_sum      = np.zeros(num_envs, dtype=np.float64)
    ang_disp_sum   = np.zeros(num_envs, dtype=np.float64)
    smoothness_sum = np.zeros(num_envs, dtype=np.float64)
    reward_sum     = np.zeros(num_envs, dtype=np.float64)
    step_count     = np.zeros(num_envs, dtype=np.int64)
    smooth_steps   = np.zeros(num_envs, dtype=np.int64)
    prev_act: np.ndarray | None = None
    _term_cache: dict = {}

    for s in range(num_steps):
        _force_command(unwrapped, vx, vy, wz)
        with torch.inference_mode():
            actions, _ = sb3_agent.predict(obs, deterministic=True)
        obs, rews_np, dones_np, _ = sb3_env.step(actions)

        fell_np = _fell_from_extras(unwrapped, dones_np, _term_cache)
        alive &= ~torch.as_tensor(fell_np, device=device)
        alive_np = alive.cpu().numpy()

        pos_history[s] = robot.data.root_pos_w[:, :2] - initial_pos

        cmd_np = unwrapped.command_manager.get_command("base_velocity").cpu().numpy()
        lin_np = robot.data.root_lin_vel_b.cpu().numpy()[:, :2]
        ang_np = robot.data.root_ang_vel_b.cpu().numpy()[:, 2:3]
        vel_vec = np.concatenate([cmd_np[:, :2] - lin_np, cmd_np[:, 2:3] - ang_np], axis=1)
        vel_err_sum += np.linalg.norm(vel_vec, axis=1) * alive_np
        yaw_err_sum += np.abs(cmd_np[:, 2] - ang_np[:, 0]) * alive_np
        ang_disp_sum += np.abs(ang_np[:, 0]) * step_dt * alive_np

        torques = robot.data.applied_torque.cpu().numpy()
        jvel    = robot.data.joint_vel.cpu().numpy()
        power_sum += np.abs(torques * jvel).sum(axis=1) * step_dt * alive_np

        reward_sum += rews_np
        if prev_act is not None:
            smoothness_sum += np.linalg.norm(actions - prev_act, axis=1) * alive_np
            smooth_steps   += alive_np.astype(np.int64)
        prev_act = actions.copy()
        step_count += alive_np.astype(np.int64)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    pos_np   = pos_history.cpu().numpy()
    alive_np = alive.cpu().numpy()
    ref      = alive_np if alive_np.any() else np.ones(num_envs, dtype=bool)

    time_vec      = np.arange(1, num_steps + 1) * step_dt
    expected_traj = np.stack([vx * time_vec, vy * time_vec], axis=1)

    mean_traj = pos_np[:, ref, :].mean(axis=1)
    std_traj  = pos_np[:, ref, :].std(axis=1)
    endpoint  = pos_np[-1, ref, :]
    ep_err    = np.linalg.norm(endpoint - expected_traj[-1], axis=1)

    expected_dist = float(np.linalg.norm(expected_traj[-1]))
    low_perf = int((np.linalg.norm(pos_np[-1], axis=1) < 0.5 * expected_dist).sum()) \
        if expected_dist > 1e-3 else 0

    ep_steps = np.maximum(step_count, 1)

    if pos_np.shape[0] > 1:
        step_dists = np.linalg.norm(np.diff(pos_np, axis=0), axis=2)
        path_lengths = step_dists.sum(axis=0)
    else:
        path_lengths = np.linalg.norm(pos_np[-1], axis=1)
    ep_path = np.maximum(path_lengths, 1e-3)

    expected_path_len = float(np.sqrt(vx**2 + vy**2)) * num_steps * step_dt
    low_perf_path = int((path_lengths < 0.5 * expected_path_len).sum()) \
        if expected_path_len > 1e-3 else 0

    return {
        "mean_endpoint":     endpoint.mean(axis=0),
        "std_endpoint":      endpoint.std(axis=0),
        "expected_endpoint": expected_traj[-1],
        "mean_traj":         mean_traj,
        "std_traj":          std_traj,
        "traj_err_per_step": np.linalg.norm(mean_traj - expected_traj, axis=1),
        "endpoint_err_mean": float(ep_err.mean()),
        "endpoint_err_std":  float(ep_err.std()),
        "traj_err_mean":     float(np.linalg.norm(mean_traj - expected_traj, axis=1).mean()),
        "vel_err_mean":      float(np.mean(vel_err_sum[ref] / ep_steps[ref])),
        "yaw_err_mean":      float(np.mean(yaw_err_sum[ref] / ep_steps[ref])),
        "cot_mean":          float(np.mean(power_sum[ref] / (robot_mass * ep_path[ref]))),
        "energy_per_rad":    float(np.mean(power_sum[ref] / np.maximum(ang_disp_sum[ref], 1e-3))),
        "smoothness_mean":   float(np.mean(smoothness_sum[ref] / np.maximum(smooth_steps[ref], 1))),
        "reward_mean":       float(np.mean(reward_sum[ref] / num_steps)),
        "survival_rate":     float(alive_np.mean()),
        "fall_count":        int((~alive_np).sum()),
        "low_perf_count":    low_perf,
        "low_perf_path":     low_perf_path,
        "total_envs":        num_envs,
    }


# ── Compass plot ──────────────────────────────────────────────────────────────
def _compass_plot(
    results_by_speed: dict[float, list[dict]],
    angles_deg: list[float],
    task: str,
    checkpoint: str,
    save_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    speeds    = sorted(results_by_speed.keys())
    max_speed = speeds[-1]
    cmap      = LinearSegmentedColormap.from_list("GR", ["#27ae60", "#c0392b"])
    norm      = Normalize(vmin=0.0, vmax=0.4)

    max_expected = float(np.linalg.norm(
        results_by_speed[max_speed][0]["expected_endpoint"]
    ))
    plot_lim = max_expected * 2.0

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_aspect("equal")

    for r in [2.5, 5.0]:
        ax.add_patch(plt.Circle((0, 0), r, color="gray", fill=False,
                                lw=1.0, ls="--", alpha=0.45, zorder=2))

    ax.axhline(0, color="black", lw=0.4, alpha=0.18)
    ax.axvline(0, color="black", lw=0.4, alpha=0.18)
    ax.plot(0, 0, "k+", ms=14, zorder=10)

    _styles: dict[float, tuple[str, str, float]] = {}
    for i, sp in enumerate(speeds):
        marker = "o" if i == len(speeds) - 1 else "D"
        _styles[sp] = ("-", marker, 1.0)

    for dir_idx, angle_deg in enumerate(angles_deg):
        θ = math.radians(angle_deg)
        cos_θ, sin_θ = math.cos(θ), math.sin(θ)
        ann_lines: list[str] = [_dir_label(angle_deg)]

        exp_max = results_by_speed[max_speed][dir_idx]["expected_endpoint"]
        ax.plot([0, exp_max[0]], [0, exp_max[1]], "--", color="#3498db",
                lw=1.5, alpha=0.7, zorder=4)
        ax.plot(cos_θ * 2.5, sin_θ * 2.5, "x", ms=9, mec="#3498db", mew=2.0,
                alpha=0.9, zorder=5)
        ax.plot(exp_max[0], exp_max[1], "x", ms=9, mec="#3498db", mew=2.0,
                alpha=0.9, zorder=5)

        for sp in speeds:
            res = results_by_speed[sp][dir_idx]
            ls, marker, alpha = _styles[sp]
            exp  = res["expected_endpoint"]
            expected_dist = float(np.linalg.norm(exp))
            color = cmap(norm(res["endpoint_err_mean"] / max(expected_dist, 1e-3)))
            ep   = res["mean_endpoint"]
            std  = res["std_endpoint"]
            traj = res["mean_traj"]

            idx2 = np.linspace(0, len(traj) - 1, min(25, len(traj)), dtype=int)
            ax.plot(traj[idx2, 0], traj[idx2, 1], ls, color=color,
                    lw=1.8, alpha=alpha, zorder=6)
            ax.plot(ep[0], ep[1], marker, ms=6, color=color, alpha=alpha, zorder=7,
                    markeredgecolor="black", markeredgewidth=0.5)
            ax.errorbar(ep[0], ep[1], xerr=std[0], yerr=std[1],
                        fmt="none", ecolor=color, alpha=0.6 * alpha, capsize=2, zorder=6)

            ann_lines.append(
                f"{sp:.1f}↦ ep={res['endpoint_err_mean']:.2f} v={res['vel_err_mean']:.2f}\n"
                f"      yaw={res['yaw_err_mean']:.2f}"
            )

        top = results_by_speed[max_speed][dir_idx]
        extra_low = max(0, top["low_perf_count"] - top["fall_count"])
        ann_lines.append(f"fail={top['fall_count']}+{extra_low}/{top['total_envs']}")

        ann_r = plot_lim * 0.65
        ax.text(cos_θ * ann_r, sin_θ * ann_r, "\n".join(ann_lines),
                ha="center", va="center", fontsize=10, zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", lw=0.8, alpha=0.92))

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xlabel("World X — Forward (+) / Backward (−)  [m]", fontsize=14)
    ax.set_ylabel("World Y — Left (+) / Right (−)  [m]", fontsize=14)
    ax.tick_params(labelsize=12)

    algo = f"SB3-{'SBX' if args_cli.use_jax else 'Torch'} {args_cli.algorithm.upper()}"
    speed_str = "  &  ".join(f"{s} m/s" for s in speeds)
    ax.set_title(
        f"Compass Evaluation — {algo} — {task}\n"
        f"{speed_str}  |  {len(angles_deg)} directions  |  "
        f"○={max_speed} m/s   ◇={speeds[0]} m/s   colour=relative endpoint error",
        fontsize=14, pad=14,
    )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("Relative endpoint error  [0=perfect, 1=full distance off]", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Compass saved: {save_path}")


# ── Yaw plot ──────────────────────────────────────────────────────────────────
def _yaw_plot(
    spin_results: list[tuple[float, dict]],
    walk_results_by_speed: dict[float, list[tuple[float, dict]]],
    yaw_rates: list[float],
    task: str,
    save_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fwd_speeds = sorted(walk_results_by_speed.keys())
    n_panels   = 1 + max(len(fwd_speeds), 1)
    fig, axes  = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))

    ax = axes[0]
    wz_vals  = [wz for wz, _ in spin_results]
    yaw_errs = [r["yaw_err_mean"]      for _, r in spin_results]
    vel_errs = [r["vel_err_mean"]      for _, r in spin_results]
    survs    = [r["survival_rate"]*100 for _, r in spin_results]
    xx = np.arange(len(spin_results))
    b_yaw = ax.bar(xx - 0.22, yaw_errs, width=0.4, color="#2980b9", alpha=0.82, label="Yaw err (rad/s)")
    ax.bar(xx + 0.22, vel_errs, width=0.4, color="#27ae60", alpha=0.82, label="Lin vel err (m/s)")
    for surv, b in zip(survs, b_yaw):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + max(yaw_errs) * 0.02,
                f"{surv:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(max(yaw_errs) * 1.4, 0.3))
    ax.set_xticks(xx)
    ax.set_xticklabels([f"{w:+.2f}" for w in wz_vals], fontsize=9)
    ax.set_xlabel("Commanded yaw rate (rad/s)", fontsize=10)
    ax.set_ylabel("Tracking error", fontsize=10)
    ax.set_title("Pure Spin  (vx = vy = 0)", fontsize=11)
    ax.legend(fontsize=8)

    for panel_idx, fwd_speed in enumerate(fwd_speeds):
        walk_results = walk_results_by_speed[fwd_speed]
        ax = axes[panel_idx + 1]

        wz_vals  = [wz for wz, _ in walk_results]
        yaw_errs = [r["yaw_err_mean"] for _, r in walk_results]
        vel_errs = [r["vel_err_mean"]  for _, r in walk_results]
        survs    = [r["survival_rate"] * 100 for _, r in walk_results]
        low_path = [r["low_perf_path"] for _, r in walk_results]
        xx = np.arange(len(walk_results))

        b_yaw = ax.bar(xx - 0.22, yaw_errs, width=0.4, color="#2980b9", alpha=0.82, label="Yaw err (rad/s)")
        ax.bar(xx + 0.22, vel_errs, width=0.4, color="#27ae60", alpha=0.82, label="Lin vel err (m/s)")
        for surv, low, b in zip(survs, low_path, b_yaw):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + max(yaw_errs) * 0.02,
                    f"{surv:.0f}%\nlow={low}", ha="center", va="bottom", fontsize=7)

        ax.set_ylim(0, 0.8 if panel_idx == 0 else 0.3)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"{w:+.2f}" for w in wz_vals], fontsize=9)
        ax.set_xlabel("Commanded yaw rate (rad/s)", fontsize=10)
        ax.set_ylabel("Tracking error", fontsize=10)
        ax.set_title(f"Walk + Turn  (vx = {fwd_speed:.2f} m/s)", fontsize=11)
        ax.legend(fontsize=8)

    algo = f"SB3-{'SBX' if args_cli.use_jax else 'Torch'} {args_cli.algorithm.upper()}"
    fig.suptitle(f"Yaw Tracking Evaluation — {algo} — {task}", fontsize=12, y=1.01)
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Yaw plot saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    device         = args_cli.device if args_cli.device else "cuda:0"
    speeds         = [float(s.strip()) for s in args_cli.speeds.split(",")]
    angles_deg     = [float(a.strip()) for a in args_cli.angles.split(",")]
    yaw_rates      = [float(r.strip()) for r in args_cli.yaw_rates.split(",")]
    yaw_fwd_speeds = [float(s.strip()) for s in args_cli.yaw_fwd_speeds.split(",")]
    checkpoint   = os.path.abspath(args_cli.checkpoint)
    output_dir   = args_cli.output_dir or os.path.dirname(checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Environment ───────────────────────────────────────────────────────────
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device
    env_cfg.seed           = agent_cfg.get("seed", 42)

    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None
    env_cfg.episode_length_s = args_cli.warmup + args_cli.duration + 2.0

    raw_env  = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    unwrapped = raw_env.unwrapped

    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)

    agent_cfg = process_sb3_cfg(agent_cfg, args_cli.num_envs)
    sb3_env = Sb3VecEnvWrapper(raw_env, fast_variant=True)

    # Load VecNormalize stats if they exist alongside the checkpoint.
    vec_norm_path = Path(checkpoint).parent / "model_vecnormalize.pkl"
    if vec_norm_path.exists():
        print(f"[eval] Loading VecNormalize: {vec_norm_path}")
        sb3_env = VecNormalize.load(str(vec_norm_path), sb3_env)
        sb3_env.training = False
        sb3_env.norm_reward = False

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"[eval] Loading checkpoint: {checkpoint}")
    action_bounds = _detect_action_bounds(checkpoint)
    # Override stored spaces with the live env's spaces so SB3's space check passes.
    # The network output dimension (not the bounds) determines architecture compatibility.
    custom_objects = {
        "observation_space": sb3_env.observation_space,
        "action_space": sb3_env.action_space,
    }
    sb3_agent = RLAlgorithm.load(checkpoint, env=sb3_env, device=device, custom_objects=custom_objects)

    # ── Robot + timing ────────────────────────────────────────────────────────
    robot = unwrapped.scene["robot"]
    try:
        step_dt: float = float(unwrapped.cfg.sim.dt) * int(unwrapped.cfg.decimation)
    except Exception:
        step_dt = 0.02
    try:
        robot_mass: float = float(robot.data.default_mass[0].sum().item())
    except Exception:
        robot_mass = 12.0

    num_steps    = max(1, int(args_cli.duration / step_dt))
    warmup_steps = max(1, int(args_cli.warmup / step_dt))

    algo = f"SB3-{'SBX' if args_cli.use_jax else 'Torch'} {args_cli.algorithm.upper()}"
    print(f"[eval] step_dt={step_dt:.4f}s  steps={num_steps}  warmup={warmup_steps}")
    print(f"[eval] action_bounds={action_bounds}")
    print(f"[eval] robot_mass={robot_mass:.2f} kg  algorithm={algo}")
    print(f"[eval] {len(angles_deg)} directions  speeds={speeds}  yaw_rates={yaw_rates}\n")

    cmd_term = unwrapped.command_manager._terms["base_velocity"]
    cmd_term.cfg.resampling_time_range = (1e9, 1e9)

    eval_kwargs = dict(
        sb3_env=sb3_env, unwrapped=unwrapped, sb3_agent=sb3_agent,
        num_steps=num_steps, warmup_steps=warmup_steps,
        step_dt=step_dt, device=device,
        robot_mass=robot_mass,
    )

    # ── Compass evaluation ────────────────────────────────────────────────────
    results_by_speed: dict[float, list[dict]] = {}

    for speed in speeds:
        results_by_speed[speed] = []
        print(f"{'─'*65}")
        print(f"Linear  |  speed={speed} m/s  |  {len(angles_deg)} directions")
        print(f"{'─'*65}")
        for angle_deg in angles_deg:
            θ  = math.radians(angle_deg)
            vx = speed * math.cos(θ)
            vy = speed * math.sin(θ)
            print(f"  [{_dir_label(angle_deg):8s} {angle_deg:5.1f}°]  ({vx:+.3f},{vy:+.3f}) ...",
                  end="", flush=True)
            res = _evaluate_direction(vx=vx, vy=vy, wz=0.0, **eval_kwargs)
            results_by_speed[speed].append(res)
            print(f"  ep={res['endpoint_err_mean']:.3f}m  vel={res['vel_err_mean']:.3f}  "
                  f"cot={res['cot_mean']:.2f}  rew={res['reward_mean']:.3f}  "
                  f"fell={res['fall_count']}/{res['total_envs']}")

    # ── Yaw evaluation ────────────────────────────────────────────────────────
    spin_results: list[tuple[float, dict]] = []
    print(f"\n{'─'*65}")
    print("Yaw sweep — pure spin  (vx = vy = 0)")
    print(f"{'─'*65}")
    for wz in yaw_rates:
        print(f"  [vx=0.00, wz={wz:+.2f}] ...", end="", flush=True)
        res = _evaluate_direction(vx=0.0, vy=0.0, wz=wz, **eval_kwargs)
        spin_results.append((wz, res))
        print(f"  yaw_err={res['yaw_err_mean']:.3f}  drift={res['endpoint_err_mean']:.3f}m  "
              f"fell={res['fall_count']}/{res['total_envs']}")

    walk_results_by_speed: dict[float, list[tuple[float, dict]]] = {}
    for fwd_speed in yaw_fwd_speeds:
        walk_results_by_speed[fwd_speed] = []
        print(f"\n{'─'*65}")
        print(f"Yaw sweep — walk+turn  (vx={fwd_speed:.2f} m/s)")
        print(f"{'─'*65}")
        for wz in yaw_rates:
            print(f"  [vx={fwd_speed:.2f}, wz={wz:+.2f}] ...", end="", flush=True)
            res = _evaluate_direction(vx=fwd_speed, vy=0.0, wz=wz, **eval_kwargs)
            walk_results_by_speed[fwd_speed].append((wz, res))
            print(f"  yaw_err={res['yaw_err_mean']:.3f}  vel_err={res['vel_err_mean']:.3f}  "
                  f"cot={res['cot_mean']:.2f}  low={res['low_perf_path']}  "
                  f"fell={res['fall_count']}/{res['total_envs']}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _compass_plot(results_by_speed, angles_deg, args_cli.task, checkpoint,
                  os.path.join(output_dir, f"compass_v{_VERSION}_{ts}.pdf"))
    _yaw_plot(spin_results, walk_results_by_speed, yaw_rates,
              args_cli.task, os.path.join(output_dir, f"yaw_v{_VERSION}_{ts}.pdf"))

    # ── Text report ───────────────────────────────────────────────────────────
    W = 125
    lines = [
        "=" * W,
        f"  DIRECTIONAL COMMAND EVALUATION — {algo}  [eval v{_VERSION}]",
        f"  task       : {args_cli.task}",
        f"  checkpoint : {checkpoint}",
        f"  step_dt={step_dt:.4f}s  robot_mass={robot_mass:.2f}kg",
        "=" * W,
    ]

    hdr = (f"{'Direction':<10} {'Angle':>6}  {'Ep.Err':>7} {'±Std':>6}  "
           f"{'TrajErr':>7}  {'VelErr':>7}  {'YawErr':>7}  "
           f"{'CoT':>6}  {'Smooth':>8}  {'Reward':>8}  {'Surv%':>6}  {'Fell+Low':>8}")
    sep = "─" * W

    for speed in speeds:
        lines += ["", sep, f"  Speed: {speed} m/s", sep, hdr, sep]
        for angle_deg, res in zip(angles_deg, results_by_speed[speed]):
            extra = max(0, res["low_perf_count"] - res["fall_count"])
            lines.append(
                f"{_dir_label(angle_deg):<10} {angle_deg:>5.1f}°  "
                f"{res['endpoint_err_mean']:>7.3f} {res['endpoint_err_std']:>6.3f}  "
                f"{res['traj_err_mean']:>7.3f}  {res['vel_err_mean']:>7.3f}  "
                f"{res['yaw_err_mean']:>7.3f}  "
                f"{res['cot_mean']:>6.2f}  {res['smoothness_mean']:>8.4f}  "
                f"{res['reward_mean']:>8.4f}  "
                f"{res['survival_rate']*100:>5.1f}%  "
                f"{res['fall_count']}+{extra}/{res['total_envs']}"
            )

    lines += ["", sep, "  YAW SWEEP — Pure Spin  (vx = vy = 0)", sep,
              f"{'wz (rad/s)':>12}  {'YawErr':>8}  {'Drift(m)':>9}  {'VelErr':>8}  {'Surv%':>6}  {'Fell':>5}", sep]
    for wz, res in spin_results:
        lines.append(
            f"{wz:>+12.2f}  {res['yaw_err_mean']:>8.4f}  {res['endpoint_err_mean']:>9.4f}  "
            f"{res['vel_err_mean']:>8.4f}  {res['survival_rate']*100:>5.1f}%  {res['fall_count']:>5}/{res['total_envs']}"
        )

    for fwd_speed, walk_results in walk_results_by_speed.items():
        lines += ["", sep, f"  YAW SWEEP — Walk+Turn  (vx={fwd_speed:.2f} m/s)", sep,
                  f"{'wz (rad/s)':>12}  {'YawErr':>8}  {'VelErr':>8}  {'CoT':>6}  {'Surv%':>6}  {'Fell':>5}  {'LowPath':>7}", sep]
        for wz, res in walk_results:
            lines.append(
                f"{wz:>+12.2f}  {res['yaw_err_mean']:>8.4f}  {res['vel_err_mean']:>8.4f}  "
                f"{res['cot_mean']:>6.2f}  {res['survival_rate']*100:>5.1f}%  {res['fall_count']:>5}  "
                f"{res['low_perf_path']:>7}/{res['total_envs']}"
            )

    sum_hdr = (f"  {'Speed':>8}  {'Ep.Err':>12}  {'TrajErr':>12}  {'VelErr':>12}  "
               f"{'YawErr':>12}  {'CoT':>12}  {'Smooth':>12}  {'Reward':>12}  {'Surv%':>8}  {'Fell+Low':>18}")
    lines += ["", sep, "  SUMMARY — mean ± std across all 16 directions", sep, sum_hdr, sep]

    all_ep, all_traj, all_vel, all_yaw, all_cot, all_smooth, all_rew, all_surv = [], [], [], [], [], [], [], []
    all_fell, all_low = 0, 0
    n_dirs = len(angles_deg)
    for speed in speeds:
        ep      = [r["endpoint_err_mean"] for r in results_by_speed[speed]]
        traj    = [r["traj_err_mean"]     for r in results_by_speed[speed]]
        vel     = [r["vel_err_mean"]      for r in results_by_speed[speed]]
        yaw     = [r["yaw_err_mean"]      for r in results_by_speed[speed]]
        cot     = [r["cot_mean"]          for r in results_by_speed[speed]]
        smooth  = [r["smoothness_mean"]   for r in results_by_speed[speed]]
        rew     = [r["reward_mean"]       for r in results_by_speed[speed]]
        surv    = [r["survival_rate"]*100 for r in results_by_speed[speed]]
        fell    = sum(r["fall_count"]                               for r in results_by_speed[speed])
        low     = sum(max(0, r["low_perf_count"] - r["fall_count"]) for r in results_by_speed[speed])
        n_envs  = results_by_speed[speed][0]["total_envs"]
        all_ep += ep; all_traj += traj; all_vel += vel; all_yaw += yaw
        all_cot += cot; all_smooth += smooth; all_rew += rew; all_surv += surv
        all_fell += fell; all_low += low
        def fmt(v): return f"{float(np.mean(v)):6.3f}±{float(np.std(v)):.3f}"
        lines.append(
            f"  {speed:>5.1f} m/s  {fmt(ep):>12}  {fmt(traj):>12}  {fmt(vel):>12}  "
            f"{fmt(yaw):>12}  {fmt(cot):>12}  {fmt(smooth):>12}  {fmt(rew):>12}  "
            f"{float(np.mean(surv)):>7.1f}%  {fell}+{low}/{n_dirs * n_envs}"
        )
    def fmt(v): return f"{float(np.mean(v)):6.3f}±{float(np.std(v)):.3f}"
    lines += [sep,
        f"  {'Overall':>8}  {fmt(all_ep):>12}  {fmt(all_traj):>12}  {fmt(all_vel):>12}  "
        f"{fmt(all_yaw):>12}  {fmt(all_cot):>12}  {fmt(all_smooth):>12}  {fmt(all_rew):>12}  "
        f"{float(np.mean(all_surv)):>7.1f}%  {all_fell}+{all_low}/{len(speeds) * n_dirs * n_envs}"
    ]

    yaw_sum_hdr = (f"  {'vx (m/s)':>10}  {'YawErr':>12}  {'VelErr':>12}  "
                   f"{'CoT':>12}  {'Surv%':>8}")
    lines += ["", sep, "  SUMMARY — Yaw sweep (mean ± std across all wz rates)",
              sep, yaw_sum_hdr, sep]
    spin_yaw  = [r["yaw_err_mean"]      for _, r in spin_results]
    spin_vel  = [r["vel_err_mean"]      for _, r in spin_results]
    spin_surv = [r["survival_rate"]*100 for _, r in spin_results]
    lines.append(
        f"  {'spin':>7}      {fmt(spin_yaw):>12}  {fmt(spin_vel):>12}  "
        f"{'     N/A':>12}  {float(np.mean(spin_surv)):>7.1f}%"
    )
    all_sw_yaw, all_sw_vel, all_sw_cot, all_sw_surv = [], [], [], []
    for fwd_speed, walk_results in sorted(walk_results_by_speed.items()):
        yaw  = [r["yaw_err_mean"]      for _, r in walk_results]
        vel  = [r["vel_err_mean"]      for _, r in walk_results]
        cot  = [r["cot_mean"]          for _, r in walk_results]
        surv = [r["survival_rate"]*100 for _, r in walk_results]
        all_sw_yaw += yaw; all_sw_vel += vel; all_sw_cot += cot; all_sw_surv += surv
        lines.append(
            f"  {fwd_speed:>7.2f} m/s  {fmt(yaw):>12}  {fmt(vel):>12}  "
            f"{fmt(cot):>12}  {float(np.mean(surv)):>7.1f}%"
        )
    all_sw_yaw += spin_yaw; all_sw_vel += spin_vel; all_sw_surv += spin_surv
    lines += [sep,
        f"  {'Overall':>8}      {fmt(all_sw_yaw):>12}  {fmt(all_sw_vel):>12}  "
        f"{'     N/A':>12}  {float(np.mean(all_sw_surv)):>7.1f}%"
    ]

    lines.append("=" * W)
    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(output_dir, f"eval_report_v{_VERSION}_{ts}.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport saved: {report_path}")

    sb3_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

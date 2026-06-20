from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import store_code_state


class GreedyOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner that periodically logs a deterministic (greedy) return.

    Every greedy_eval_interval iterations, after the PPO update, runs
    num_steps_per_env steps with act_inference (mean actions, no noise) and
    logs completed episode returns as 'avg_return'. This matches what FlashSAC
    and SB3 report as their eval return, removing the upward bias from
    exploration noise in the training rollout.

    The interval is set to run approximately 10 times over the full training.
    """

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        greedy_rewbuffer: deque = deque(maxlen=100)
        greedy_cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        greedy_eval_interval = max(1, num_learning_iterations // 10)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if (it - start_iter + 1) % greedy_eval_interval == 0:
                obs = self._run_greedy_eval(obs, greedy_rewbuffer, greedy_cur_reward_sum)

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _run_greedy_eval(
        self,
        obs: torch.Tensor,
        greedy_rewbuffer: deque,
        greedy_cur_reward_sum: torch.Tensor,
    ) -> torch.Tensor:
        self.eval_mode()
        greedy_cur_reward_sum.zero_()

        with torch.inference_mode():
            for _ in range(self.num_steps_per_env):
                actions = self.alg.policy.act_inference(obs)
                obs, rewards, dones, _ = self.env.step(actions.to(self.env.device))
                obs = obs.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                greedy_cur_reward_sum += rewards
                new_ids = (dones > 0).nonzero(as_tuple=False)
                greedy_rewbuffer.extend(greedy_cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                greedy_cur_reward_sum[new_ids] = 0.0

        self.train_mode()
        return obs

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width, pad)

        greedy_rewbuffer: deque = locs.get("greedy_rewbuffer", deque())
        if len(greedy_rewbuffer) > 0:
            self.writer.add_scalar("avg_return", statistics.mean(greedy_rewbuffer), locs["it"])

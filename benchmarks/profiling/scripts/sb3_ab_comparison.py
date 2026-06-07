#!/usr/bin/env python3
"""A/B Time Decomposition: SB3 ReplayBuffer vs FastReplayBuffer.

Runs the same GranularProfiledDQN with both buffer implementations
and produces a side-by-side time decomposition table.

Usage:
    conda activate FastReplay
    python benchmarks/profiling/scripts/sb3_ab_comparison.py --steps 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.type_aliases import TrainFreq, RolloutReturn

from fastreplay_sb3 import FastReplayBuffer


class GranularProfiledDQN(DQN):
    """DQN subclass that instruments every operation in the training loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timers: dict[str, float] = defaultdict(float)
        self.call_counts: dict[str, int] = defaultdict(int)

    def _tick(self, label: str) -> float:
        return time.perf_counter()

    def _tock(self, label: str, start: float) -> None:
        self.timers[label] += time.perf_counter() - start
        self.call_counts[label] += 1

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: ActionNoise | None = None,
        learning_starts: int = 0,
        log_interval: int | None = None,
    ) -> RolloutReturn:
        self.policy.set_training_mode(False)
        num_collected_steps, num_collected_episodes = 0, 0
        assert isinstance(env, VecEnv)
        assert train_freq.frequency > 0

        t = self._tick("callback_rollout")
        callback.on_rollout_start()
        self._tock("callback_rollout", t)

        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            t = self._tick("policy_inference")
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            self._tock("policy_inference", t)

            t = self._tick("env_step")
            new_obs, rewards, dones, infos = env.step(actions)
            self._tock("env_step", t)

            t = self._tick("bookkeeping")
            self.num_timesteps += env.num_envs
            num_collected_steps += 1
            self._tock("bookkeeping", t)

            t = self._tick("callback_on_step")
            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)
            self._tock("callback_on_step", t)

            t = self._tick("info_buffer_update")
            self._update_info_buffer(infos, dones)
            self._tock("info_buffer_update", t)

            t = self._tick("store_transition_preprocess")
            if self._vec_normalize_env is not None:
                new_obs_ = self._vec_normalize_env.get_original_obs()
                reward_ = self._vec_normalize_env.get_original_reward()
            else:
                self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, rewards
            next_obs = deepcopy(new_obs_)
            for i, done in enumerate(dones):
                if done and infos[i].get("terminal_observation") is not None:
                    if isinstance(next_obs, dict):
                        next_obs_ = infos[i]["terminal_observation"]
                        if self._vec_normalize_env is not None:
                            next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                        for key in next_obs.keys():
                            next_obs[key][i] = next_obs_[key]
                    else:
                        next_obs[i] = infos[i]["terminal_observation"]
                        if self._vec_normalize_env is not None:
                            next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])
            self._tock("store_transition_preprocess", t)

            t = self._tick("buffer_add")
            replay_buffer.add(
                self._last_original_obs, next_obs,
                buffer_actions, reward_, dones, infos,
            )
            self._tock("buffer_add", t)

            t = self._tick("store_transition_postprocess")
            self._last_obs = new_obs
            if self._vec_normalize_env is not None:
                self._last_original_obs = new_obs_
            self._tock("store_transition_postprocess", t)

            t = self._tick("progress_update")
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._tock("progress_update", t)

            t = self._tick("dqn_on_step")
            self._on_step()
            self._tock("dqn_on_step", t)

            t = self._tick("episode_bookkeeping")
            for idx, done in enumerate(dones):
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1
                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        t_log = self._tick("dump_logs")
                        self.dump_logs()
                        self._tock("dump_logs", t_log)
            self._tock("episode_bookkeeping", t)

        t = self._tick("callback_rollout")
        callback.on_rollout_end()
        self._tock("callback_rollout", t)

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Time the full train() and buffer.sample() separately."""
        t_total = self._tick("train_total")
        original_sample = self.replay_buffer.sample
        sample_time_acc = 0.0

        def timed_sample(*args, **kwargs):
            nonlocal sample_time_acc
            t = time.perf_counter()
            result = original_sample(*args, **kwargs)
            sample_time_acc += time.perf_counter() - t
            return result

        self.replay_buffer.sample = timed_sample
        super().train(gradient_steps, batch_size)
        self.replay_buffer.sample = original_sample

        total_train = time.perf_counter() - t_total
        self.timers["buffer_sample"] += sample_time_acc
        self.call_counts["buffer_sample"] += gradient_steps
        self.timers["gradient_optimization"] += (total_train - sample_time_acc)
        self.call_counts["gradient_optimization"] += gradient_steps


# Category grouping
CATEGORY_MAP = {
    "gradient_optimization": "Gradient",
    "policy_inference": "Policy Inference",
    "buffer_sample": "Buffer (sample)",
    "buffer_add": "Buffer (add)",
    "env_step": "Environment",
    "store_transition_preprocess": "Data Preprocessing",
    "store_transition_postprocess": "Data Preprocessing",
    "dqn_on_step": "DQN Housekeeping",
    "callback_on_step": "Framework Overhead",
    "callback_rollout": "Framework Overhead",
    "info_buffer_update": "Framework Overhead",
    "episode_bookkeeping": "Framework Overhead",
    "progress_update": "Framework Overhead",
    "bookkeeping": "Framework Overhead",
    "train_setup": "Gradient",
    "dump_logs": "Framework Overhead",
    "python_loop_overhead": "Python Interpreter",
}

LABEL_MAP = {
    "gradient_optimization": "Gradient Optimization",
    "policy_inference": "Policy Inference",
    "buffer_sample": "Buffer Sample",
    "buffer_add": "Buffer Add",
    "env_step": "Environment Step",
    "store_transition_preprocess": "Data Preprocessing",
    "store_transition_postprocess": "Obs pointer update",
    "dqn_on_step": "DQN Housekeeping",
    "callback_on_step": "Callback Dispatch",
    "callback_rollout": "Callback (rollout)",
    "info_buffer_update": "Info Buffer Update",
    "episode_bookkeeping": "Episode Bookkeeping",
    "progress_update": "Progress Update",
    "bookkeeping": "Counter Increments",
    "train_setup": "Train Setup",
    "dump_logs": "Logger dump_logs",
    "python_loop_overhead": "Python Loop Overhead",
}


def run_one(label: str, replay_buffer_class, timesteps: int, buffer_size: int, seed: int):
    """Run one profiled training and return (wall_total, timers, counts)."""
    print(f"\n{'='*70}")
    print(f"  Profiling: {label}")
    print(f"{'='*70}")

    env = gym.make("CartPole-v1")
    model = GranularProfiledDQN(
        "MlpPolicy", env,
        replay_buffer_class=replay_buffer_class,
        buffer_size=buffer_size,
        learning_starts=1000,
        verbose=0,
        seed=seed,
    )

    wall_start = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    wall_total = time.perf_counter() - wall_start
    print(f"  Training finished in {wall_total:.4f}s")

    timers = dict(model.timers)
    counts = dict(model.call_counts)

    accounted = sum(timers.values())
    timers["python_loop_overhead"] = max(0.0, wall_total - accounted)

    return wall_total, timers, counts


def aggregate_categories(timers, wall_total):
    """Aggregate timers into categories. Returns sorted list of (cat, time, pct)."""
    cat_times = defaultdict(float)
    for key, t in timers.items():
        cat = CATEGORY_MAP.get(key, "Other")
        cat_times[cat] += t
    result = [(cat, t, t / wall_total * 100) for cat, t in cat_times.items()]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def print_comparison(label_a, wall_a, timers_a, label_b, wall_b, timers_b):
    """Print side-by-side comparison table."""
    cats_a = aggregate_categories(timers_a, wall_a)
    cats_b = aggregate_categories(timers_b, wall_b)

    # Build lookup dicts
    dict_a = {cat: (t, pct) for cat, t, pct in cats_a}
    dict_b = {cat: (t, pct) for cat, t, pct in cats_b}

    all_cats = list(dict.fromkeys([c[0] for c in cats_a] + [c[0] for c in cats_b]))

    print(f"\n{'='*90}")
    print(f"  Side-by-Side Time Decomposition ({label_a} vs {label_b})")
    print(f"{'='*90}")
    print(f"  {'Category':<25} | {label_a:>20} | {label_b:>20} | {'Diff':>10}")
    print(f"  {'-'*25}-+-{'-'*20}-+-{'-'*20}-+-{'-'*10}")

    for cat in all_cats:
        ta, pa = dict_a.get(cat, (0, 0))
        tb, pb = dict_b.get(cat, (0, 0))
        diff = tb - ta
        print(f"  {cat:<25} | {ta:>8.3f}s ({pa:>5.1f}%) | {tb:>8.3f}s ({pb:>5.1f}%) | {diff:>+8.3f}s")

    print(f"  {'-'*25}-+-{'-'*20}-+-{'-'*20}-+-{'-'*10}")
    diff_total = wall_b - wall_a
    print(f"  {'Total Wall Time':<25} | {wall_a:>8.3f}s (100.0%) | {wall_b:>8.3f}s (100.0%) | {diff_total:>+8.3f}s")
    print(f"{'='*90}")

    # Buffer-specific breakdown
    print(f"\n  Buffer Detail Breakdown:")
    for key in ["buffer_add", "buffer_sample"]:
        lbl = LABEL_MAP.get(key, key)
        va = timers_a.get(key, 0)
        vb = timers_b.get(key, 0)
        pa = va / wall_a * 100
        pb = vb / wall_b * 100
        diff = vb - va
        print(f"    {lbl:<23} | {va:>8.3f}s ({pa:>5.1f}%) | {vb:>8.3f}s ({pb:>5.1f}%) | {diff:>+8.3f}s")

    buf_a = timers_a.get("buffer_add", 0) + timers_a.get("buffer_sample", 0)
    buf_b = timers_b.get("buffer_add", 0) + timers_b.get("buffer_sample", 0)
    pa_tot = buf_a / wall_a * 100
    pb_tot = buf_b / wall_b * 100
    print(f"    {'Buffer Total':<23} | {buf_a:>8.3f}s ({pa_tot:>5.1f}%) | {buf_b:>8.3f}s ({pb_tot:>5.1f}%) | {buf_b - buf_a:>+8.3f}s")

    print(f"\n  Amdahl's Law:")
    print(f"    SB3 Default buffer fraction: {pa_tot:.1f}% → max speedup ≈ {1/(1-pa_tot/100):.1f}%")
    print(f"    FastReplay buffer fraction:  {pb_tot:.1f}% → overhead is {(wall_b - wall_a)/wall_a*100:+.1f}% of total\n")


def avg_timers(timer_list: list[dict[str, float]]) -> dict[str, float]:
    """Average multiple timer dicts element-wise."""
    all_keys = set()
    for t in timer_list:
        all_keys.update(t.keys())
    result = {}
    for k in all_keys:
        vals = [t.get(k, 0.0) for t in timer_list]
        result[k] = sum(vals) / len(vals)
    return result


def main():
    parser = argparse.ArgumentParser(description="A/B Time Decomposition: SB3 vs FastReplay")
    parser.add_argument("--steps", type=int, default=10000, help="Training timesteps")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per buffer type")
    args = parser.parse_args()

    NUM_TRIALS = args.trials

    # Interleaved ABAB order to reduce systematic bias
    sb3_results = []  # list of (wall, timers, counts)
    fr_results = []

    print(f"\n{'#'*70}")
    print(f"  Running {NUM_TRIALS} trials in interleaved ABAB order")
    print(f"  Steps per trial: {args.steps}, Buffer size: {args.buffer_size}")
    print(f"{'#'*70}")

    for trial in range(NUM_TRIALS):
        print(f"\n{'─'*70}")
        print(f"  Trial {trial+1}/{NUM_TRIALS}")
        print(f"{'─'*70}")

        # A: SB3 Default
        wall_a, timers_a, counts_a = run_one(
            f"SB3 ReplayBuffer (trial {trial+1})", ReplayBuffer,
            args.steps, args.buffer_size, args.seed)
        sb3_results.append((wall_a, timers_a, counts_a))

        # B: FastReplayBuffer
        wall_b, timers_b, counts_b = run_one(
            f"FastReplayBuffer (trial {trial+1})", FastReplayBuffer,
            args.steps, args.buffer_size, args.seed)
        fr_results.append((wall_b, timers_b, counts_b))

    # Per-trial summary
    print(f"\n{'='*70}")
    print(f"  Per-Trial Wall Times")
    print(f"{'='*70}")
    print(f"  {'Trial':>7} | {'SB3 Default':>14} | {'FastReplay':>14} | {'Diff':>10}")
    print(f"  {'-'*7}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")
    for i in range(NUM_TRIALS):
        wa = sb3_results[i][0]
        wb = fr_results[i][0]
        print(f"  {i+1:>7} | {wa:>11.3f}s | {wb:>11.3f}s | {wb-wa:>+8.3f}s")

    # Compute averages
    avg_wall_a = sum(r[0] for r in sb3_results) / NUM_TRIALS
    avg_wall_b = sum(r[0] for r in fr_results) / NUM_TRIALS
    avg_timers_a = avg_timers([r[1] for r in sb3_results])
    avg_timers_b = avg_timers([r[1] for r in fr_results])

    print(f"  {'-'*7}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")
    print(f"  {'Avg':>7} | {avg_wall_a:>11.3f}s | {avg_wall_b:>11.3f}s | {avg_wall_b-avg_wall_a:>+8.3f}s")
    print(f"  Overhead: {(avg_wall_b - avg_wall_a)/avg_wall_a*100:+.1f}%")

    # Print averaged comparison
    print_comparison("SB3 Default (avg)", avg_wall_a, avg_timers_a,
                     "FastReplay (avg)", avg_wall_b, avg_timers_b)

    # Save raw data
    raw = {
        "timesteps": args.steps,
        "num_trials": NUM_TRIALS,
        "per_trial": {
            "sb3_walls": [r[0] for r in sb3_results],
            "fr_walls": [r[0] for r in fr_results],
        },
        "averaged": {
            "sb3": {"wall_total_s": avg_wall_a, "timers": avg_timers_a,
                    "categories": {c: {"time_s": t, "pct": p} for c, t, p in aggregate_categories(avg_timers_a, avg_wall_a)}},
            "fastreplay": {"wall_total_s": avg_wall_b, "timers": avg_timers_b,
                           "categories": {c: {"time_s": t, "pct": p} for c, t, p in aggregate_categories(avg_timers_b, avg_wall_b)}},
        },
    }
    out_path = Path(__file__).resolve().parents[1] / "ab_comparison.json"
    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\n  Raw data saved to: {out_path}")


if __name__ == "__main__":
    main()

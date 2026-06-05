                        #!/usr/bin/env python3
"""Demo: Train DQN on CartPole-v1 using FastReplayBuffer as a drop-in replacement.

This script demonstrates that FastReplayBuffer is a fully functional
SB3 replay buffer.  It trains two identical DQN agents — one with the
default SB3 ``ReplayBuffer`` and one with ``FastReplayBuffer`` — and
compares their final evaluation rewards to confirm equivalence.

Usage
-----
::

    conda activate FastReplay
    python examples/sb3_dqn_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path so fastreplay_sb3 can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from fastreplay_sb3 import FastReplayBuffer


def train_and_evaluate(
    replay_buffer_class,
    label: str,
    seed: int = 42,
    total_timesteps: int = 50_000,
) -> tuple[float, float, float]:
    """Train DQN and return (mean_reward, std_reward, wall_time)."""
    print(f"\n{'='*60}")
    print(f"  Training DQN with {label}")
    print(f"{'='*60}")

    t0 = time.perf_counter()

    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        replay_buffer_class=replay_buffer_class,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=250,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)

    wall_time = time.perf_counter() - t0

    # Evaluate over 20 episodes
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=20)

    print(f"  Wall time  : {wall_time:.2f}s")
    print(f"  Reward     : {mean_reward:.1f} ± {std_reward:.1f}")

    # If using FastReplayBuffer, show C++ ring state
    if hasattr(model.replay_buffer, "ring_size"):
        print(f"  C++ ring   : {model.replay_buffer.ring_size} / {model.replay_buffer.ring_capacity}")

    return mean_reward, std_reward, wall_time


def main():
    print("FastReplay SB3 Integration Demo")
    print("=" * 60)

    # --- Run both ---
    fr_reward, fr_std, fr_time = train_and_evaluate(
        FastReplayBuffer, "FastReplayBuffer (C++ index)", seed=42
    )
    sb3_reward, sb3_std, sb3_time = train_and_evaluate(
        None, "SB3 default ReplayBuffer", seed=42  # None = SB3 default
    )

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'Buffer':<30s} {'Reward':>10s} {'Time':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'FastReplayBuffer (C++)':<30s} {fr_reward:>7.1f}±{fr_std:<4.1f} {fr_time:>8.2f}s")
    print(f"  {'SB3 ReplayBuffer (default)':<30s} {sb3_reward:>7.1f}±{sb3_std:<4.1f} {sb3_time:>8.2f}s")
    print()

    # Both should achieve approximately solved performance (reward ~300+)
    if fr_reward >= 200 and sb3_reward >= 200:
        print("  ✅ Both agents trained successfully. FastReplayBuffer is a valid drop-in.")
    else:
        print("  ⚠️  Training may need more timesteps. Check reward curves.")

    print()


if __name__ == "__main__":
    main()

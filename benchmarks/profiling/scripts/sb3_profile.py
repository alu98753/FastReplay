"""DQN training loop profiling script for Issue #31.

This script implements wrapper classes to decompose reinforcement learning (RL) 
training loop times into specific components: environment step, replay buffer 
addition, replay buffer sampling, and gradient optimization (train).
"""

import time
import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer


class TimeProfiledEnv(gym.Wrapper):
    """Gymnasium environment wrapper to accumulate execution time of environment step."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.total_step_time = 0.0

    def step(self, action):
        start = time.perf_counter()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_step_time += (time.perf_counter() - start)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class TimeProfiledReplayBuffer(ReplayBuffer):
    """ReplayBuffer subclass to evaluate the time spent in memory storage and retrieval operations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_add_time = 0.0
        self.total_sample_time = 0.0

    def add(self, *args, **kwargs):
        start = time.perf_counter()
        super().add(*args, **kwargs)
        self.total_add_time += (time.perf_counter() - start)

    def sample(self, *args, **kwargs):
        start = time.perf_counter()
        batch = super().sample(*args, **kwargs)
        self.total_sample_time += (time.perf_counter() - start)
        return batch


class TimeProfiledDQN(DQN):
    """DQN subclass to isolate and measure the duration of the gradient optimization phase."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_train_time = 0.0

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        start = time.perf_counter()
        super().train(gradient_steps, batch_size)
        self.total_train_time += (time.perf_counter() - start)


def run_profiling(timesteps: int = 10000, buffer_size: int = 50000):
    """Execute training and output component-level time decomposition statistics."""
    print(f"Initializing CartPole-v1 environment and TimeProfiledDQN...")
    raw_env = gym.make("CartPole-v1")
    profiled_env = TimeProfiledEnv(raw_env)

    model = TimeProfiledDQN(
        "MlpPolicy",
        profiled_env,
        buffer_size=buffer_size,
        learning_starts=1000,
        replay_buffer_class=TimeProfiledReplayBuffer,
        verbose=0,
        seed=42,
    )

    print(f"Starting training for {timesteps} steps...")
    start_wall_time = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    total_wall_time = time.perf_counter() - start_wall_time
    print("Training finished.")

    # Retrieve accumulated time variables
    env_step_time = profiled_env.total_step_time
    buffer_add_time = model.replay_buffer.total_add_time
    buffer_sample_time = model.replay_buffer.total_sample_time
    
    # Net train time excludes sampling time since sample() is called inside train()
    total_train_time = model.total_train_time
    net_grad_time = max(0.0, total_train_time - buffer_sample_time)
    
    # Calculate residual time for wrappers, policies, prediction, and other overhead
    other_time = max(0.0, total_wall_time - (env_step_time + buffer_add_time + buffer_sample_time + net_grad_time))

    print("\n" + "=" * 55)
    print(" " * 12 + "DQN Training Loop Time Decomposition")
    print("=" * 55)
    print(f"{'Component':<25} | {'Accumulated Time (s)':<20} | {'Ratio (%)':<10}")
    print("-" * 55)
    print(f"{'Environment Step':<25} | {env_step_time:<20.4f} | {env_step_time / total_wall_time * 100:<10.2f}")
    print(f"{'Buffer Add (Write)':<25} | {buffer_add_time:<20.4f} | {buffer_add_time / total_wall_time * 100:<10.2f}")
    print(f"{'Buffer Sample (Read)':<25} | {buffer_sample_time:<20.4f} | {buffer_sample_time / total_wall_time * 100:<10.2f}")
    print(f"{'Gradient Optimization':<25} | {net_grad_time:<20.4f} | {net_grad_time / total_wall_time * 100:<10.2f}")
    print(f"{'Other / SB3 Overhead':<25} | {other_time:<20.4f} | {other_time / total_wall_time * 100:<10.2f}")
    print("-" * 55)
    print(f"{'Total Wall Time':<25} | {total_wall_time:<20.4f} | {100.0:<10.2f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile SB3 DQN training loop.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps to execute.")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size.")
    args = parser.parse_args()
    run_profiling(timesteps=args.steps, buffer_size=args.buffer_size)

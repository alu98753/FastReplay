"""Test suite to validate correctness of the SB3 profiling implementation."""

import sys
from pathlib import Path
import pytest
import gymnasium as gym

# Setup Python path to import sb3_profile script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "profiling" / "scripts"))

import sb3_profile


def test_time_profiled_env():
    """Verify that TimeProfiledEnv accumulates environment step duration correctly."""
    raw_env = gym.make("CartPole-v1")
    env = sb3_profile.TimeProfiledEnv(raw_env)
    
    assert env.total_step_time == 0.0
    
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
    assert env.total_step_time > 0.0
    assert isinstance(env.total_step_time, float)


def test_sb3_profile_execution():
    """Verify that the full profiling sequence executes successfully without errors."""
    # Run a short training sequence to validate integration correctness
    sb3_profile.run_profiling(timesteps=500, buffer_size=1000)


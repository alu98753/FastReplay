"""Test suite for FastReplayBuffer SB3 compatibility.

Validates that FastReplayBuffer can be used as a drop-in replacement
for SB3's ReplayBuffer by verifying:
  1. Construction with SB3-style parameters
  2. add() / sample() round-trip correctness
  3. C++ ring state consistency
  4. Integration with DQN training (smoke test)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import gymnasium as gym

# Ensure fastreplay_sb3 is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastreplay_sb3 import FastReplayBuffer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cartpole_spaces():
    """Return (observation_space, action_space) for CartPole-v1."""
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    act_space = env.action_space
    env.close()
    return obs_space, act_space


@pytest.fixture
def buffer(cartpole_spaces):
    """Create a small FastReplayBuffer for CartPole-v1."""
    obs_space, act_space = cartpole_spaces
    return FastReplayBuffer(
        buffer_size=100,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_construction(cartpole_spaces):
    """FastReplayBuffer can be constructed with SB3 constructor signature."""
    obs_space, act_space = cartpole_spaces
    buf = FastReplayBuffer(
        buffer_size=1000,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        n_envs=1,
        optimize_memory_usage=False,
    )
    assert buf.buffer_size == 1000
    assert buf.ring_capacity == 1000
    assert buf.ring_size == 0
    assert buf.pos == 0
    assert not buf.full


def test_add_single(buffer):
    """A single add() stores data and advances pos."""
    obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    next_obs = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)
    action = np.array([1])
    reward = np.array([1.0])
    done = np.array([False])
    infos = [{}]

    buffer.add(obs, next_obs, action, reward, done, infos)

    assert buffer.pos == 1
    assert buffer.ring_size == 1
    np.testing.assert_array_almost_equal(buffer.observations[0, 0], obs)


def test_add_multiple_and_sample(buffer):
    """Add several transitions, then sample a batch."""
    n_transitions = 50
    for i in range(n_transitions):
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        action = np.array([np.random.randint(2)])
        reward = np.array([float(i)])
        done = np.array([False])
        infos = [{}]
        buffer.add(obs, next_obs, action, reward, done, infos)

    assert buffer.pos == 50
    assert buffer.ring_size == 50

    # Sample a batch
    samples = buffer.sample(batch_size=16)
    assert samples.observations.shape == (16, 4)
    assert samples.actions.shape == (16, 1)
    assert samples.rewards.shape == (16, 1)


def test_wrap_around(buffer):
    """Buffer wraps correctly when exceeding capacity."""
    for i in range(150):  # capacity=100, so wraps at 100
        obs = np.full(4, float(i), dtype=np.float32)
        next_obs = np.full(4, float(i + 1), dtype=np.float32)
        action = np.array([i % 2])
        reward = np.array([float(i)])
        done = np.array([i == 149])
        infos = [{}]
        buffer.add(obs, next_obs, action, reward, done, infos)

    assert buffer.full is True
    assert buffer.pos == 50  # 150 % 100 = 50
    assert buffer.ring_size == buffer.ring_capacity

    # Sample should still work correctly
    samples = buffer.sample(batch_size=32)
    assert samples.observations.shape == (32, 4)


def test_reset(buffer):
    """reset() clears both Python state and C++ ring."""
    for i in range(30):
        obs = np.random.randn(4).astype(np.float32)
        buffer.add(obs, obs, np.array([0]), np.array([0.0]),
                   np.array([False]), [{}])

    assert buffer.pos == 30
    assert buffer.ring_size == 30

    buffer.reset()

    assert buffer.pos == 0
    assert not buffer.full
    assert buffer.ring_size == 0


def test_dqn_smoke():
    """Smoke test: DQN can train with FastReplayBuffer without crashing."""
    from stable_baselines3 import DQN

    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        replay_buffer_class=FastReplayBuffer,
        learning_starts=100,
        buffer_size=500,
        batch_size=32,
        train_freq=4,
        verbose=0,
        seed=42,
    )
    # Train for a very short time — just verify it doesn't crash
    model.learn(total_timesteps=500)

    assert model.replay_buffer.ring_size > 0
    assert isinstance(model.replay_buffer, FastReplayBuffer)

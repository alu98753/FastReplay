"""FastReplayBuffer: SB3-compatible replay buffer backed by C++ atomic index management.

This module provides a drop-in replacement for SB3's ``ReplayBuffer``
that delegates circular-index bookkeeping (pos / full / wrap-around)
to FastReplay's C++ SPSC ring buffer.  Data storage remains in
pre-allocated NumPy arrays (identical to SB3), because the bottleneck
is NumPy fancy-indexing during ``sample()``, **not** index arithmetic.

The C++ RingBuffer maintains a parallel index stream: every ``add()``
pushes the current write position into the ring.  This means the C++
side always knows the valid index range and could, in a future
concurrent architecture, allow a separate learner thread to safely
call ``sample()`` while the actor thread calls ``add()`` — without
any Python-level lock.

Usage
-----
.. code-block:: python

    from stable_baselines3 import DQN
    from fastreplay_sb3 import FastReplayBuffer

    model = DQN("MlpPolicy", "CartPole-v1",
                replay_buffer_class=FastReplayBuffer,
                verbose=1)
    model.learn(total_timesteps=100_000)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

import fastreplay  # C++ pybind11 module


class FastReplayBuffer(ReplayBuffer):
    """SB3 ``ReplayBuffer`` with C++ atomic index tracking.

    Inherits **all** SB3 ``ReplayBuffer`` behaviour (data storage,
    ``sample()``, ``_get_samples()``, ``to_torch()``).  The override
    in ``add()`` pushes the write index into the C++ ``RingBuffer``
    so that the C++ side maintains a parallel record of valid indices.

    In the current single-threaded SB3 usage this does not improve
    performance — it demonstrates integration and provides the
    foundation for a future concurrent actor/learner architecture.

    Parameters
    ----------
    Identical to ``stable_baselines3.common.buffers.ReplayBuffer``.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        # Initialise the parent (allocates NumPy arrays, sets self.pos = 0)
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        # --- FastReplay addition: C++ index tracker -----------------------
        # Maintains a parallel stream of write indices in C++ with atomic
        # guarantees.  The ring capacity matches the buffer so that when
        # the buffer wraps, the ring also wraps and the oldest index is
        # evicted.  We store the write position (self.pos) as the element.
        self._index_ring = fastreplay.RingBuffer(self.buffer_size)

    # ------------------------------------------------------------------
    # Override: add() — push index into C++ ring after writing data
    # ------------------------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Store one transition.  Index is tracked by C++ RingBuffer."""

        # --- C++ index tracking: record the current write position ------
        # When the ring is full, pop the oldest index first (mirrors the
        # circular overwrite semantics of SB3).
        if self._index_ring.size() == self._index_ring.capacity():
            self._index_ring.pop()
        self._index_ring.push(self.pos)

        # --- Data storage + Python pos advance: delegate to parent ------
        super().add(obs, next_obs, action, reward, done, infos)

    def reset(self) -> None:
        """Reset buffer and C++ index tracker."""
        super().reset()
        self._index_ring = fastreplay.RingBuffer(self.buffer_size)

    # ------------------------------------------------------------------
    # Accessors for the C++ ring state (useful for testing / debugging)
    # ------------------------------------------------------------------
    @property
    def ring_size(self) -> int:
        """Number of valid indices tracked by the C++ ring."""
        return self._index_ring.size()

    @property
    def ring_capacity(self) -> int:
        """Capacity of the C++ index ring."""
        return self._index_ring.capacity()

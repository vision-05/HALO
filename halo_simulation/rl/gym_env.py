"""Gymnasium wrapper around :class:`TemperatureRlDriver` (optional dependency)."""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat, cast

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from halo_simulation.rl.driver import ACTION_DELTAS, TemperatureRlDriver

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover
    raise ImportError("Install gymnasium to use HaloTemperatureRlEnv: pip install gymnasium") from e


class HaloTemperatureRlEnv(gym.Env):
    """One-day temperature-conflict scenario; discrete nudges to thermostat comfort setpoint."""

    metadata = {"render_modes": []}

    def __init__(self, step_minutes: float = 15.0, seed: int = 0) -> None:
        super().__init__()
        self._driver = TemperatureRlDriver(step_minutes=step_minutes)
        self._base_seed = int(seed)
        self.action_space = spaces.Discrete(len(ACTION_DELTAS))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        ep_seed = int(seed if seed is not None else self._base_seed)
        if self.np_random is not None:
            ep_seed = int(ep_seed) ^ int(self.np_random.integers(0, 2**31 - 1))
        obs = self._driver.reset(ep_seed)
        return cast(ObsType, obs), {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._driver.step(int(action))
        return cast(ObsType, obs), reward, terminated, truncated, info

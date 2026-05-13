"""Headless RL driver: macro-steps through ``TemperatureConflictScenario``."""

from __future__ import annotations

from typing import Any

import numpy as np

from halo_simulation import config
from halo_simulation.agents.device_agent import ThermostatDeviceAgent
from halo_simulation.rl.observation import build_temperature_rl_observation
from halo_simulation.scenarios.temperature_conflict import TemperatureConflictScenario

# Discrete RL actions: nudge comfort setpoint (°C) before next SimPy slice.
ACTION_DELTAS: tuple[float, float, float] = (-0.5, 0.0, 0.5)


class TemperatureRlDriver:
    """Builds one-day temperature-conflict runs and advances them in fixed minute chunks."""

    def __init__(self, step_minutes: float = 15.0) -> None:
        self.step_minutes = float(step_minutes)
        self.scenario: TemperatureConflictScenario | None = None
        self.episode_seed: int = 0

    def _thermostat(self) -> ThermostatDeviceAgent:
        if self.scenario is None:
            raise RuntimeError("TemperatureRlDriver: call reset() first")
        for agent in self.scenario.agents:
            if getattr(agent, "agent_id", None) == "device_thermostat":
                return agent  # type: ignore[return-value]
        raise RuntimeError("TemperatureRlDriver: no device_thermostat in scenario")

    def reset(self, seed: int) -> np.ndarray:
        self.episode_seed = int(seed)
        self.scenario = TemperatureConflictScenario(seed=self.episode_seed, days=1)
        self.scenario.build()
        self.scenario.register_all()
        self.scenario.start_processes()
        return self.observe()

    def observe(self) -> np.ndarray:
        assert self.scenario is not None
        return build_temperature_rl_observation(self.scenario)

    def _reward(self, th: ThermostatDeviceAgent) -> float:
        cur = float(th._state.get("current_temp", 18.0))
        prefs = getattr(th, "_preferences", {}) or {}
        if not prefs:
            return -float(abs(cur - 20.5))
        total_w = 0.0
        total = 0.0
        for p in prefs.values():
            if not p.get("is_home", True):
                continue
            w = float(p.get("comfort_weight", config.DEFAULT_COMFORT_WEIGHT))
            tgt_pref = float(p.get("temperature", cur))
            total += w * abs(cur - tgt_pref)
            total_w += w
        comfort = total / total_w if total_w > 0 else abs(cur - 20.5)
        carbon = float(th._last_carbon) / max(1.0, float(config.CARBON_HIGH_THRESHOLD))
        return float(-comfort - 0.05 * carbon)

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.scenario is None:
            raise RuntimeError("TemperatureRlDriver: call reset() first")
        th = self._thermostat()
        ai = int(np.clip(action_index, 0, len(ACTION_DELTAS) - 1))
        delta = ACTION_DELTAS[ai]
        apply_info = th.apply_rl_comfort_delta(delta)
        t_before = float(self.scenario.env.now)
        horizon = float(config.MINUTES_PER_DAY * self.scenario.days)
        until = min(t_before + self.step_minutes, horizon)
        self.scenario.env.run(until=until)
        obs = self.observe()
        reward = self._reward(th)
        terminated = bool(self.scenario.env.now >= horizon - 1e-9)
        info: dict[str, Any] = {"apply": apply_info, "sim_time": float(self.scenario.env.now)}
        return obs, reward, terminated, False, info

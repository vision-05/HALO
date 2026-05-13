"""Shared 9-dim observation for thermostat RL — training and live inference must match."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from halo_simulation import config


def build_temperature_rl_observation(scenario: Any) -> np.ndarray:
    """Same vector as :meth:`TemperatureRlDriver.observe` (keep in sync)."""
    thermo = None
    for agent in scenario.agents:
        if getattr(agent, "agent_id", None) == "device_thermostat":
            thermo = agent
            break
    if thermo is None:
        raise RuntimeError("build_temperature_rl_observation: no device_thermostat in scenario")

    s = float(scenario.env.now)
    mod = s % float(config.MINUTES_PER_DAY)
    ang = 2.0 * math.pi * mod / float(config.MINUTES_PER_DAY)
    tsin, tcos = math.sin(ang), math.cos(ang)
    cur = float(thermo._state.get("current_temp", 18.0))
    tgt = float(thermo._state.get("target_temp", 20.0))
    if thermo._last_outdoor is not None:
        out = float(thermo._last_outdoor)
    else:
        out = float(config.WEATHER_BASELINE_TEMP + config.WEATHER_WINTER_OFFSET)
    carb = float(thermo._last_carbon)
    neg = 1.0 if thermo._negotiation_in_progress else 0.0

    lo = float(config.THERMOSTAT_MIN)
    hi = float(config.THERMOSTAT_MAX)

    def norm_temp(t: float) -> float:
        return float(np.clip((t - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0))

    alice_h, bob_h = 0.0, 0.0
    for agent in scenario.agents:
        aid = getattr(agent, "agent_id", "")
        if aid == "person_alice":
            alice_h = 1.0 if getattr(agent, "is_home", True) else -1.0
        elif aid == "person_bob":
            bob_h = 1.0 if getattr(agent, "is_home", True) else -1.0

    return np.array(
        [
            float(tsin),
            float(tcos),
            float(norm_temp(cur)),
            float(norm_temp(tgt)),
            float(norm_temp(out)),
            float(np.clip(carb / 400.0, 0.0, 1.0) * 2.0 - 1.0),
            float(neg * 2.0 - 1.0),
            float(alice_h),
            float(bob_h),
        ],
        dtype=np.float32,
    )

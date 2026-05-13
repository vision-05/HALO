"""Smoke tests for SimPy RL driver (no gymnasium required)."""

from __future__ import annotations

import numpy as np

from halo_simulation.rl.driver import ACTION_DELTAS, TemperatureRlDriver
from halo_simulation.rl.observation import build_temperature_rl_observation


def test_driver_reset_returns_obs_vector():
    d = TemperatureRlDriver(step_minutes=60.0)
    obs = d.reset(seed=7)
    assert obs.shape == (9,)
    assert np.all(obs >= -1.01) and np.all(obs <= 1.01)


def test_driver_step_advances_time_and_terminates_one_day():
    d = TemperatureRlDriver(step_minutes=60.0)
    d.reset(seed=0)
    assert d.scenario is not None
    last_t = -1.0
    steps = 0
    terminated = False
    while not terminated and steps < 30:
        _, _, terminated, _, info = d.step(1)
        t = float(info["sim_time"])
        assert t > last_t
        last_t = t
        steps += 1
    assert terminated
    assert last_t >= 1440.0 - 1e-6


def test_action_deltas_length_matches_discrete_three():
    assert len(ACTION_DELTAS) == 3


def test_observe_matches_build_temperature_rl_observation():
    d = TemperatureRlDriver(step_minutes=60.0)
    d.reset(seed=11)
    assert d.scenario is not None
    o1 = d.observe()
    o2 = build_temperature_rl_observation(d.scenario)
    np.testing.assert_array_almost_equal(o1, o2, decimal=6)

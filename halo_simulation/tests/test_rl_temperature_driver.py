"""Smoke tests for SimPy RL driver (no gymnasium required)."""

from __future__ import annotations

import numpy as np

from halo_simulation.rl.driver import ACTION_DELTAS, TemperatureRlDriver
from halo_simulation.rl.live_inference import _sb3_load_path
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


def test_sb3_load_path_prefers_zip_when_stem_is_directory(tmp_path):
    """SB3 stem path must not be a directory (common if checkpoint was extracted next to .zip)."""
    stem_dir = tmp_path / "mymodel"
    stem_dir.mkdir()
    zip_path = tmp_path / "mymodel.zip"
    zip_path.write_bytes(b"dummy")
    assert _sb3_load_path(str(zip_path)) == str(zip_path.resolve())


def test_sb3_load_path_uses_stem_when_no_directory(tmp_path):
    zip_path = tmp_path / "solo.zip"
    zip_path.write_bytes(b"x")
    assert _sb3_load_path(str(zip_path)) == str(zip_path.with_suffix(""))

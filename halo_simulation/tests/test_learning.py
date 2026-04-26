"""Tests for preference learning."""

import numpy as np

from halo_simulation import config
from halo_simulation.learning.preference_model import PreferenceModel


def test_ema_updates():
    rng = np.random.default_rng(0)
    m = PreferenceModel("thermostat", rng)
    h = 10
    m.observe_temperature_preference(25.0, h)
    assert m.ema_at_hour(h) > config.BAYESIAN_PRIOR_MU


def test_bayesian_shifts_toward_observation():
    rng = np.random.default_rng(1)
    m = PreferenceModel("thermostat", rng)
    old_mu = m.bayesian_mu
    for _ in range(5):
        m.observe_temperature_preference(30.0, 12)
    assert m.bayesian_mu > old_mu


def test_routine_stable_flag():
    rng = np.random.default_rng(2)
    m = PreferenceModel("thermostat", rng)
    for d in range(10):
        m.record_day_schedule(8 * 60 + 30.0, 18 * 60.0)
    assert m.routine_stable


def test_routine_unstable_with_noise():
    rng = np.random.default_rng(3)
    m = PreferenceModel("thermostat", rng)
    for d in range(10):
        m.record_day_schedule(8 * 60 + 30.0 + d * 60.0, 18 * 60.0)
    assert not m.routine_stable

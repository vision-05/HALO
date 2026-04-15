"""Tests for weighted negotiation protocol helpers."""

import numpy as np
import pytest

from halo_simulation import config
from halo_simulation.negotiation import protocol


def test_weighted_average_two_agents():
    v = [22.0, 19.0]
    w = [0.8, 0.8]
    p = protocol.combined_proposal(v, w, 20.0, config.DEFAULT_DEVICE_WEIGHT, 200.0)
    assert 19.0 < p < 22.0


def test_weighted_average_three_agents():
    v = [22.0, 19.0, 21.0]
    w = [0.8, 0.8, 0.4]
    p = protocol.combined_proposal(v, w, 20.0, config.DEFAULT_DEVICE_WEIGHT, 200.0)
    assert config.THERMOSTAT_MIN <= p <= config.THERMOSTAT_MAX


def test_convergence_detected():
    assert protocol.converged([21.0, 21.0, 21.1], threshold=0.5)
    assert not protocol.converged([22.0, 19.0], threshold=0.5)


def test_fallback_after_max_iterations():
    assert protocol.iterations_exceeded(config.MAX_ITERATIONS)
    assert not protocol.iterations_exceeded(config.MAX_ITERATIONS - 1)


def test_unweighted_fallback():
    v = [22.0, 19.0]
    m = protocol.unweighted_average(v)
    assert abs(m - 20.5) < 1e-6


def test_implicit_accept_timeout():
    assert not protocol.implicit_accept_timeout_elapsed(0.0, 3.0, timeout=5.0)
    assert protocol.implicit_accept_timeout_elapsed(0.0, 6.0, timeout=5.0)


def test_high_carbon_boosts_device_side():
    low_c = protocol.combined_proposal([22.0, 19.0], [0.8, 0.8], 20.0, 0.4, 200.0)
    high_c = protocol.combined_proposal([22.0, 19.0], [0.8, 0.8], 20.0, 0.4, 300.0)
    assert high_c != low_c or protocol.carbon_band(300.0) == "high"

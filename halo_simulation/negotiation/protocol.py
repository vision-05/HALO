"""
Weighted preference convergence — pure logic used by device agents.
Numbers pulled from config at call sites; this module imports config for defaults in helpers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from halo_simulation import config


def presence_multiplier(is_home: bool) -> float:
    return 1.0 if is_home else 0.25


def effective_person_weight(
    comfort_weight: float,
    is_home: bool,
) -> float:
    return comfort_weight * presence_multiplier(is_home)


def carbon_band(intensity_gco2: float) -> str:
    if intensity_gco2 < 150:
        return "low"
    if intensity_gco2 <= 250:
        return "medium"
    return "high"


def device_energy_weight_multiplier(carbon_intensity: float) -> float:
    """When carbon is high, device-side energy weight increases (person weights scaled down)."""
    if carbon_band(carbon_intensity) == "high":
        return 1.0 + config.CARBON_WEIGHT_BOOST
    return 1.0


def weighted_proposal(
    values: Sequence[float],
    person_weights: Sequence[float],
    device_optimal: float,
    device_weight: float,
    carbon_intensity: float,
) -> float:
    """
    proposal = (sum(v_i * w_i) + device_optimal * device_w_eff) / (sum(w_i) + device_w_eff)
    where device_w_eff incorporates carbon high boost on the device side.
    """
    if len(values) != len(person_weights):
        raise ValueError("values and person_weights length mismatch")
    boost = device_energy_weight_multiplier(carbon_intensity)
    dev_w = device_weight * boost
    num = sum(v * w for v, w in zip(values, person_weights, strict=True)) + device_optimal * dev_w
    den = sum(person_weights) + dev_w
    if den <= 0:
        return float(np.mean(values)) if values else device_optimal
    raw = num / den
    return float(np.clip(raw, config.THERMOSTAT_MIN, config.THERMOSTAT_MAX))


def variance_of_values(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.var(arr))


def unweighted_average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def satisfaction_score(
    final_value: float,
    preference: float,
    preference_range: float | None = None,
) -> float:
    rng = preference_range if preference_range is not None else config.TEMPERATURE_PREFERENCE_RANGE
    if rng <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - abs(final_value - preference) / rng))


def combined_proposal(
    values: Sequence[float],
    person_weights: Sequence[float],
    device_optimal: float,
    device_weight: float,
    carbon_intensity: float,
) -> float:
    """Initial or iterated proposal using weighted average + device longevity pull + carbon-aware device weight."""
    return weighted_proposal(values, person_weights, device_optimal, device_weight, carbon_intensity)


def iterations_exceeded(iteration: int) -> bool:
    return iteration >= config.MAX_ITERATIONS


def converged(values: Sequence[float], threshold: float | None = None) -> bool:
    th = threshold if threshold is not None else config.CONVERGENCE_THRESHOLD
    return variance_of_values(values) < th


def credible_interval_90(mu: float, sigma: float) -> tuple[float, float]:
    z = 1.645  # 90% for normal
    return mu - z * sigma, mu + z * sigma


def implicit_accept_timeout_elapsed(
    start_time: float,
    now: float,
    timeout: float | None = None,
) -> bool:
    t = timeout if timeout is not None else config.NEGOTIATION_TIMEOUT
    return (now - start_time) >= t

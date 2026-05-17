"""
Weighted preference convergence — pure logic used by device agents.
Numbers pulled from config at call sites; this module imports config for defaults in helpers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from halo_simulation import config


def presence_multiplier(is_home: bool) -> float:
    return 1.0 if is_home else float(config.PRESENCE_MULTIPLIER_AWAY)


def effective_person_weight(
    comfort_weight: float,
    is_home: bool,
) -> float:
    """
    Stake in weighted thermostat/shower proposals: ``comfort_weight`` comes from each person's
    latest ``PreferenceDeclaration`` payload (default ``DEFAULT_COMFORT_WEIGHT``, reduced when
    away via ``AWAY_COMFORT_WEIGHT``, and may drop on energy-related ``ExternalDisruptionEvent``).
    Multiplied by ``presence_multiplier`` (1.0 home / 0.25 away). See ``PersonAgent._broadcast_preferences``.
    """
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
    *,
    clip_lo: float | None = None,
    clip_hi: float | None = None,
) -> float:
    """
    proposal = (sum(v_i * w_i) + device_optimal * device_w_eff) / (sum(w_i) + device_w_eff)
    where device_w_eff incorporates carbon high boost on the device side.
    """
    if len(values) != len(person_weights):
        raise ValueError("values and person_weights length mismatch")
    lo = float(config.THERMOSTAT_MIN) if clip_lo is None else float(clip_lo)
    hi = float(config.THERMOSTAT_MAX) if clip_hi is None else float(clip_hi)
    boost = device_energy_weight_multiplier(carbon_intensity)
    dev_w = device_weight * boost
    num = sum(v * w for v, w in zip(values, person_weights, strict=True)) + device_optimal * dev_w
    den = sum(person_weights) + dev_w
    if den <= 0:
        return float(np.mean(values)) if values else device_optimal
    raw = num / den
    return float(np.clip(raw, lo, hi))


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
    *,
    clip_lo: float | None = None,
    clip_hi: float | None = None,
) -> float:
    """Initial or iterated proposal using weighted average + device longevity pull + carbon-aware device weight."""
    return weighted_proposal(
        values,
        person_weights,
        device_optimal,
        device_weight,
        carbon_intensity,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
    )


def iterations_exceeded(iteration: int) -> bool:
    return iteration >= config.MAX_ITERATIONS


def converged(values: Sequence[float], threshold: float | None = None) -> bool:
    th = threshold if threshold is not None else config.CONVERGENCE_THRESHOLD
    return variance_of_values(values) < th


def shower_minutes_from_comfort_temp(temp: float) -> float:
    """
    Map declared comfort °C to a preferred shower duration (minutes).

    This is a modelling shortcut: the sim already has people broadcast thermal comfort (°C),
    but not a separate “shower length” field. A linear map from thermostat range to minute range
    gives each resident a distinct default stake in shower negotiation without extra messages.
    It is illustrative (not a claim about real physiology).
    """
    lo_t = float(config.THERMOSTAT_MIN)
    hi_t = float(config.THERMOSTAT_MAX)
    lo_m = float(config.SHOWER_DURATION_MIN_MINUTES)
    hi_m = float(config.SHOWER_DURATION_MAX_MINUTES)
    t = max(lo_t, min(hi_t, float(temp)))
    if hi_t <= lo_t:
        return (lo_m + hi_m) / 2.0
    frac = (t - lo_t) / (hi_t - lo_t)
    return lo_m + frac * (hi_m - lo_m)


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

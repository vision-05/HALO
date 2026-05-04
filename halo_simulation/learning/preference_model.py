"""EMA, Bayesian preference, and routine detection for person agents."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

import config


class PreferenceModel:
    """
    Per-device preference learning: EMA by hour, Bayesian temperature belief,
    and routine stability over a rolling window.
    """

    def __init__(
        self,
        device_type: str,
        rng: np.random.Generator,
        prior_mu: float | None = None,
        prior_sigma: float | None = None,
    ) -> None:
        self.device_type = device_type
        self._rng = rng
        mu = prior_mu if prior_mu is not None else config.BAYESIAN_PRIOR_MU
        sigma = prior_sigma if prior_sigma is not None else config.BAYESIAN_PRIOR_SIGMA
        self._bayesian_mu = mu
        self._bayesian_sigma_sq = sigma**2
        self._ema_by_hour: list[float] = [mu] * 24
        self._departures: deque[float] = deque(maxlen=config.ROUTINE_WINDOW_DAYS)
        self._arrivals: deque[float] = deque(maxlen=config.ROUTINE_WINDOW_DAYS)
        self._routine_stable = False

    @property
    def bayesian_mu(self) -> float:
        return self._bayesian_mu

    @property
    def bayesian_sigma(self) -> float:
        return math.sqrt(max(self._bayesian_sigma_sq, 1e-9))

    def tolerance_from_bayesian(self) -> float:
        low, high = self.credible_interval_90()
        return (high - low) / 2.0

    def credible_interval_90(self) -> tuple[float, float]:
        z = 1.645
        m = self._bayesian_mu
        s = math.sqrt(max(self._bayesian_sigma_sq, 1e-9))
        return m - z * s, m + z * s

    def ema_at_hour(self, hour: int) -> float:
        return self._ema_by_hour[hour % 24]

    def observe_temperature_preference(self, value: float, hour_of_day: int) -> None:
        h = hour_of_day % 24
        a = config.EMA_ALPHA
        self._ema_by_hour[h] = a * value + (1 - a) * self._ema_by_hour[h]
        self._bayesian_update(value)

    def _bayesian_update(self, observed: float) -> None:
        """Normal likelihood with known variance; conjugate update on mean."""
        sigma_obs = 1.0
        prior_var = self._bayesian_sigma_sq
        prior_mu = self._bayesian_mu
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / sigma_obs**2)
        posterior_mean = posterior_var * (prior_mu / prior_var + observed / sigma_obs**2)
        self._bayesian_mu = posterior_mean
        self._bayesian_sigma_sq = posterior_var

    def record_day_schedule(self, departure_minute: float, arrival_minute: float) -> bool:
        """
        Record today's departure/arrival (minutes from midnight).
        Returns True if this day should update the model (not an anomaly).
        """
        if len(self._departures) >= 2:
            mean_d = float(np.mean(self._departures))
            std_d = float(np.std(self._departures))
            if std_d < 1e-6:
                std_d = 1e-6
            if abs(departure_minute - mean_d) > config.ANOMALY_THRESHOLD_MULTIPLIER * std_d:
                return False
        self._departures.append(departure_minute)
        self._arrivals.append(arrival_minute)
        self._update_routine_stability()
        return True

    def _update_routine_stability(self) -> None:
        if len(self._departures) < 3:
            self._routine_stable = False
            return
        std_d = float(np.std(self._departures))
        std_a = float(np.std(self._arrivals))
        self._routine_stable = max(std_d, std_a) < config.ROUTINE_STABLE_STD_MINUTES

    @property
    def routine_stable(self) -> bool:
        return self._routine_stable

    def end_of_day_update(
        self,
        observed_preference: float,
        hour_of_day: int,
        departure_minute: float,
        arrival_minute: float,
    ) -> dict[str, Any]:
        """Apply daily learning; returns snapshot for metrics."""
        anomaly_ok = self.record_day_schedule(departure_minute, arrival_minute)
        if anomaly_ok:
            self.observe_temperature_preference(observed_preference, hour_of_day)
        return {
            "ema_value": self.ema_at_hour(hour_of_day),
            "bayesian_mu": self._bayesian_mu,
            "bayesian_sigma": self.bayesian_sigma,
            "routine_stable": self._routine_stable,
        }

"""Aggregation agents — external signals (grid carbon, weather)."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import simpy

from halo_simulation import config
from halo_simulation.agents.base_agent import BaseAgent
from halo_simulation.metrics.collector import MetricsCollector
from halo_simulation.negotiation import protocol
from halo_simulation.negotiation.message import Message, MessageTypes

logger = logging.getLogger(__name__)


class GridCarbonAgent(BaseAgent):
    """Broadcasts UK grid carbon intensity (gCO2/kWh) — live API or synthetic profile."""

    def __init__(
        self,
        agent_id: str,
        env: simpy.Environment,
        message_bus: Any,
        rng: np.random.Generator,
        metrics: MetricsCollector | None,
        force_evening_peak: bool = False,
        api_client: Any | None = None,
    ) -> None:
        super().__init__(agent_id, "specialist", env, message_bus, metrics)
        self._rng = rng
        self._force_evening_peak = force_evening_peak
        self._api_client = api_client
        self._last_value = float(config.CARBON_HOURLY_BASELINE[12])

    def _intensity_at(self, now: float) -> float:
        minute_of_day = now % config.MINUTES_PER_DAY
        hour = int(minute_of_day // 60) % 24
        base = float(config.CARBON_HOURLY_BASELINE[hour])
        noise = float(self._rng.normal(0, 30))
        val = base + noise
        if self._force_evening_peak:
            if config.CARBON_SPIKE_START_MINUTE <= minute_of_day <= config.CARBON_SPIKE_END_MINUTE:
                val = max(val, float(config.CARBON_SPIKE_INTENSITY))
        return float(max(50.0, val))

    def _forecast_synthetic(self, now: float) -> list[float]:
        out = []
        t = now
        for _ in range(4):
            t += 60.0
            out.append(self._intensity_at(t))
        return out

    def _forecast_from_api_slots(self, slots: list[Any], fallback: float) -> list[float]:
        vals: list[float] = []
        for slot in slots[:8]:
            if isinstance(slot, dict) and slot.get("value") is not None:
                vals.append(float(slot["value"]))
        if len(vals) >= 4:
            return [vals[i] for i in (0, 2, 4, 6)] if len(vals) >= 8 else vals[:4]
        while len(vals) < 4:
            vals.append(fallback)
        return vals[:4]

    def run(self):
        while True:
            try:
                now = self.env.now
                if self._api_client:
                    data = self._api_client.get_carbon_intensity(self.env.now)
                    current = float(data["value"])
                    minute_of_day = now % config.MINUTES_PER_DAY
                    if self._force_evening_peak:
                        if config.CARBON_SPIKE_START_MINUTE <= minute_of_day <= config.CARBON_SPIKE_END_MINUTE:
                            current = max(current, float(config.CARBON_SPIKE_INTENSITY))
                    forecast = self._forecast_from_api_slots(
                        data.get("forecast") or [],
                        current,
                    )
                    band = str(data["level"])
                    data_source = str(data.get("source", "synthetic"))
                else:
                    current = self._intensity_at(now)
                    forecast = self._forecast_synthetic(now)
                    band = protocol.carbon_band(current)
                    data_source = "synthetic"

                self._last_value = current
                m = Message.create(
                    self.agent_id,
                    "broadcast",
                    MessageTypes.CarbonIntensityUpdate,
                    {
                        "current": current,
                        "forecast_4h": forecast,
                        "band": band,
                        "source": data_source,
                    },
                    self.env.now,
                )
                self.broadcast(m)
            except Exception as exc:
                logger.exception("GridCarbonAgent degraded: %s", exc)
                self.broadcast(
                    Message.create(
                        self.agent_id,
                        "broadcast",
                        MessageTypes.SpecialistUnavailable,
                        {"source": "grid_carbon", "last_known": self._last_value},
                        self.env.now,
                    )
                )
            yield self.env.timeout(config.CARBON_BROADCAST_INTERVAL)


class WeatherAgent(BaseAgent):
    """Outdoor temperature with seasonal offset and rare heatwave / cold snap events — or live API."""

    def __init__(
        self,
        agent_id: str,
        env: simpy.Environment,
        message_bus: Any,
        rng: np.random.Generator,
        metrics: MetricsCollector | None,
        season_offset: float = 0.0,
        api_client: Any | None = None,
    ) -> None:
        super().__init__(agent_id, "specialist", env, message_bus, metrics)
        self._rng = rng
        self._season_offset = season_offset
        self._api_client = api_client
        self._last_temp = config.WEATHER_BASELINE_TEMP + season_offset
        self._hot_days = 0
        self._freezing_days = 0

    def _temp_at(self, now: float) -> float:
        minute_of_day = now % config.MINUTES_PER_DAY
        phase = 2 * math.pi * (minute_of_day / config.MINUTES_PER_DAY)
        daily = 6.0 * math.sin(phase - math.pi / 2)
        noise = float(self._rng.normal(0, 0.5))
        t = config.WEATHER_BASELINE_TEMP + self._season_offset + daily + noise
        return float(t)

    def _roll_daily_event(self, peak: float, low: float) -> None:
        if peak > 32.0:
            self._hot_days += 1
        else:
            self._hot_days = 0
        if low < 0.0:
            self._freezing_days += 1
        else:
            self._freezing_days = 0

    def _event_label(self) -> str | None:
        if self._hot_days >= 3:
            return "heatwave"
        if self._freezing_days >= 3:
            return "cold_snap"
        return None

    def run(self):
        if self._api_client:
            while True:
                try:
                    w = self._api_client.get_weather(self.env.now)
                    temp = float(w["temperature"])
                    self._last_temp = temp
                    event = (
                        "heatwave"
                        if w["is_heatwave"]
                        else ("cold_snap" if w["is_cold_snap"] else None)
                    )
                    self.broadcast(
                        Message.create(
                            self.agent_id,
                            "broadcast",
                            MessageTypes.WeatherUpdate,
                            {
                                "outdoor_temp_c": temp,
                                "apparent_temp_c": w["feels_like"],
                                "condition": w["condition"],
                                "wind_speed_10m": w["wind_speed"],
                                "day_index": int(self.env.now // config.MINUTES_PER_DAY),
                                "special_event": event,
                                "source": str(w.get("source", "synthetic")),
                                "is_heatwave": bool(w["is_heatwave"]),
                                "is_cold_snap": bool(w["is_cold_snap"]),
                            },
                            self.env.now,
                        )
                    )
                except Exception as exc:
                    logger.exception("WeatherAgent degraded: %s", exc)
                    self.broadcast(
                        Message.create(
                            self.agent_id,
                            "broadcast",
                            MessageTypes.SpecialistUnavailable,
                            {"source": "weather", "last_known": self._last_temp},
                            self.env.now,
                        )
                    )
                yield self.env.timeout(config.WEATHER_BROADCAST_INTERVAL)

        last_day = -1
        day_peak = 0.0
        day_min = 0.0
        while True:
            try:
                now = self.env.now
                day = int(now // config.MINUTES_PER_DAY)
                temp = self._temp_at(now)
                if day != last_day:
                    if last_day >= 0:
                        self._roll_daily_event(day_peak, day_min)
                    last_day = day
                    day_peak = temp
                    day_min = temp
                else:
                    day_peak = max(day_peak, temp)
                    day_min = min(day_min, temp)
                self._last_temp = temp
                event = self._event_label()
                self.broadcast(
                    Message.create(
                        self.agent_id,
                        "broadcast",
                        MessageTypes.WeatherUpdate,
                        {
                            "outdoor_temp_c": temp,
                            "day_index": int(now // config.MINUTES_PER_DAY),
                            "special_event": event,
                            "source": "synthetic",
                        },
                        self.env.now,
                    )
                )
            except Exception as exc:
                logger.exception("WeatherAgent degraded: %s", exc)
                self.broadcast(
                    Message.create(
                        self.agent_id,
                        "broadcast",
                        MessageTypes.SpecialistUnavailable,
                        {"source": "weather", "last_known": self._last_temp},
                        self.env.now,
                    )
                )
            yield self.env.timeout(config.WEATHER_BROADCAST_INTERVAL)

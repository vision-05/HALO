"""
Real-time external data client for HALO simulation.
Connects to:
- National Grid ESO carbon intensity API (api.carbonintensity.org.uk)
- Open-Meteo weather API (api.open-meteo.com)

Both APIs are free and require no API key.
All calls are synchronous (using httpx in sync mode) because they are
called from within SimPy's simulation thread, not an async context.

Simulation time: when `sim_minute` is passed (SimPy `env.now`), readings are
taken from API timelines anchored at client creation so multi-day runs see
changing weather and grid intensity instead of a single cached snapshot.

When `sim_minute` is omitted, callers get *wall-clock* "now" (for /api/status).

Every call has a 5-second timeout and falls back to cached/synthetic data on
any failure. The simulation never crashes due to API issues.
"""

from __future__ import annotations

import bisect
import logging
import math
import random
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx

from halo_simulation import config

logger = logging.getLogger(__name__)

CARBON_BASE_URL = "https://api.carbonintensity.org.uk"
WEATHER_BASE_URL = "https://api.open-meteo.com"
LONDON_LAT = 51.5074
LONDON_LON = -0.1278
LONDON_TZ = ZoneInfo("Europe/London")
REQUEST_TIMEOUT = 5.0  # seconds
# Open-Meteo free forecast horizon (days of hourly data)
WEATHER_FORECAST_DAYS = 16


def map_wmo_weather_code(code: int) -> str:
    """WMO weather code → HALO condition string (shared by client + chart API)."""
    if code == 0:
        return "clear"
    if code in (1, 2, 3):
        return "cloudy"
    if code in (45, 48):
        return "cloudy"
    if 51 <= code <= 67:
        return "rain"
    if 71 <= code <= 77:
        return "snow"
    if 80 <= code <= 82:
        return "rain"
    if 95 <= code <= 99:
        return "storm"
    return "cloudy"


def open_meteo_hour_start_unix_ms(iso_local: str) -> int:
    """Open-Meteo local time string (Europe/London) → Unix time in milliseconds (UTC)."""
    t = str(iso_local).replace("Z", "")
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LONDON_TZ)
    else:
        dt = dt.astimezone(LONDON_TZ)
    return int(dt.timestamp() * 1000)


def fetch_weather_hourly_chart_data(
    lat: float = LONDON_LAT,
    lon: float = LONDON_LON,
    forecast_days: int = WEATHER_FORECAST_DAYS,
) -> dict[str, Any]:
    """
    One-shot Open-Meteo hourly pull for the full forecast window (London default).
    Used by GET /api/weather_series and the dashboard chart.
    """
    fd = max(1, min(int(forecast_days), 16))
    client = httpx.Client(
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "HALO-Simulation/1.0"},
    )
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "timezone": "Europe/London",
            "forecast_days": fd,
        }
        response = client.get(f"{WEATHER_BASE_URL}/v1/forecast", params=params)
        response.raise_for_status()
        data = response.json()
        hourly = data.get("hourly") or {}
        times: list[str] = [str(t) for t in (hourly.get("time") or [])]
        temps = hourly.get("temperature_2m") or []
        feels = hourly.get("apparent_temperature") or []
        codes = hourly.get("weather_code") or []
        winds = hourly.get("wind_speed_10m") or []
        n = len(times)
        temperature_2m: list[float | None] = []
        apparent_temperature: list[float | None] = []
        weather_code: list[int] = []
        wind_speed_10m: list[float | None] = []
        condition: list[str] = []
        time_ms: list[int] = []
        for i in range(n):
            temperature_2m.append(float(temps[i]) if i < len(temps) and temps[i] is not None else None)
            apparent_temperature.append(float(feels[i]) if i < len(feels) and feels[i] is not None else None)
            wc = int(codes[i]) if i < len(codes) and codes[i] is not None else 3
            weather_code.append(wc)
            wind_speed_10m.append(float(winds[i]) if i < len(winds) and winds[i] is not None else None)
            condition.append(map_wmo_weather_code(wc))
            try:
                time_ms.append(open_meteo_hour_start_unix_ms(times[i]))
            except (TypeError, ValueError, OSError):
                if time_ms:
                    time_ms.append(time_ms[-1] + 3600000)
                else:
                    time_ms.append(0)
        return {
            "latitude": lat,
            "longitude": lon,
            "timezone": "Europe/London",
            "forecast_days": fd,
            "hours": n,
            "time": times,
            "time_ms": time_ms,
            "temperature_2m": temperature_2m,
            "apparent_temperature": apparent_temperature,
            "weather_code": weather_code,
            "wind_speed_10m": wind_speed_10m,
            "condition": condition,
        }
    finally:
        client.close()


class ExternalDataClient:
    def __init__(self, lat: float = LONDON_LAT, lon: float = LONDON_LON) -> None:
        self.lat = lat
        self.lon = lon

        # Wall time when this client was created = simulation minute 0.
        self._sim_epoch_utc = datetime.now(timezone.utc)

        self._carbon_cache: Optional[dict[str, Any]] = None
        self._weather_cache: Optional[dict[str, Any]] = None
        self._carbon_forecast_cache: list[dict[str, Any]] = []

        self._carbon_last_fetched: float = 0.0
        self._weather_last_fetched: float = 0.0

        self._carbon_ttl: float = 1800.0  # 30 minutes (wall-clock "now" only)
        self._weather_ttl: float = 900.0  # 15 minutes

        self.api_status: dict[str, str] = {
            "carbon": "unknown",
            "weather": "unknown",
        }

        self._client = httpx.Client(
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "HALO-Simulation/1.0"},
        )

        # Sim-time weather: hourly series (loaded on first sim_minute request)
        self._weather_hour_starts: list[datetime] = []
        self._weather_hour_rows: list[dict[str, Any]] = []

        # Sim-time carbon: National Grid half-hour slots per London calendar day
        self._carbon_day_cache: dict[str, list[dict[str, Any]]] = {}

    @property
    def sim_epoch_utc(self) -> datetime:
        return self._sim_epoch_utc

    @staticmethod
    def _parse_carbon_time(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    @staticmethod
    def _parse_om_time(s: str) -> datetime:
        """Open-Meteo local time string -> Europe/London aware."""
        t = s.replace("Z", "")
        if len(t) <= 16:
            dt = datetime.fromisoformat(t)
        else:
            dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=LONDON_TZ)
        return dt.astimezone(LONDON_TZ)

    def get_carbon_intensity(self, sim_minute: float | None = None) -> dict[str, Any]:
        """
        Carbon intensity for wall-clock "now" (sim_minute is None) or for a
        simulated instant (anchor + sim_minute) mapped to UK grid half-hours.
        """
        if sim_minute is None:
            return self._get_carbon_wall_clock_now()
        return self._get_carbon_at_sim(float(sim_minute))

    def _get_carbon_wall_clock_now(self) -> dict[str, Any]:
        now = time.time()
        if self._carbon_cache and (now - self._carbon_last_fetched) < self._carbon_ttl:
            result = dict(self._carbon_cache)
            result["source"] = "cached"
            result["forecast"] = list(result.get("forecast") or [])
            self.api_status["carbon"] = "cached"
            return result

        try:
            response = self._client.get(f"{CARBON_BASE_URL}/intensity")
            response.raise_for_status()
            data = response.json()
            intensity_block = data["data"][0]["intensity"]
            raw_value = intensity_block.get("actual") or intensity_block.get("forecast")
            if raw_value is None:
                raise ValueError("No intensity value in response")
            raw_index = intensity_block.get("index", "moderate")
            level = self._map_carbon_index(str(raw_index))
            result = {
                "value": int(raw_value),
                "level": level,
                "source": "live",
                "forecast": self._fetch_carbon_forecast(),
            }
            self._carbon_cache = result
            self._carbon_last_fetched = now
            self.api_status["carbon"] = "live"
            logger.info("Carbon API: %s gCO2/kWh (%s)", raw_value, level)
            return result
        except Exception as e:
            logger.warning("Carbon API failed: %s. Using fallback.", e)
            self.api_status["carbon"] = "error" if not self._carbon_cache else "cached"
            return self._carbon_cache or self._synthetic_carbon(None)

    def _get_carbon_at_sim(self, sim_minute: float) -> dict[str, Any]:
        virtual = self._sim_epoch_utc + timedelta(minutes=sim_minute)
        if virtual.tzinfo is None:
            virtual = virtual.replace(tzinfo=timezone.utc)

        try:
            day_london = virtual.astimezone(LONDON_TZ).date()
            slots = self._fetch_carbon_day_slots(day_london)
            if not slots:
                raise ValueError("No carbon slots for day")

            idx = self._carbon_slot_index(virtual, slots)
            intensity = slots[idx].get("intensity", {})
            raw_value = intensity.get("actual") or intensity.get("forecast")
            if raw_value is None:
                raise ValueError("Empty intensity in slot")
            level = self._map_carbon_index(str(intensity.get("index", "moderate")))
            forecast = self._carbon_forecast_from_day_slot(day_london, idx)

            self.api_status["carbon"] = "live"
            return {
                "value": int(raw_value),
                "level": level,
                "source": "live",
                "forecast": forecast,
            }
        except Exception as e:
            logger.warning("Carbon at sim_minute=%s failed: %s. Using fallback.", sim_minute, e)
            self.api_status["carbon"] = "error"
            return self._synthetic_carbon(virtual)

    def _fetch_carbon_day_slots(self, day_london: date) -> list[dict[str, Any]]:
        key = day_london.isoformat()
        if key in self._carbon_day_cache:
            return self._carbon_day_cache[key]
        try:
            response = self._client.get(f"{CARBON_BASE_URL}/intensity/date/{key}")
            response.raise_for_status()
            data = response.json()
            slots = list(data.get("data") or [])
            self._carbon_day_cache[key] = slots
            return slots
        except Exception as e:
            logger.warning("Carbon day %s fetch failed: %s", key, e)
            self._carbon_day_cache[key] = []
            return []

    def _carbon_slot_index(self, virtual_utc: datetime, slots: list[dict[str, Any]]) -> int:
        for i, slot in enumerate(slots):
            t0 = self._parse_carbon_time(str(slot["from"]))
            t1 = self._parse_carbon_time(str(slot["to"]))
            if t0 <= virtual_utc < t1:
                return i
        # Fallback: nearest by midpoint distance
        best_i = 0
        best_d = float("inf")
        for i, slot in enumerate(slots):
            t0 = self._parse_carbon_time(str(slot["from"]))
            t1 = self._parse_carbon_time(str(slot["to"]))
            mid = t0 + (t1 - t0) / 2
            d = abs((virtual_utc - mid).total_seconds())
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _carbon_forecast_from_day_slot(self, day_london: date, idx: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cur_day = day_london
        slots = self._fetch_carbon_day_slots(cur_day)
        i = idx + 1
        while len(out) < 8:
            if i >= len(slots):
                cur_day = cur_day + timedelta(days=1)
                slots = self._fetch_carbon_day_slots(cur_day)
                i = 0
                if not slots:
                    break
                continue
            slot = slots[i]
            intensity = slot.get("intensity", {})
            value = intensity.get("forecast") or intensity.get("actual")
            if value is not None:
                out.append(
                    {
                        "from": slot.get("from", ""),
                        "to": slot.get("to", ""),
                        "value": int(value),
                        "level": self._map_carbon_index(str(intensity.get("index", "moderate"))),
                    }
                )
            i += 1
        return out

    def _fetch_carbon_forecast(self) -> list[dict[str, Any]]:
        """Wall-clock24h-style forecast (used for /intensity probe path)."""
        slots: list[dict[str, Any]] = []
        try:
            response = self._client.get(f"{CARBON_BASE_URL}/intensity/pt24h")
            response.raise_for_status()
            data = response.json()
            slots = data.get("data", [])
        except Exception as e:
            logger.warning("Carbon forecast pt24h failed: %s", e)
            try:
                d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                response = self._client.get(f"{CARBON_BASE_URL}/intensity/date/{d}")
                response.raise_for_status()
                data = response.json()
                slots = data.get("data", [])
            except Exception as e2:
                logger.warning("Carbon forecast date fallback failed: %s", e2)
                return list(self._carbon_forecast_cache)

        forecast: list[dict[str, Any]] = []
        for slot in slots[:8]:
            intensity = slot.get("intensity", {})
            value = intensity.get("forecast") or intensity.get("actual")
            if value is not None:
                forecast.append(
                    {
                        "from": slot.get("from", ""),
                        "to": slot.get("to", ""),
                        "value": int(value),
                        "level": self._map_carbon_index(str(intensity.get("index", "moderate"))),
                    }
                )
        self._carbon_forecast_cache = forecast
        return forecast

    def get_weather(self, sim_minute: float | None = None) -> dict[str, Any]:
        if sim_minute is None:
            return self._get_weather_wall_clock_now()
        return self._get_weather_at_sim(float(sim_minute))

    def _get_weather_wall_clock_now(self) -> dict[str, Any]:
        now = time.time()
        if self._weather_cache and (now - self._weather_last_fetched) < self._weather_ttl:
            result = dict(self._weather_cache)
            result["source"] = "cached"
            result["hourly_forecast"] = list(result.get("hourly_forecast") or [])
            self.api_status["weather"] = "cached"
            return result

        try:
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "hourly": "temperature_2m",
                "timezone": "Europe/London",
                "forecast_days": 1,
            }
            response = self._client.get(f"{WEATHER_BASE_URL}/v1/forecast", params=params)
            response.raise_for_status()
            data = response.json()
            current = data["current"]
            temp = current["temperature_2m"]
            feels_like = current["apparent_temperature"]
            weather_code = current["weather_code"]
            wind_speed = current["wind_speed_10m"]
            hourly_temps = data.get("hourly", {}).get("temperature_2m", [])
            result = {
                "temperature": round(float(temp), 1),
                "feels_like": round(float(feels_like), 1),
                "condition": map_wmo_weather_code(int(weather_code)),
                "wind_speed": round(float(wind_speed), 1),
                "is_heatwave": float(temp) > 30.0,
                "is_cold_snap": float(temp) < 2.0,
                "source": "live",
                "hourly_forecast": [round(float(t), 1) for t in hourly_temps],
            }
            self._weather_cache = result
            self._weather_last_fetched = now
            self.api_status["weather"] = "live"
            logger.info("Weather API: %s°C, %s", temp, result["condition"])
            return result
        except Exception as e:
            logger.warning("Weather API failed: %s. Using fallback.", e)
            self.api_status["weather"] = "error" if not self._weather_cache else "cached"
            return self._weather_cache or self._synthetic_weather(None)

    def _ensure_weather_hourly_series(self) -> None:
        if self._weather_hour_starts:
            return
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "timezone": "Europe/London",
            "forecast_days": WEATHER_FORECAST_DAYS,
        }
        response = self._client.get(f"{WEATHER_BASE_URL}/v1/forecast", params=params)
        response.raise_for_status()
        data = response.json()
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        temps = hourly.get("temperature_2m") or []
        feels = hourly.get("apparent_temperature") or []
        codes = hourly.get("weather_code") or []
        winds = hourly.get("wind_speed_10m") or []
        starts: list[datetime] = []
        rows: list[dict[str, Any]] = []
        for i, ts in enumerate(times):
            t0 = self._parse_om_time(str(ts))
            starts.append(t0)
            rows.append(
                {
                    "temperature": float(temps[i]) if i < len(temps) else 10.0,
                    "feels_like": float(feels[i]) if i < len(feels) else float(temps[i]) if i < len(temps) else 10.0,
                    "weather_code": int(codes[i]) if i < len(codes) else 3,
                    "wind_speed": float(winds[i]) if i < len(winds) else 10.0,
                }
            )
        self._weather_hour_starts = starts
        self._weather_hour_rows = rows

    def _get_weather_at_sim(self, sim_minute: float) -> dict[str, Any]:
        virtual = self._sim_epoch_utc + timedelta(minutes=sim_minute)
        if virtual.tzinfo is None:
            virtual = virtual.replace(tzinfo=timezone.utc)
        v_local = virtual.astimezone(LONDON_TZ)

        try:
            self._ensure_weather_hourly_series()
            if not self._weather_hour_starts:
                raise ValueError("Empty weather series")

            idx = bisect.bisect_right(self._weather_hour_starts, v_local) - 1
            idx = max(0, min(idx, len(self._weather_hour_rows) - 1))
            row = self._weather_hour_rows[idx]
            temp = row["temperature"]
            feels = row["feels_like"]
            code = row["weather_code"]
            wind = row["wind_speed"]

            hf = [
                round(float(self._weather_hour_rows[j]["temperature"]), 1)
                for j in range(idx, min(idx + 24, len(self._weather_hour_rows)))
            ]

            self.api_status["weather"] = "live"
            return {
                "temperature": round(float(temp), 1),
                "feels_like": round(float(feels), 1),
                "condition": map_wmo_weather_code(int(code)),
                "wind_speed": round(float(wind), 1),
                "is_heatwave": float(temp) > 30.0,
                "is_cold_snap": float(temp) < 2.0,
                "source": "live",
                "hourly_forecast": hf,
            }
        except Exception as e:
            logger.warning("Weather at sim_minute=%s failed: %s. Using fallback.", sim_minute, e)
            self.api_status["weather"] = "error"
            return self._synthetic_weather(v_local)

    def _map_carbon_index(self, index: str) -> str:
        mapping = {
            "very low": "low",
            "low": "low",
            "moderate": "medium",
            "high": "high",
            "very high": "high",
        }
        return mapping.get(index.lower().strip(), "medium")

    def _synthetic_carbon(self, virtual_utc: datetime | None) -> dict[str, Any]:
        if virtual_utc is None:
            hour = datetime.now(LONDON_TZ).hour
        else:
            hour = virtual_utc.astimezone(LONDON_TZ).hour
        base = float(config.CARBON_HOURLY_BASELINE[hour])
        value = max(50, int(base + random.gauss(0, 15)))
        level = "low" if value < 100 else ("medium" if value < 200 else "high")
        return {
            "value": value,
            "level": level,
            "source": "synthetic",
            "forecast": [],
        }

    def _synthetic_weather(self, v_local: datetime | None) -> dict[str, Any]:
        if v_local is None:
            hour = datetime.now(timezone.utc).hour
        else:
            hour = v_local.hour
        temp = config.WEATHER_BASELINE_TEMP + 4 * math.sin((hour - 6) * math.pi / 8)
        temp = round(float(temp), 1)
        return {
            "temperature": temp,
            "feels_like": round(temp - 2.0, 1),
            "condition": "cloudy",
            "wind_speed": 10.0,
            "is_heatwave": False,
            "is_cold_snap": False,
            "source": "synthetic",
            "hourly_forecast": [],
        }

    def close(self) -> None:
        self._client.close()

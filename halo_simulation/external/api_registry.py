"""Registry of permitted external APIs for LLM specialist decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ApiDefinition:
    api_id: str
    name: str
    description: str
    base_url: str
    endpoint: str
    method: str
    params: dict[str, Any]
    param_description: str
    trigger_conditions: list[str]
    trigger_keywords: list[str]
    halo_message_type: str
    result_schema: dict[str, Any]
    cooldown_minutes: int = 60
    enabled: bool = True


class ApiRegistry:
    def __init__(self) -> None:
        self._apis: dict[str, ApiDefinition] = {}
        self._last_called: dict[str, float] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all built-in external API definitions."""
        defs = [
            ApiDefinition(
                api_id="fuel_prices",
                name="UK Government Fuel Price Data",
                description=(
                    "Returns average UK petrol and diesel prices per litre in pence, "
                    "updated weekly by the UK government."
                ),
                base_url="https://www.gov.uk",
                endpoint="/government/statistics/weekly-road-fuel-prices",
                method="GET",
                params={},
                param_description=(
                    "No parameters needed. Fetches the official weekly road fuel prices statistics landing page (HTML)."
                ),
                trigger_conditions=["WeatherUpdate", "CarbonIntensityUpdate"],
                trigger_keywords=["cold snap", "heatwave", "high carbon", "very high", "heating"],
                halo_message_type="CostPressureUpdate",
                result_schema={
                    "petrol_pence_per_litre": float,
                    "diesel_pence_per_litre": float,
                },
                cooldown_minutes=240,
            ),
            ApiDefinition(
                api_id="grocery_prices",
                name="Open Food Facts Product Search",
                description=(
                    "Searches Open Food Facts database for grocery product information "
                    "including nutritional data. Useful for shopping agent decisions."
                ),
                base_url="https://world.openfoodfacts.org",
                endpoint="/cgi/search.pl",
                method="GET",
                params={
                    "action": "process",
                    "json": "1",
                    "page_size": "5",
                    "sort_by": "popularity",
                },
                param_description=(
                    "Add 'search_terms' param for the product to search. "
                    "Returns product names, brands, and nutritional info."
                ),
                trigger_conditions=["SleepNotice", "ArrivalNotice"],
                trigger_keywords=["sleep", "arrived home", "evening", "shopping"],
                halo_message_type="GrocerySignalUpdate",
                result_schema={"products": list, "count": int},
                cooldown_minutes=180,
            ),
            ApiDefinition(
                api_id="news_disruptions",
                name="NewsAPI — London Disruption Headlines",
                description=(
                    "Fetches recent news headlines about disruptions in London: transport strikes, "
                    "weather warnings, energy alerts, major events that could affect occupant schedules."
                ),
                base_url="https://newsapi.org",
                endpoint="/v2/everything",
                method="GET",
                params={
                    "q": "London disruption OR strike OR weather warning OR power outage",
                    "language": "en",
                    "pageSize": "5",
                    "sortBy": "publishedAt",
                },
                param_description=(
                    "Searches for recent London disruption news. Requires 'apiKey' param "
                    "added at call time from NEWSAPI_KEY or NEWS_API env var."
                ),
                trigger_conditions=["DepartureNotice", "ArrivalNotice", "WeatherUpdate"],
                trigger_keywords=["departure", "arrived", "cold snap", "heatwave", "storm"],
                halo_message_type="ExternalDisruptionEvent",
                result_schema={"articles": list, "total_results": int},
                cooldown_minutes=120,
            ),
            ApiDefinition(
                api_id="severe_weather",
                name="Open-Meteo Hourly Forecast — Severe Conditions",
                description=(
                    "Fetches the next 12 hours of hourly temperature and precipitation forecast "
                    "for London. Used to detect upcoming heatwaves, cold snaps, or storms before they arrive."
                ),
                base_url="https://api.open-meteo.com",
                endpoint="/v1/forecast",
                method="GET",
                params={
                    "latitude": 51.5074,
                    "longitude": -0.1278,
                    "hourly": "temperature_2m,precipitation_probability,weather_code",
                    "timezone": "Europe/London",
                    "forecast_days": 1,
                },
                param_description="Returns hourly forecast for London today. No API key needed.",
                trigger_conditions=["WeatherUpdate", "CarbonIntensityUpdate"],
                trigger_keywords=["temperature", "cold", "hot", "heatwave", "storm", "rain"],
                halo_message_type="WeatherForecastAlert",
                result_schema={"hourly": dict},
                cooldown_minutes=90,
            ),
        ]
        for d in defs:
            self._apis[d.api_id] = d

    def get(self, api_id: str) -> Optional[ApiDefinition]:
        return self._apis.get(api_id)

    def all(self) -> list[ApiDefinition]:
        return list(self._apis.values())

    def is_on_cooldown(self, api_id: str, current_sim_time: float) -> bool:
        last = self._last_called.get(api_id, -9999.0)
        api = self._apis.get(api_id)
        if not api:
            return True
        return (current_sim_time - last) < float(api.cooldown_minutes)

    def mark_called(self, api_id: str, sim_time: float) -> None:
        self._last_called[api_id] = sim_time

    def get_summary_for_prompt(self) -> str:
        """
        Returns a plain text summary of all enabled APIs for inclusion
        in the LLM reasoning prompt.
        """
        lines: list[str] = []
        for api in self._apis.values():
            if not api.enabled:
                continue
            lines.append(
                f"- api_id: {api.api_id}\n"
                f"  name: {api.name}\n"
                f"  description: {api.description}\n"
                f"  relevant when: {', '.join(api.trigger_conditions)} events "
                f"or keywords: {', '.join(api.trigger_keywords)}"
            )
        return "\n\n".join(lines)

    def get_relevant_apis(self, msg_type: str, summary: str) -> list[ApiDefinition]:
        """
        Returns APIs whose trigger_conditions include msg_type OR
        whose trigger_keywords appear in summary (case insensitive).
        """
        relevant: list[ApiDefinition] = []
        summary_lower = summary.lower()
        for api in self._apis.values():
            if not api.enabled:
                continue
            if msg_type in api.trigger_conditions:
                relevant.append(api)
                continue
            if any(kw in summary_lower for kw in api.trigger_keywords):
                relevant.append(api)
        return relevant

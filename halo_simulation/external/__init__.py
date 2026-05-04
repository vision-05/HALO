"""External live data integrations (carbon, weather)."""

from external.api_client import (
    ExternalDataClient,
    fetch_weather_hourly_chart_data,
    map_wmo_weather_code,
)

__all__ = ["ExternalDataClient", "fetch_weather_hourly_chart_data", "map_wmo_weather_code"]

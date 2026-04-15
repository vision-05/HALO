"""Two occupants with conflicting temperature preferences."""

from __future__ import annotations

from halo_simulation import config
from halo_simulation.agents.device_agent import ThermostatDeviceAgent
from halo_simulation.agents.person_agent import PersonAgent
from halo_simulation.agents.specialist_agent import GridCarbonAgent, WeatherAgent
from halo_simulation.metrics.collector import MetricsCollector
from halo_simulation.scenarios.base_scenario import BaseScenario


class TemperatureConflictScenario(BaseScenario):
    def __init__(self, seed: int, days: int = 14, api_client=None) -> None:
        metrics = MetricsCollector("temperature_conflict")
        super().__init__(seed, days, metrics)
        self._api_client = api_client

    def build(self) -> None:
        alice = PersonAgent(
            "person_alice",
            "Alice",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            schedule={
                "wake": 6 * 60,
                "leave": 8 * 60 + 30,
                "return": 18 * 60,
                "sleep": 23 * 60,
            },
            preferred_temperature=22.0,
            scenario_name="temperature_conflict",
        )
        bob = PersonAgent(
            "person_bob",
            "Bob",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            schedule={
                "wake": 7 * 60,
                "leave": 8 * 60 + 30,
                "return": 18 * 60,
                "sleep": 23 * 60,
            },
            preferred_temperature=19.0,
            scenario_name="temperature_conflict",
            skip_commute=True,
        )
        thermo = ThermostatDeviceAgent(
            "device_thermostat",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            scenario_name="temperature_conflict",
        )
        weather = WeatherAgent(
            "specialist_weather",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            season_offset=config.WEATHER_WINTER_OFFSET,
            api_client=self._api_client,
        )
        carbon = GridCarbonAgent(
            "specialist_carbon",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            force_evening_peak=False,
            api_client=self._api_client,
        )
        self._agents = [alice, bob, thermo, weather, carbon]

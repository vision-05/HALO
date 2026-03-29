"""Thermostat failures with recovery; rest of network continues."""

from __future__ import annotations

from halo_simulation import config
from halo_simulation.agents.device_agent import ShowerDeviceAgent, ThermostatDeviceAgent
from halo_simulation.agents.person_agent import PersonAgent
from halo_simulation.agents.specialist_agent import GridCarbonAgent, WeatherAgent
from halo_simulation.metrics.collector import MetricsCollector
from halo_simulation.scenarios.base_scenario import BaseScenario


class DeviceFailureScenario(BaseScenario):
    def __init__(self, seed: int, days: int = 7) -> None:
        metrics = MetricsCollector("device_failure")
        super().__init__(seed, days, metrics)

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
            preferred_temperature=21.0,
            scenario_name="device_failure",
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
            preferred_temperature=20.0,
            scenario_name="device_failure",
        )
        thermo = ThermostatDeviceAgent(
            "device_thermostat",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            scenario_name="device_failure",
            failure_probability=0.05,
        )
        shower = ShowerDeviceAgent(
            "device_shower",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            scenario_name="device_failure",
        )
        weather = WeatherAgent(
            "specialist_weather",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            season_offset=config.WEATHER_WINTER_OFFSET,
        )
        carbon = GridCarbonAgent(
            "specialist_carbon",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            force_evening_peak=False,
        )
        self._agents = [alice, bob, thermo, shower, weather, carbon]

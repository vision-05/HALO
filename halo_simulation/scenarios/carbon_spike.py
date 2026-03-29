"""High evening carbon — dishwasher defers; thermostat responds via protocol weights."""

from __future__ import annotations

from halo_simulation.agents.device_agent import DishwasherDeviceAgent, ThermostatDeviceAgent
from halo_simulation.agents.person_agent import PersonAgent
from halo_simulation.agents.specialist_agent import GridCarbonAgent
from halo_simulation.metrics.collector import MetricsCollector
from halo_simulation.scenarios.base_scenario import BaseScenario


class CarbonSpikeScenario(BaseScenario):
    def __init__(self, seed: int, days: int = 7) -> None:
        metrics = MetricsCollector("carbon_spike")
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
                "leave": 9 * 60,
                "return": 17 * 60,
                "sleep": 23 * 60,
            },
            preferred_temperature=20.0,
            scenario_name="carbon_spike",
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
                "leave": 9 * 60,
                "return": 17 * 60,
                "sleep": 23 * 60,
            },
            preferred_temperature=20.5,
            scenario_name="carbon_spike",
        )
        thermo = ThermostatDeviceAgent(
            "device_thermostat",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            scenario_name="carbon_spike",
        )
        dish = DishwasherDeviceAgent(
            "device_dishwasher",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            scenario_name="carbon_spike",
        )
        carbon = GridCarbonAgent(
            "specialist_carbon",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            force_evening_peak=True,
        )
        self._agents = [alice, bob, thermo, dish, carbon]

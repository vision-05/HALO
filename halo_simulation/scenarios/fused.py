"""Fused demonstration: scripted Alice + Bob, human CliPersonAgent, thermostat, dishwasher,
shower, specialists; evening carbon spike; lower random thermostat failure."""

from __future__ import annotations

import queue
from typing import Any

from halo_simulation import config
from halo_simulation.agents.cli_person import CliPersonAgent
from halo_simulation.agents.device_agent import (
    DishwasherDeviceAgent,
    ShowerDeviceAgent,
    ThermostatDeviceAgent,
)
from halo_simulation.agents.person_agent import PersonAgent
from halo_simulation.agents.specialist_agent import GridCarbonAgent, WeatherAgent
from halo_simulation.human_bridge import BridgeInjector, CLI_PERSON_ID
from halo_simulation.metrics.collector import MetricsCollector
from halo_simulation.scenarios.base_scenario import BaseScenario


class FusedScenario(BaseScenario):
    """
    Single continuous SimPy run combining conflict, carbon-aware appliances, resilience,
    and human-in-the-loop negotiation (inject queue).

    Pass ``inject_queue`` for REST/CLI bridge commands (see ``human_bridge`` module).
    """

    def __init__(
        self,
        seed: int,
        days: int,
        inject_queue: queue.Queue,
        api_client: Any | None = None,
        status_reply: queue.Queue | None = None,
    ) -> None:
        metrics = MetricsCollector("fused")
        super().__init__(seed, days, metrics)
        self._inject_queue = inject_queue
        self._api_client = api_client
        self._status_reply: queue.Queue = status_reply if status_reply is not None else queue.Queue(maxsize=4)

    @property
    def status_reply_queue(self) -> queue.Queue:
        return self._status_reply

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
                "return": 17 * 60 + 30,
                "sleep": 23 * 60,
            },
            preferred_temperature=22.0,
            scenario_name="fused",
        )
        bob = PersonAgent(
            "person_bob",
            "Bob",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            schedule={
                "wake": 11 * 60,
                "leave": 15 * 60,
                "return": 17 * 60,
                "sleep": 23 * 60,
            },
            preferred_temperature=19.0,
            scenario_name="fused",
        )
        cli = CliPersonAgent(
            CLI_PERSON_ID,
            "CLI",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            # Placeholders for learningModel only; CliPersonAgent uses manual_schedule (default True).
            schedule={
                "wake": 0,
                "leave": 12 * 60,
                "return": 13 * 60,
                "sleep": 23 * 60,
            },
            manual_negotiation=True,
            preferred_temperature=25.0,
            scenario_name="fused",
        )
        thermo = ThermostatDeviceAgent(
            "device_thermostat",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            scenario_name="fused",
            # NOTE: sampled once per simulated minute in ThermostatDeviceAgent.run —
            # 0.02 would almost always fault before the first negotiating pair declares prefs.
            failure_probability=float("5e-5"),
        )
        dish = DishwasherDeviceAgent(
            "device_dishwasher",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            scenario_name="fused",
        )
        shower = ShowerDeviceAgent(
            "device_shower",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            scenario_name="fused",
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
            force_evening_peak=True,
            api_client=self._api_client,
        )
        injector = BridgeInjector(
            self.env,
            self.bus,
            self._inject_queue,
            cli,
            status_reply=self._status_reply,
        )
        self._agents = [alice, bob, cli, thermo, dish, shower, weather, carbon, injector]

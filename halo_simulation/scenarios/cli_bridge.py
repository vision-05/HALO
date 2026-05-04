"""Demo scenario: automated occupant + CLI-controlled ``person_cli`` + thermostat + specialists."""

from __future__ import annotations

import queue
from typing import Any

import config
from agents.cli_person import CliPersonAgent
from agents.device_agent import ThermostatDeviceAgent
from agents.person_agent import PersonAgent
from agents.specialist_agent import GridCarbonAgent, WeatherAgent
from human_bridge import BridgeInjector, CLI_PERSON_ID
from metrics.collector import MetricsCollector
from scenarios.base_scenario import BaseScenario


class CliBridgeScenario(BaseScenario):
    """
    Two people (Bob scripted, ``person_cli`` manual negotiation) share one thermostat.

    Pass ``inject_queue`` for human-in-the-loop commands (see ``human_bridge`` module docstring).
    """

    def __init__(
        self,
        seed: int,
        days: int,
        inject_queue: queue.Queue,
        api_client: Any | None = None,
        status_reply: queue.Queue | None = None,
    ) -> None:
        metrics = MetricsCollector("cli_bridge")
        super().__init__(seed, days, metrics)
        self._inject_queue = inject_queue
        self._api_client = api_client
        self._status_reply: queue.Queue = status_reply if status_reply is not None else queue.Queue(maxsize=4)

    @property
    def status_reply_queue(self) -> queue.Queue:
        return self._status_reply

    def build(self) -> None:
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
            preferred_temperature=19.0,
            scenario_name="cli_bridge",
        )
        cli = CliPersonAgent(
            CLI_PERSON_ID,
            "CLI",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            schedule={
                "wake": 6 * 60,
                "leave": 8 * 60,
                "return": 18 * 60,
                "sleep": 23 * 60,
            },
            manual_negotiation=True,
            preferred_temperature=23.0,
            scenario_name="cli_bridge",
            skip_commute=True,
        )
        thermo = ThermostatDeviceAgent(
            "device_thermostat",
            self.env,
            self.bus,
            self.rng,
            self.metrics,
            scenario_name="cli_bridge",
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
        injector = BridgeInjector(
            self.env,
            self.bus,
            self._inject_queue,
            cli,
            status_reply=self._status_reply,
        )
        self._agents = [bob, cli, thermo, weather, carbon, injector]

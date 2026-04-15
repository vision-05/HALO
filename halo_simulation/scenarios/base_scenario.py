"""Base scenario — SimPy environment wiring."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import simpy

from halo_simulation import config
from halo_simulation.metrics.collector import MetricsCollector
from halo_simulation.negotiation.message import MessageBus

logger = logging.getLogger(__name__)


class BaseScenario(ABC):
    def __init__(
        self,
        seed: int,
        days: int,
        metrics: MetricsCollector,
    ) -> None:
        self.seed = seed
        self.days = days
        self.metrics = metrics
        self.rng = np.random.default_rng(seed)
        self.env = simpy.Environment()
        self.bus = MessageBus(self.env, metrics=metrics)
        self._agents: list[Any] = []

    def register_all(self) -> None:
        for a in self._agents:
            self.bus.register(a)

    def start_processes(self) -> None:
        for a in self._agents:
            self.env.process(a.run())

    @abstractmethod
    def build(self) -> None:
        ...

    def run_simulation(self) -> dict[str, Any]:
        self.build()
        self.register_all()
        self.start_processes()
        until = config.MINUTES_PER_DAY * self.days
        logger.info("Running scenario %s until t=%s", self.metrics.scenario_name, until)
        self.env.run(until=until)
        paths = self.metrics.save_outputs()
        stats = self.metrics.summary_stats()
        stats["output_paths"] = paths
        return stats

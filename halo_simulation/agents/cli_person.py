"""CLI-controlled advocator: manual thermostat negotiation responses."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import simpy

from halo_simulation import config
from halo_simulation.agents.person_agent import PersonAgent
from halo_simulation.metrics.collector import MetricsCollector
from halo_simulation.negotiation.message import Message, MessageTypes

logger = logging.getLogger(__name__)


class CliPersonAgent(PersonAgent):
    """
    Like ``PersonAgent`` but does **not** auto-reply to ``NegotiationProposal`` when
    ``manual_negotiation`` is True (use bridge ``send-counter`` / ``send-accept`` / ``send-reject``).
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        env: simpy.Environment,
        message_bus: Any,
        rng: np.random.Generator,
        metrics: MetricsCollector | None,
        schedule: dict[str, int],
        manual_negotiation: bool = True,
        preferred_temperature: float = 21.0,
        scenario_name: str = "cli_bridge",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id,
            name,
            env,
            message_bus,
            rng,
            metrics,
            schedule=schedule,
            preferred_temperature=preferred_temperature,
            scenario_name=scenario_name,
            **kwargs,
        )
        self.manual_negotiation = manual_negotiation
        self._pending_negotiation: dict[str, Any] | None = None

    @property
    def pending_negotiation(self) -> dict[str, Any] | None:
        return self._pending_negotiation.copy() if self._pending_negotiation else None

    @property
    def state_snapshot(self) -> dict[str, Any]:
        return {
            "preferred_temperature": float(self._state.get("preferred_temperature", 21.0)),
            "is_home": bool(self._state.get("is_home", True)),
            "comfort_weight": float(self._state.get("comfort_weight", config.DEFAULT_COMFORT_WEIGHT)),
        }

    def _handle_message(self, msg: Message) -> None:
        if self.manual_negotiation and msg.msg_type == MessageTypes.NegotiationProposal:
            pl = msg.payload
            self._pending_negotiation = {
                "negotiation_id": pl.get("negotiation_id", ""),
                "device_id": pl.get("device_id", msg.sender_id),
                "proposed_value": float(pl.get("proposed_value", 0.0)),
                "attribute": pl.get("attribute", "temperature"),
                "round": pl.get("round"),
            }
            logger.info("CliPersonAgent %s: awaiting manual reply for nid=%s", self.agent_id, self._pending_negotiation.get("negotiation_id"))
            return
        super()._handle_message(msg)

    def set_preferred_temperature(self, value: float) -> None:
        v = float(max(config.THERMOSTAT_MIN, min(config.THERMOSTAT_MAX, value)))
        self._state["preferred_temperature"] = v

    def broadcast_preferences(self) -> None:
        self._broadcast_preferences()

    def simulate_leave(self) -> None:
        self._last_leave_minute = float(self.env.now % config.MINUTES_PER_DAY)
        self._state["is_home"] = False
        self._state["comfort_weight"] = config.AWAY_COMFORT_WEIGHT
        m = Message.create(
            self.agent_id,
            "broadcast",
            MessageTypes.DepartureNotice,
            {"name": self.name},
            self.env.now,
        )
        self.broadcast(m)

    def simulate_return(self) -> None:
        self._last_return_minute = float(self.env.now % config.MINUTES_PER_DAY)
        self._state["is_home"] = True
        self._state["comfort_weight"] = self._comfort_weight_home
        m = Message.create(
            self.agent_id,
            "broadcast",
            MessageTypes.ArrivalNotice,
            {"name": self.name},
            self.env.now,
        )
        self.broadcast(m)
        self._broadcast_preferences()

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
    Like ``PersonAgent`` but:

    - **manual_negotiation** (default True): does not auto-reply to ``NegotiationProposal`` —
      use inject ``send_counter`` / ``send_accept`` / ``send_reject``.
    - **manual_schedule** (default True): no simulated wake / leave / return / sleep — presence
      and preference broadcasts come only from inject (``set_pref``, ``leave``, ``return``).
      Schedule dict is retained only for learning helpers that read ``sleep`` / leave-return hints.
      Set ``manual_schedule=False`` to restore scripted day-cycle behaviour (e.g. automated demos).
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
        manual_schedule: bool = True,
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
        self._manual_schedule = manual_schedule
        self._pending_negotiation: dict[str, Any] | None = None

    def run(self):
        if self._manual_schedule:
            yield from self._manual_presence_loop()
        else:
            yield from super().run()

    def _manual_presence_loop(self):
        """Inbox-only loop; fires end-of-day learning when simulation date advances."""
        prev_day = int(self.env.now // config.MINUTES_PER_DAY)
        while True:
            cur_day = int(self.env.now // config.MINUTES_PER_DAY)
            while prev_day < cur_day:
                self._end_of_day_learning(float(prev_day * config.MINUTES_PER_DAY))
                prev_day += 1

            deadline = self.env.now + 1.0
            while self.env.now < deadline:
                remaining = max(0.0, deadline - self.env.now)
                res, get_ev = yield from self.wait_inbox_or_timeout(remaining)
                if get_ev in res:
                    self._handle_message(res[get_ev])
                    yield from self.drain_inbox_burst(self._handle_message)
                else:
                    yield from self.drain_inbox_burst(self._handle_message)
                    break

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

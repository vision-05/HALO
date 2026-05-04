"""Advocator agent — human occupant with preferences and schedule."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import simpy

import config
from agents.base_agent import BaseAgent
from learning.preference_model import PreferenceModel
from metrics.collector import LearningEvent, MetricsCollector
from negotiation.message import Message, MessageTypes

logger = logging.getLogger(__name__)


class PersonAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        name: str,
        env: simpy.Environment,
        message_bus: Any,
        rng: np.random.Generator,
        metrics: MetricsCollector | None,
        schedule: dict[str, int],
        schedule_noise_std: int = 15,
        comfort_weight: float | None = None,
        preferred_temperature: float = 21.0,
        preferred_lighting: float = 70.0,
        scenario_name: str = "default",
        skip_commute: bool = False,
    ) -> None:
        super().__init__(agent_id, "person", env, message_bus, metrics)
        self.name = name
        self._skip_commute = skip_commute
        self._rng = rng
        self._schedule_base = dict(schedule)
        self.schedule_noise_std = schedule_noise_std
        cw = comfort_weight if comfort_weight is not None else config.DEFAULT_COMFORT_WEIGHT
        self._comfort_weight_home = cw
        self._state["comfort_weight"] = cw
        self._state["is_home"] = True
        self._state["preferred_temperature"] = preferred_temperature
        self._state["preferred_lighting"] = preferred_lighting
        self._scenario_name = scenario_name

        self.preference_model = PreferenceModel("thermostat", rng)
        self._last_declared_temp = preferred_temperature
        self._last_day_index = -1
        self._last_leave_minute = float(self._schedule_base["leave"])
        self._last_return_minute = float(self._schedule_base["return"])

    @property
    def comfort_weight(self) -> float:
        return float(self._state.get("comfort_weight", config.DEFAULT_COMFORT_WEIGHT))

    @property
    def is_home(self) -> bool:
        return bool(self._state.get("is_home", True))

    def _sample_day_minutes(self, day_start: float) -> dict[str, float]:
        noise = lambda: float(self._rng.normal(0, self.schedule_noise_std))
        base = self._schedule_base
        return {
            "wake": day_start + base["wake"] + noise(),
            "leave": day_start + base["leave"] + noise(),
            "return": day_start + base["return"] + noise(),
            "sleep": day_start + base["sleep"] + noise(),
        }

    def run(self):
        while True:
            day_start = (self.env.now // config.MINUTES_PER_DAY) * config.MINUTES_PER_DAY
            times = self._sample_day_minutes(day_start)
            raw_events = [
                ("wake", times["wake"]),
                ("leave", times["leave"]),
                ("return", times["return"]),
                ("sleep", times["sleep"]),
            ]
            if self._skip_commute:
                raw_events = [(k, t) for k, t in raw_events if k not in ("leave", "return")]
            events = sorted(raw_events, key=lambda x: x[1])

            for kind, abs_t in events:
                while self.env.now < abs_t:
                    delay = abs_t - self.env.now
                    result, get_ev = yield from self.wait_inbox_or_timeout(delay)
                    if get_ev in result:
                        self._handle_message(result[get_ev])
                        yield from self.drain_inbox_burst(self._handle_message)
                    else:
                        yield from self.drain_inbox_burst(self._handle_message)
                self._dispatch_presence(kind, day_start, abs_t)

            self._end_of_day_learning(day_start)
            next_day = day_start + config.MINUTES_PER_DAY
            yield self.env.timeout(max(0.0, next_day - self.env.now))

    def _dispatch_presence(self, kind: str, day_start: float, abs_t: float) -> None:
        if kind == "wake":
            self._broadcast_preferences()
        elif kind == "leave":
            self._last_leave_minute = float(abs_t % config.MINUTES_PER_DAY)
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
        elif kind == "return":
            self._last_return_minute = float(abs_t % config.MINUTES_PER_DAY)
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
        elif kind == "sleep":
            m = Message.create(
                self.agent_id,
                "broadcast",
                MessageTypes.SleepNotice,
                {"name": self.name},
                self.env.now,
            )
            self.broadcast(m)

    def _broadcast_preferences(self) -> None:
        payload = {
            "person_id": self.agent_id,
            "preferences": {
                "temperature": self._state["preferred_temperature"],
                "lighting": self._state["preferred_lighting"],
            },
            "comfort_weight": self._state["comfort_weight"],
            "is_home": self._state["is_home"],
        }
        m = Message.create(
            self.agent_id,
            "broadcast",
            MessageTypes.PreferenceDeclaration,
            payload,
            self.env.now,
        )
        self.broadcast(m)

    def _handle_message(self, msg: Message) -> None:
        if msg.msg_type == MessageTypes.NegotiationProposal:
            self._respond_negotiation(msg)
        elif msg.msg_type == MessageTypes.NegotiationResolved:
            pl = msg.payload
            if pl.get("attribute") == "temperature":
                self.record_resolved_temperature(float(pl.get("final_value", self._last_declared_temp)))
        elif msg.msg_type == MessageTypes.ActuationCommand:
            pass

    def _respond_negotiation(self, msg: Message) -> None:
        pl = msg.payload
        proposed = float(pl.get("proposed_value", 0))
        nid = pl.get("negotiation_id", "")
        device_id = pl.get("device_id", "")
        attr = pl.get("attribute", "temperature")
        pref = float(self._state.get("preferred_temperature", 21.0))
        tol = max(
            config.TEMPERATURE_TOLERANCE,
            self.preference_model.tolerance_from_bayesian(),
        )
        recipient = msg.sender_id

        if attr == "temperature":
            if proposed < config.THERMOSTAT_MIN:
                out = Message.create(
                    self.agent_id,
                    recipient,
                    MessageTypes.NegotiationReject,
                    {"negotiation_id": nid, "reason": "below_min_safe", "device_id": device_id},
                    self.env.now,
                )
                self.send(recipient, out)
                return
            if abs(proposed - pref) <= tol:
                out = Message.create(
                    self.agent_id,
                    recipient,
                    MessageTypes.NegotiationAccept,
                    {"negotiation_id": nid, "device_id": device_id},
                    self.env.now,
                )
            else:
                # Move toward compromise (not a hard re-assert of pref) so variance can fall
                # below CONVERGENCE_THRESHOLD within MAX_ITERATIONS.
                counter_val = (proposed + pref) / 2.0
                counter_val = max(
                    config.THERMOSTAT_MIN,
                    min(config.THERMOSTAT_MAX, counter_val),
                )
                out = Message.create(
                    self.agent_id,
                    recipient,
                    MessageTypes.NegotiationCounter,
                    {
                        "negotiation_id": nid,
                        "counter_value": counter_val,
                        "device_id": device_id,
                        "attribute": attr,
                    },
                    self.env.now,
                )
            self.send(recipient, out)

    def _end_of_day_learning(self, day_start: float) -> None:
        day_index = int(day_start // config.MINUTES_PER_DAY)
        if day_index == self._last_day_index:
            return
        self._last_day_index = day_index

        dep = self._last_leave_minute
        arr = self._last_return_minute
        hour = int((self._schedule_base["sleep"] // 60) % 24)
        snap = self.preference_model.end_of_day_update(
            self._last_declared_temp,
            hour,
            dep,
            arr,
        )
        if self._metrics:
            self._metrics.log_learning(
                LearningEvent(
                    timestamp=self.env.now,
                    person_id=self.agent_id,
                    device_type="thermostat",
                    ema_value=snap["ema_value"],
                    bayesian_mu=snap["bayesian_mu"],
                    bayesian_sigma=snap["bayesian_sigma"],
                    routine_stable=snap["routine_stable"],
                )
            )

    def record_resolved_temperature(self, final_temp: float) -> None:
        """Call when negotiation resolves so learning tracks observed compromise."""
        self._last_declared_temp = final_temp

"""Advocator agent — human occupant with preferences and schedule."""

from __future__ import annotations

import logging
from typing import Any

from halo_simulation.household_meals import HouseholdMealContext

import numpy as np
import simpy

from halo_simulation import config
from halo_simulation.agents.base_agent import BaseAgent
from halo_simulation.learning.preference_model import PreferenceModel
from halo_simulation.metrics.collector import LearningEvent, MetricsCollector
from halo_simulation.negotiation import protocol
from halo_simulation.negotiation.message import Message, MessageTypes

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
        preferred_shower_minutes: float | None = None,
        scenario_name: str = "default",
        skip_commute: bool = False,
        favorite_meals: list[str] | None = None,
        meal_context: HouseholdMealContext | None = None,
        dishwasher_run_after_return_delay_range: tuple[float, float] | None = None,
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
        if preferred_shower_minutes is not None:
            self._state["preferred_shower_minutes"] = float(
                np.clip(
                    float(preferred_shower_minutes),
                    config.SHOWER_DURATION_MIN_MINUTES,
                    config.SHOWER_DURATION_MAX_MINUTES,
                )
            )
        self._scenario_name = scenario_name

        raw_meals = [str(x).strip() for x in (favorite_meals or []) if str(x).strip()]
        self._favorite_meals = raw_meals[:5]
        self._meal_context = meal_context
        if len(self._favorite_meals) >= 1 and meal_context is not None:
            meal_context.register_person(self.agent_id, self._favorite_meals)

        self.preference_model = PreferenceModel("thermostat", rng)
        self._last_declared_temp = preferred_temperature
        self._last_day_index = -1
        self._last_leave_minute = float(self._schedule_base["leave"])
        self._last_return_minute = float(self._schedule_base["return"])
        self._return_disruption_delta_minutes = 0.0
        self._dishwasher_after_return_delay_range = dishwasher_run_after_return_delay_range

    def set_favorite_meals(self, meals: list[str]) -> None:
        self._favorite_meals = [str(x).strip() for x in meals if str(x).strip()][:5]
        if self._meal_context is not None and len(self._favorite_meals) >= 1:
            self._meal_context.update_favorites(self.agent_id, self._favorite_meals)

    @property
    def comfort_weight(self) -> float:
        return float(self._state.get("comfort_weight", config.DEFAULT_COMFORT_WEIGHT))

    @property
    def is_home(self) -> bool:
        return bool(self._state.get("is_home", True))

    def _sample_day_minutes(self, day_start: float) -> dict[str, float]:
        noise = lambda: float(self._rng.normal(0, self.schedule_noise_std))
        base = self._schedule_base
        times = {
            "wake": day_start + base["wake"] + noise(),
            "leave": day_start + base["leave"] + noise(),
            "return": day_start + base["return"] + noise(),
            "sleep": day_start + base["sleep"] + noise(),
        }
        d = self._return_disruption_delta_minutes
        if d and not self._skip_commute:
            times["return"] = times["return"] + d
            cap = times["sleep"] - 60.0
            if times["return"] > cap:
                times["return"] = cap
        return times

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
            self._broadcast_preferences()
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
            if self._dishwasher_after_return_delay_range is not None:
                lo, hi = self._dishwasher_after_return_delay_range
                self.env.process(self._delayed_dishwasher_request(float(lo), float(hi)))
        elif kind == "sleep":
            m = Message.create(
                self.agent_id,
                "broadcast",
                MessageTypes.SleepNotice,
                {"name": self.name},
                self.env.now,
            )
            self.broadcast(m)
            self._record_evening_meal_if_applicable()

    def _delayed_dishwasher_request(self, lo: float, hi: float) -> Any:
        a, b = (lo, hi) if lo <= hi else (hi, lo)
        delay = float(self._rng.uniform(a, b))
        yield self.env.timeout(max(0.0, delay))
        m = Message.create(
            self.agent_id,
            "device_dishwasher",
            MessageTypes.DishwasherRunRequest,
            {"requester_id": self.agent_id, "urgency": 0.55},
            self.env.now,
        )
        self.send("device_dishwasher", m)

    def _record_evening_meal_if_applicable(self) -> None:
        if self._meal_context is None or len(self._favorite_meals) < 1:
            return
        meal = self._meal_context.pick_evening_meal(self.agent_id, self._rng)
        self._meal_context.record_dinner(self.agent_id, self.env.now, meal)

    def _broadcast_preferences(self) -> None:
        prefs: dict[str, Any] = {
            "temperature": self._state["preferred_temperature"],
            "lighting": self._state["preferred_lighting"],
        }
        if "preferred_shower_minutes" in self._state:
            prefs["shower_minutes"] = float(self._state["preferred_shower_minutes"])
        payload = {
            "person_id": self.agent_id,
            "preferences": prefs,
            "comfort_weight": self._state["comfort_weight"],
            "is_home": self._state["is_home"],
        }
        if self._favorite_meals:
            payload["favorite_meals"] = list(self._favorite_meals)
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
            attr = str(pl.get("attribute", "temperature") or "temperature")
            if attr == "temperature":
                self.record_resolved_temperature(float(pl.get("final_value", self._last_declared_temp)))
        elif msg.msg_type == MessageTypes.ActuationCommand:
            pass
        elif msg.msg_type == MessageTypes.ExternalDisruptionEvent:
            self._handle_external_disruption(msg.payload or {})

    def _handle_external_disruption(self, pl: dict[str, Any]) -> None:
        summary = str(pl.get("summary", "")).lower()
        severity = str(pl.get("severity", "low")).lower()
        transport_keywords = (
            "strike",
            "tube",
            "rail",
            "train",
            "bus",
            "delay",
            "disruption",
            "transport",
        )
        if any(kw in summary for kw in transport_keywords):
            delay_minutes = {"low": 15.0, "medium": 45.0, "high": 90.0}.get(severity, 30.0)
            self._return_disruption_delta_minutes += delay_minutes
            logger.info(
                "%s: transport disruption — return delayed by %s sim min (total delta %s)",
                self.name,
                delay_minutes,
                self._return_disruption_delta_minutes,
            )
            self._broadcast_preferences()
        energy_keywords = ("power", "outage", "blackout", "energy", "electricity", "grid")
        if any(kw in summary for kw in energy_keywords):
            cw = float(self._state.get("comfort_weight", config.DEFAULT_COMFORT_WEIGHT))
            self._state["comfort_weight"] = max(cw - 0.2, 0.3)
            logger.info("%s: energy disruption — comfort_weight reduced", self.name)

    def _respond_negotiation(self, msg: Message) -> None:
        pl = msg.payload
        proposed = float(pl.get("proposed_value", 0))
        nid = pl.get("negotiation_id", "")
        device_id = pl.get("device_id", "")
        attr = pl.get("attribute", "temperature")
        recipient = msg.sender_id

        if attr == "temperature":
            pref = float(self._state.get("preferred_temperature", 21.0))
            tol = max(
                config.TEMPERATURE_TOLERANCE,
                self.preference_model.tolerance_from_bayesian(),
            )
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
            return

        if attr == "dishwasher_delay":
            pref = 0.0
            tol = float(config.DISHWASHER_DELAY_NEGOTIATION_TOLERANCE_MIN)
            hi = float(config.DISHWASHER_DEFER_MINUTES_MAX)
            proposed_clamped = max(0.0, min(hi, proposed))
            if abs(proposed_clamped - pref) <= tol:
                out = Message.create(
                    self.agent_id,
                    recipient,
                    MessageTypes.NegotiationAccept,
                    {
                        "negotiation_id": nid,
                        "device_id": device_id,
                        "attribute": "dishwasher_delay",
                    },
                    self.env.now,
                )
            else:
                counter_val = (proposed_clamped + pref) / 2.0
                counter_val = max(0.0, min(hi, counter_val))
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
            return

        if attr == "shower_minutes":
            pref = float(protocol.shower_minutes_from_comfort_temp(float(self._state.get("preferred_temperature", 21.0))))
            tol = float(config.SHOWER_MINUTES_TOLERANCE)
            lo = float(config.SHOWER_DURATION_MIN_MINUTES)
            hi = float(config.SHOWER_DURATION_MAX_MINUTES)
            if proposed < lo:
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
                counter_val = (proposed + pref) / 2.0
                counter_val = max(lo, min(hi, counter_val))
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
            return

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

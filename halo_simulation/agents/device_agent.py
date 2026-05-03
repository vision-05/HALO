"""Governor agents — smart devices with state machines and negotiation."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
import simpy

from halo_simulation import config
from halo_simulation.agents.base_agent import BaseAgent
from halo_simulation.metrics.collector import FailureEvent, MetricsCollector, NegotiationEvent
from halo_simulation.negotiation import protocol
from halo_simulation.negotiation.message import Message, MessageTypes

logger = logging.getLogger(__name__)

_SHOWER_USER_OMIT = object()


class DeviceAgent(BaseAgent):
    """Base device with failure/recovery and optional state machine hooks."""

    def __init__(
        self,
        agent_id: str,
        device_type: str,
        env: simpy.Environment,
        message_bus: Any,
        rng: np.random.Generator,
        metrics: MetricsCollector | None,
        device_weight: float | None = None,
        energy_cost_per_hour: float = 0.5,
        carbon_sensitivity: float = 0.5,
        failure_probability: float | None = None,
        scenario_name: str = "default",
    ) -> None:
        super().__init__(agent_id, "device", env, message_bus, metrics)
        self.device_type = device_type
        self._state["device_state"] = "idle"
        self.device_weight = device_weight if device_weight is not None else config.DEFAULT_DEVICE_WEIGHT
        self.energy_cost_per_hour = energy_cost_per_hour
        self.carbon_sensitivity = carbon_sensitivity
        self.failure_probability = failure_probability if failure_probability is not None else config.DEFAULT_FAILURE_PROBABILITY
        self._rng = rng
        self._scenario_name = scenario_name
        self._recovery_attempts = 0
        self._failed_at: float | None = None

    def _transition(self, new_state: str, valid_from: set[str]) -> None:
        cur = self._state.get("device_state", "idle")
        if cur not in valid_from:
            msg = f"Invalid transition {cur} -> {new_state} for {self.agent_id}"
            logger.error(msg)
            raise ValueError(msg)
        self._state["device_state"] = new_state

    def _maybe_sample_failure(self) -> None:
        if self._state.get("device_state") == "failed":
            return
        if self._state.get("device_state") == "maintenance_required":
            return
        if self._rng.random() < self.failure_probability:
            self._enter_failed()

    def _enter_failed(self) -> None:
        self._state["device_state"] = "failed"
        self._failed_at = self.env.now
        self._recovery_attempts = 0
        m = Message.create(
            self.agent_id,
            "broadcast",
            MessageTypes.DeviceFailureNotice,
            {"device_id": self.agent_id, "device_type": self.device_type},
            self.env.now,
        )
        self.broadcast(m)
        self.env.process(self._recovery_process())

    def _recovery_process(self):
        while self._recovery_attempts < config.MAX_RECOVERY_ATTEMPTS:
            yield self.env.timeout(config.FAILURE_RECOVERY_TIMEOUT)
            self._recovery_attempts += 1
            success = self._rng.random() < 0.85
            if success:
                self._state["device_state"] = "idle"
                t_failed = self._failed_at or self.env.now
                dt = self.env.now - t_failed
                if self._metrics:
                    self._metrics.log_failure(
                        FailureEvent(
                            timestamp=self.env.now,
                            device_id=self.agent_id,
                            failure_type="random",
                            recovery_attempts=self._recovery_attempts,
                            recovery_succeeded=True,
                            time_in_failed_state=dt,
                        )
                    )
                self._failed_at = None
                rm = Message.create(
                    self.agent_id,
                    "broadcast",
                    MessageTypes.DeviceRecoveryNotice,
                    {"device_id": self.agent_id},
                    self.env.now,
                )
                self.broadcast(rm)
                return
        self._state["device_state"] = "maintenance_required"
        t_failed = self._failed_at or self.env.now
        dt = self.env.now - t_failed
        if self._metrics:
            self._metrics.log_failure(
                FailureEvent(
                    timestamp=self.env.now,
                    device_id=self.agent_id,
                    failure_type="random",
                    recovery_attempts=self._recovery_attempts,
                    recovery_succeeded=False,
                    time_in_failed_state=dt,
                )
            )


class ThermostatDeviceAgent(DeviceAgent):
    def __init__(
        self,
        agent_id: str,
        env: simpy.Environment,
        message_bus: Any,
        rng: np.random.Generator,
        metrics: MetricsCollector | None,
        device_optimal: float = 20.0,
        scenario_name: str = "default",
        failure_probability: float | None = None,
    ) -> None:
        super().__init__(
            agent_id,
            "thermostat",
            env,
            message_bus,
            rng,
            metrics,
            failure_probability=failure_probability,
            scenario_name=scenario_name,
        )
        self._device_optimal = device_optimal
        self._comfort_setpoint = float(device_optimal)
        self._state["current_temp"] = 18.0
        self._state["target_temp"] = device_optimal
        self._preferences: dict[str, dict[str, Any]] = {}
        self._last_carbon = float(config.CARBON_HOURLY_BASELINE[12])
        self._last_outdoor: float | None = None
        self._negotiation_in_progress = False
        self._last_resolved: float | None = None

    def _outdoor_heating_should_off(self) -> bool:
        if self._last_outdoor is None:
            return False
        return self._last_outdoor >= config.OUTDOOR_HEATING_OFF_CELSIUS

    def _applied_setpoint(self) -> float:
        """Comfort target from negotiation; forced to minimum when outdoors is warm enough."""
        if self._outdoor_heating_should_off():
            return float(config.THERMOSTAT_MIN)
        return float(self._comfort_setpoint)

    def _apply_outdoor_heating_rule(self) -> None:
        if self._state.get("device_state") in ("failed", "maintenance_required"):
            return
        applied = self._applied_setpoint()
        self._state["target_temp"] = applied
        outdoor_off = self._outdoor_heating_should_off()
        self.send(
            self.agent_id,
            Message.create(
                self.agent_id,
                self.agent_id,
                MessageTypes.ActuationCommand,
                {
                    "target_temperature": applied,
                    "outdoor_heating_off": outdoor_off,
                },
                self.env.now,
            ),
        )

    def _handle_message(self, msg: Message) -> None:
        if msg.msg_type == MessageTypes.CarbonIntensityUpdate:
            self._last_carbon = float(msg.payload.get("current", self._last_carbon))
        elif msg.msg_type == MessageTypes.WeatherUpdate:
            raw = msg.payload.get("outdoor_temp_c")
            if raw is not None:
                self._last_outdoor = float(raw)
                if not self._negotiation_in_progress:
                    self._apply_outdoor_heating_rule()
        elif msg.msg_type == MessageTypes.DeviceRecoveryNotice:
            if msg.payload.get("device_id") == self.agent_id:
                self._maybe_start_negotiation()
        elif msg.msg_type == MessageTypes.PreferenceDeclaration:
            pl = msg.payload
            pid = pl.get("person_id", msg.sender_id)
            prefs = pl.get("preferences", {})
            temp = float(prefs.get("temperature", self._state["target_temp"]))
            self._preferences[pid] = {
                "temperature": temp,
                "comfort_weight": float(pl.get("comfort_weight", config.DEFAULT_COMFORT_WEIGHT)),
                "is_home": bool(pl.get("is_home", True)),
            }
            self._maybe_start_negotiation()
        elif msg.msg_type in (
            MessageTypes.NegotiationAccept,
            MessageTypes.NegotiationCounter,
            MessageTypes.NegotiationReject,
        ):
            pass

    def _negotiation_response_types(self) -> frozenset[str]:
        return frozenset(
            {
                MessageTypes.NegotiationAccept,
                MessageTypes.NegotiationCounter,
                MessageTypes.NegotiationReject,
            }
        )

    def _conflict(self) -> bool:
        if len(self._preferences) < 2:
            return False
        temps = [v["temperature"] for v in self._preferences.values()]
        return max(temps) - min(temps) > 1e-3

    def _maybe_start_negotiation(self) -> None:
        if self._state.get("device_state") == "failed":
            return
        if self._state.get("device_state") == "maintenance_required":
            return
        if self._negotiation_in_progress:
            return
        if not self._conflict():
            return
        self._negotiation_in_progress = True
        self.env.process(self._negotiation_run())

    def _negotiation_run(self):
        try:
            participants = list(self._preferences.keys())
            original_values = [self._preferences[p]["temperature"] for p in participants]

            def eff_weight(pid: str) -> float:
                info = self._preferences[pid]
                return protocol.effective_person_weight(
                    float(info["comfort_weight"]),
                    bool(info["is_home"]),
                )

            weights = [eff_weight(p) for p in participants]
            current_values = list(original_values)
            iteration = 0
            converged_flag = False
            fallback_used = False
            final_value = float(np.mean(current_values))
            nid = str(uuid.uuid4())

            while iteration < config.MAX_ITERATIONS:
                iteration += 1
                proposal = protocol.combined_proposal(
                    current_values,
                    weights,
                    self._device_optimal,
                    self.device_weight,
                    self._last_carbon,
                )
                counters: dict[str, float] = {}
                for pid in participants:
                    m = Message.create(
                        self.agent_id,
                        pid,
                        MessageTypes.NegotiationProposal,
                        {
                            "negotiation_id": nid,
                            "round": iteration,
                            "proposed_value": proposal,
                            "device_id": self.agent_id,
                            "attribute": "temperature",
                        },
                        self.env.now,
                    )
                    self.send(pid, m)

                responses: dict[str, str] = {}
                deadline = self.env.now + config.NEGOTIATION_TIMEOUT
                while len(responses) < len(participants) and self.env.now < deadline:
                    remaining = max(0.0, deadline - self.env.now)
                    res, get_ev = yield from self.wait_inbox_or_timeout(remaining)
                    if get_ev in res:
                        rmsg = res[get_ev]
                        if rmsg.msg_type not in self._negotiation_response_types():
                            self._handle_message(rmsg)
                            continue
                        pl = rmsg.payload
                        if pl.get("negotiation_id") != nid:
                            self._handle_message(rmsg)
                            continue
                        sender = rmsg.sender_id
                        if rmsg.msg_type == MessageTypes.NegotiationAccept:
                            responses[sender] = "accept"
                        elif rmsg.msg_type == MessageTypes.NegotiationCounter:
                            responses[sender] = "counter"
                            counters[sender] = float(pl.get("counter_value", proposal))
                        else:
                            responses[sender] = "reject"
                    else:
                        break

                for pid in participants:
                    if pid not in responses:
                        responses[pid] = "timeout"

                new_values = []
                for i, pid in enumerate(participants):
                    r = responses[pid]
                    if r == "counter" and pid in counters:
                        new_values.append(counters[pid])
                    elif r in ("accept", "timeout"):
                        new_values.append(proposal)
                    else:
                        new_values.append(current_values[i])

                if protocol.converged(new_values):
                    converged_flag = True
                    final_value = proposal
                    break

                current_values = new_values

            if not converged_flag:
                fallback_used = config.FALLBACK_TO_UNWEIGHTED_AVERAGE
                if fallback_used:
                    final_value = float(
                        np.clip(
                            protocol.unweighted_average(original_values),
                            config.THERMOSTAT_MIN,
                            config.THERMOSTAT_MAX,
                        )
                    )
                else:
                    final_value = protocol.combined_proposal(
                        current_values,
                        weights,
                        self._device_optimal,
                        self.device_weight,
                        self._last_carbon,
                    )

            self._comfort_setpoint = final_value
            self._last_resolved = final_value
            applied = self._applied_setpoint()
            self._state["target_temp"] = applied

            sat = {
                p: protocol.satisfaction_score(
                    final_value,
                    self._preferences[p]["temperature"],
                )
                for p in participants
            }

            prefs_snapshot = {p: self._preferences[p]["temperature"] for p in participants}

            if self._metrics:
                self._metrics.log_negotiation(
                    NegotiationEvent(
                        timestamp=self.env.now,
                        scenario=self._scenario_name,
                        device_id=self.agent_id,
                        participants=participants,
                        iterations=iteration,
                        converged=converged_flag,
                        final_value=final_value,
                        satisfaction_scores=sat,
                        carbon_intensity=self._last_carbon,
                        fallback_used=fallback_used,
                        participant_preferences=prefs_snapshot,
                    )
                )

            self.broadcast(
                Message.create(
                    self.agent_id,
                    "broadcast",
                    MessageTypes.NegotiationResolved,
                    {
                        "final_value": final_value,
                        "device_id": self.agent_id,
                        "attribute": "temperature",
                        "iterations": iteration,
                        "converged": converged_flag,
                        "fallback_used": fallback_used,
                        "participant_preferences": prefs_snapshot,
                    },
                    self.env.now,
                )
            )

            if fallback_used and not converged_flag:
                self.broadcast(
                    Message.create(
                        self.agent_id,
                        "broadcast",
                        MessageTypes.NegotiationFailed,
                        {"device_id": self.agent_id, "final_value": final_value},
                        self.env.now,
                    )
                )

            self.send(
                self.agent_id,
                Message.create(
                    self.agent_id,
                    self.agent_id,
                    MessageTypes.ActuationCommand,
                    {
                        "target_temperature": applied,
                        "outdoor_heating_off": self._outdoor_heating_should_off(),
                    },
                    self.env.now,
                ),
            )
        finally:
            self._negotiation_in_progress = False

    def run(self):
        while True:
            deadline = self.env.now + 1.0
            while self.env.now < deadline:
                if self._negotiation_in_progress:
                    yield self.env.timeout(max(0.0, deadline - self.env.now))
                    break
                remaining = max(0.0, deadline - self.env.now)
                res, get_ev = yield from self.wait_inbox_or_timeout(remaining)
                if get_ev in res:
                    self._handle_message(res[get_ev])
                    yield from self.drain_inbox_burst(self._handle_message)
                else:
                    yield from self.drain_inbox_burst(self._handle_message)
                    break
            if self._state.get("device_state") not in ("failed", "maintenance_required"):
                self._maybe_sample_failure()


class DishwasherDeviceAgent(DeviceAgent):
    VALID = {
        "idle": {"scheduled", "running", "failed"},
        "scheduled": {"running", "idle", "failed"},
        "running": {"complete", "failed"},
        "complete": {"idle", "failed"},
        "failed": {"idle", "maintenance_required"},
        "maintenance_required": set(),
    }

    def __init__(
        self,
        agent_id: str,
        env: simpy.Environment,
        message_bus: Any,
        rng: np.random.Generator,
        metrics: MetricsCollector | None,
        scenario_name: str = "default",
    ) -> None:
        super().__init__(agent_id, "dishwasher", env, message_bus, rng, metrics, scenario_name=scenario_name)
        self._state["device_state"] = "idle"
        self._last_carbon = float(config.CARBON_HOURLY_BASELINE[18])
        self._scheduled_start: float | None = None
        self._energy_kwh = 0.0
        self._scheduled_day = -1

    def _go(self, new: str) -> None:
        cur = self._state["device_state"]
        if new not in self.VALID.get(cur, set()):
            raise ValueError(f"dishwasher invalid {cur} -> {new}")
        self._state["device_state"] = new

    def _handle_message(self, msg: Message) -> None:
        if msg.msg_type == MessageTypes.CarbonIntensityUpdate:
            self._last_carbon = float(msg.payload.get("current", self._last_carbon))
        elif msg.msg_type == MessageTypes.PreferenceDeclaration:
            if self._state["device_state"] == "idle":
                self.env.process(self._schedule_run())

    def _schedule_run(self):
        now = self.env.now
        day = int(now // config.MINUTES_PER_DAY)
        if self._scheduled_day == day:
            return
        self._scheduled_day = day
        minute_of_day = now % config.MINUTES_PER_DAY
        high_carbon = (
            config.CARBON_SPIKE_START_MINUTE <= minute_of_day <= config.CARBON_SPIKE_END_MINUTE
        ) or self._last_carbon > config.CARBON_HIGH_THRESHOLD

        if high_carbon and self.carbon_sensitivity > 0.3:
            start = ((now // config.MINUTES_PER_DAY) * config.MINUTES_PER_DAY) + config.DISHWASHER_LOW_CARBON_AFTER_MINUTE
            if start <= now:
                start += config.MINUTES_PER_DAY
        else:
            start = now + self._rng.integers(30, 90)

        self._scheduled_start = start
        self._go("scheduled")
        yield self.env.timeout(max(0.0, start - now))
        if self._state["device_state"] != "scheduled":
            return
        self._go("running")
        dur = 90.0
        self._energy_kwh += self.energy_cost_per_hour * (dur / 60.0)
        yield self.env.timeout(dur)
        self._go("complete")
        yield self.env.timeout(5)
        self._go("idle")

    def run(self):
        while True:
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
            self._maybe_sample_failure()


class ShowerDeviceAgent(DeviceAgent):
    """Shared tank: fixed drain per use, passive recharge; faster refill when grid carbon is below threshold."""

    def __init__(
        self,
        agent_id: str,
        env: simpy.Environment,
        message_bus: Any,
        rng: np.random.Generator,
        metrics: MetricsCollector | None,
        scenario_name: str = "default",
    ) -> None:
        super().__init__(agent_id, "shower", env, message_bus, rng, metrics, scenario_name=scenario_name)
        self._state["device_state"] = "idle"
        self._state["hot_water_available"] = 1.0
        self._last_carbon = float(config.CARBON_HOURLY_BASELINE[12])

    def _publish_hw(
        self,
        *,
        notify_feed: bool = False,
        feed_suffix: str = "",
        device_activity: str | None = None,
        shower_user_id: Any = _SHOWER_USER_OMIT,
    ) -> None:
        payload: dict[str, Any] = {
            "device_id": self.agent_id,
            "hot_water_fraction": float(self._state.get("hot_water_available", 0.0)),
            "notify_feed": notify_feed,
        }
        if shower_user_id is not _SHOWER_USER_OMIT:
            payload["shower_user_id"] = shower_user_id
        suf = feed_suffix.strip()
        if suf:
            payload["feed_suffix"] = suf
        if device_activity is not None:
            payload["device_activity"] = device_activity
        self.broadcast(
            Message.create(
                self.agent_id,
                "broadcast",
                MessageTypes.DeviceTelemetry,
                payload,
                self.env.now,
            )
        )

    def _handle_message(self, msg: Message) -> None:
        if msg.msg_type == MessageTypes.CarbonIntensityUpdate:
            self._last_carbon = float(msg.payload.get("current", self._last_carbon))
            return
        if msg.msg_type == MessageTypes.ArrivalNotice:
            self.env.process(self._use_shower(msg.sender_id))

    def _recharge_step(self, dt_minutes: float) -> None:
        ds = self._state.get("device_state")
        if ds in ("failed", "maintenance_required"):
            return
        grid_clean = self._last_carbon < float(config.CARBON_HIGH_THRESHOLD)
        mult = config.HOT_WATER_RECHARGE_GRID_CLEAN_MULTIPLIER if grid_clean else 1.0
        rate = float(config.HOT_WATER_RECHARGE_PER_MINUTE_BASE) * mult
        before = float(self._state.get("hot_water_available", 1.0))
        after = min(1.0, before + rate * dt_minutes)
        if abs(after - before) < 1e-12:
            return
        self._state["hot_water_available"] = after
        self._publish_hw()

    def _use_shower(self, person_id: str):
        if self._state.get("device_state") != "idle":
            return
        cost = float(config.HOT_WATER_DRAIN_PER_SHOWER)
        level = float(self._state.get("hot_water_available", 1.0))
        if level + 1e-12 < cost:
            self._publish_hw(
                notify_feed=True,
                feed_suffix=f"skip — depleted ({person_id})",
            )
            return
        self._state["device_state"] = "running"
        self._publish_hw(device_activity="running", shower_user_id=person_id)
        yield self.env.timeout(15.0)
        level = float(self._state.get("hot_water_available", 1.0))
        self._state["hot_water_available"] = max(0.0, level - cost)
        self._state["device_state"] = "idle"
        pct = int(round(100.0 * float(self._state["hot_water_available"])))
        self._publish_hw(
            device_activity="idle",
            notify_feed=True,
            feed_suffix=f"done · {person_id} · tank {pct}%",
            shower_user_id=None,
        )

    def run(self):
        while True:
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
            self._recharge_step(1.0)
            self._maybe_sample_failure()


class LightsDeviceAgent(DeviceAgent):
    def __init__(
        self,
        agent_id: str,
        env: simpy.Environment,
        message_bus: Any,
        rng: np.random.Generator,
        metrics: MetricsCollector | None,
        scenario_name: str = "default",
    ) -> None:
        super().__init__(agent_id, "lights", env, message_bus, rng, metrics, scenario_name=scenario_name)
        self._state["device_state"] = "idle"
        self._state["brightness"] = 0.0

    def _handle_message(self, msg: Message) -> None:
        if msg.msg_type == MessageTypes.PreferenceDeclaration:
            prefs = msg.payload.get("preferences", {})
            b = float(prefs.get("lighting", 0))
            self._state["brightness"] = b

    def run(self):
        while True:
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
            self._maybe_sample_failure()

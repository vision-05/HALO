"""Governor agents — smart devices with state machines and negotiation."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Generator
from typing import Any

import numpy as np
import simpy

from halo_simulation import config
from halo_simulation.agents.base_agent import BaseAgent
from halo_simulation.external.llm_client import LLMClient
from halo_simulation.metrics.collector import (
    FailureEvent,
    LLMApiCallEvent,
    MetricsCollector,
    NegotiationEvent,
)
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

    def _negotiation_response_types(self) -> frozenset[str]:
        return frozenset(
            {
                MessageTypes.NegotiationAccept,
                MessageTypes.NegotiationCounter,
                MessageTypes.NegotiationReject,
            }
        )

    def _yield_collect_negotiation_round_responses(
        self,
        negotiation_id: str,
        proposal: float,
        participants: list[str],
        *,
        indefinite_wait: bool,
    ) -> Generator[Any, Any, tuple[dict[str, str], dict[str, float]]]:
        """Gather accept/counter/reject for one proposal round; optional sim-time deadline (see config)."""
        responses: dict[str, str] = {}
        counters: dict[str, float] = {}
        n_needed = len(participants)
        types = self._negotiation_response_types()
        part_set = frozenset(participants)

        if indefinite_wait:
            while len(responses) < n_needed:
                rmsg: Message = yield self.inbox.get()
                if rmsg.msg_type not in types:
                    self._handle_message(rmsg)
                    continue
                pl = rmsg.payload or {}
                if pl.get("negotiation_id") != negotiation_id:
                    self._handle_message(rmsg)
                    continue
                sender = rmsg.sender_id
                if sender not in part_set:
                    self._handle_message(rmsg)
                    continue
                if rmsg.msg_type == MessageTypes.NegotiationAccept:
                    responses[sender] = "accept"
                elif rmsg.msg_type == MessageTypes.NegotiationCounter:
                    responses[sender] = "counter"
                    counters[sender] = float(pl.get("counter_value", proposal))
                else:
                    responses[sender] = "reject"
        else:
            deadline = self.env.now + config.NEGOTIATION_TIMEOUT
            while len(responses) < n_needed and self.env.now < deadline:
                remaining = max(0.0, deadline - self.env.now)
                res, get_ev = yield from self.wait_inbox_or_timeout(remaining)
                if get_ev not in res:
                    break
                rmsg = res[get_ev]
                if rmsg.msg_type not in types:
                    self._handle_message(rmsg)
                    continue
                pl = rmsg.payload or {}
                if pl.get("negotiation_id") != negotiation_id:
                    self._handle_message(rmsg)
                    continue
                sender = rmsg.sender_id
                if sender not in part_set:
                    self._handle_message(rmsg)
                    continue
                if rmsg.msg_type == MessageTypes.NegotiationAccept:
                    responses[sender] = "accept"
                elif rmsg.msg_type == MessageTypes.NegotiationCounter:
                    responses[sender] = "counter"
                    counters[sender] = float(pl.get("counter_value", proposal))
                else:
                    responses[sender] = "reject"

        return responses, counters


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
        self._cost_pressure_boost = 0.0
        self._cost_pressure_until = 0.0
        self._temp_ceiling_override = float(config.THERMOSTAT_MAX)
        self._ceiling_override_until = 0.0
        self._temp_floor_override = float(config.THERMOSTAT_MIN)
        self._floor_override_until = 0.0

    def _effective_device_weight(self) -> float:
        boost = 0.0
        if self.env.now < self._cost_pressure_until:
            boost = self._cost_pressure_boost
        return min(float(self.device_weight) + boost, 0.9)

    def _maybe_fast_reweight_setpoint(self) -> None:
        """Apply one weighted proposal from current prefs (presence/weights) without negotiation rounds."""
        if self._state.get("device_state") in ("failed", "maintenance_required"):
            return
        if self._negotiation_in_progress:
            return
        if len(self._preferences) < 1:
            return

        participants = list(self._preferences.keys())

        def eff_weight(pid: str) -> float:
            info = self._preferences[pid]
            return protocol.effective_person_weight(
                float(info["comfort_weight"]),
                bool(info["is_home"]),
            )

        weights = [eff_weight(p) for p in participants]
        values = [float(self._preferences[p]["temperature"]) for p in participants]
        proposal = protocol.combined_proposal(
            values,
            weights,
            self._device_optimal,
            self._effective_device_weight(),
            self._last_carbon,
        )
        self._comfort_setpoint = self._clamp_setpoint(float(proposal))
        self._state["target_temp"] = self._applied_setpoint()
        applied = float(self._state["target_temp"])
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

    def _clamp_setpoint(self, value: float) -> float:
        low = float(config.THERMOSTAT_MIN)
        high = float(config.THERMOSTAT_MAX)
        if self.env.now < self._floor_override_until:
            low = max(low, float(self._temp_floor_override))
        if self.env.now < self._ceiling_override_until:
            high = min(high, float(self._temp_ceiling_override))
        return max(low, min(high, float(value)))

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
            if not self._negotiation_in_progress and len(self._preferences) >= 1:
                self._maybe_fast_reweight_setpoint()
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
            if not self._negotiation_in_progress and len(self._preferences) >= 1 and not self._conflict():
                self._maybe_fast_reweight_setpoint()
        elif msg.msg_type == MessageTypes.CostPressureUpdate:
            sev = str(msg.payload.get("severity", "low")).lower()
            if sev == "high":
                self._cost_pressure_boost = 0.4
                self._cost_pressure_until = self.env.now + 120.0
            elif sev == "medium":
                self._cost_pressure_boost = 0.2
                self._cost_pressure_until = self.env.now + 60.0
            else:
                self._cost_pressure_boost = 0.1
                self._cost_pressure_until = self.env.now + 30.0
            logger.info(
                "Thermostat: cost pressure (%s), device_weight boost %.2f",
                sev,
                self._cost_pressure_boost,
            )
            if not self._negotiation_in_progress and len(self._preferences) >= 1:
                self._maybe_fast_reweight_setpoint()
        elif msg.msg_type == MessageTypes.WeatherForecastAlert:
            pl = msg.payload or {}
            hourly = pl.get("hourly")
            temps: list[float] = []
            if isinstance(hourly, dict):
                raw = hourly.get("temperature_2m")
                if isinstance(raw, list):
                    temps = [float(x) for x in raw[:24] if x is not None]
            elif isinstance(hourly, list):
                temps = [float(x) for x in hourly[:24]]
            if temps:
                upcoming_max = max(temps[:6])
                upcoming_min = min(temps[:6])
                if upcoming_max > 30.0:
                    self._temp_ceiling_override = 22.0
                    self._ceiling_override_until = self.env.now + 180.0
                    logger.info("Thermostat: heatwave forecast — ceiling 22°C")
                elif upcoming_min < 2.0:
                    self._temp_floor_override = 17.0
                    self._floor_override_until = self.env.now + 180.0
                    logger.info("Thermostat: cold snap forecast — floor 17°C")
        elif msg.msg_type in (
            MessageTypes.NegotiationAccept,
            MessageTypes.NegotiationCounter,
            MessageTypes.NegotiationReject,
        ):
            pass

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
                    self._effective_device_weight(),
                    self._last_carbon,
                )
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

                # Finite wait only: if person_cli is in the round with manual_negotiation, indefinite
                # wait would block the thermostat forever while Alice/Bob have already replied.
                responses, counters = yield from self._yield_collect_negotiation_round_responses(
                    nid,
                    proposal,
                    participants,
                    indefinite_wait=False,
                )

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
                        self._effective_device_weight(),
                        self._last_carbon,
                    )

            final_value = self._clamp_setpoint(float(final_value))
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
                        preference_attribute="temperature",
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
        failure_probability: float | None = None,
    ) -> None:
        super().__init__(
            agent_id,
            "dishwasher",
            env,
            message_bus,
            rng,
            metrics,
            failure_probability=failure_probability,
            scenario_name=scenario_name,
        )
        self._state["device_state"] = "idle"
        self._last_carbon = float(config.CARBON_HOURLY_BASELINE[18])
        self._last_carbon_forecast: list[float] = []
        self._energy_kwh = 0.0
        self._pending: dict[str, dict[str, float]] = {}
        self._person_snapshot: dict[str, dict[str, Any]] = {}
        self._negotiation_in_progress = False
        self._eval_busy = False
        self._llm_client = LLMClient(api_key=config.anthropic_api_key())
        self._declined_retry_seq = 0

    def _go(self, new: str) -> None:
        cur = self._state["device_state"]
        if new not in self.VALID.get(cur, set()):
            raise ValueError(f"dishwasher invalid {cur} -> {new}")
        self._state["device_state"] = new
        # Stream/UI: dishwasher FSM does not otherwise hit the message bus. DeviceTelemetry is special-cased
        # in MessageBus.broadcast (no inbox fan-out) but StreamingMessageBus still emits agent_state rows.
        if new != "complete":
            self.broadcast(
                Message.create(
                    self.agent_id,
                    "broadcast",
                    MessageTypes.DeviceTelemetry,
                    {
                        "device_id": self.agent_id,
                        "device_activity": new,
                        "notify_feed": False,
                    },
                    self.env.now,
                )
            )

    def _maybe_sample_failure(self) -> None:
        """Main ``run()`` interleaves with nested ``_try_evaluate_pending`` / defer / run processes.
        Random failure must not fire while ``_eval_busy`` or the post-defer guard aborts before
        ``scheduled``/``running`` (UI stuck on Wait with no Run).
        """
        if self._eval_busy:
            return
        super()._maybe_sample_failure()

    def _dishwasher_device_weight(self) -> float:
        return float(self.device_weight)

    def _person_weight(self, pid: str) -> float:
        meta = self._person_snapshot.get(pid)
        if not meta:
            return protocol.effective_person_weight(float(config.DEFAULT_COMFORT_WEIGHT), True)
        return protocol.effective_person_weight(
            float(meta.get("comfort_weight", config.DEFAULT_COMFORT_WEIGHT)),
            bool(meta.get("is_home", True)),
        )

    def _handle_message(self, msg: Message) -> None:
        if msg.msg_type == MessageTypes.CarbonIntensityUpdate:
            pl = msg.payload or {}
            self._last_carbon = float(pl.get("current", self._last_carbon))
            fc = pl.get("forecast_4h")
            if isinstance(fc, list):
                self._last_carbon_forecast = [float(x) for x in fc[:8] if x is not None]
            if (
                self._state["device_state"] == "idle"
                and self._pending
                and not self._negotiation_in_progress
            ):
                self.env.process(self._evaluate_pending_pipeline())
        elif msg.msg_type == MessageTypes.PreferenceDeclaration:
            pl = msg.payload or {}
            pid = str(pl.get("person_id", msg.sender_id))
            self._person_snapshot[pid] = {
                "comfort_weight": float(pl.get("comfort_weight", config.DEFAULT_COMFORT_WEIGHT)),
                "is_home": bool(pl.get("is_home", True)),
            }
        elif msg.msg_type == MessageTypes.DishwasherRunRequest:
            pl = msg.payload or {}
            rid = str(pl.get("requester_id", msg.sender_id))
            raw_u = pl.get("urgency", 0.5)
            try:
                u = float(raw_u)
            except (TypeError, ValueError):
                u = 0.5
            self._pending[rid] = {"urgency": max(0.0, min(1.0, u)), "at": float(self.env.now)}
            if self._state["device_state"] == "idle" and not self._negotiation_in_progress:
                self.env.process(self._evaluate_pending_pipeline())

    def _evaluate_pending_pipeline(self) -> Generator[Any, Any, None]:
        yield from self._try_evaluate_pending()

    def _pending_oldest_age_minutes(self) -> float:
        if not self._pending:
            return 0.0
        oldest = min(float(v["at"]) for v in self._pending.values())
        return max(0.0, float(self.env.now) - oldest)

    def _schedule_declined_retry(self) -> None:
        self._declined_retry_seq += 1
        seq = self._declined_retry_seq
        self.env.process(self._declined_retry_after_timeout(seq))

    def _declined_retry_after_timeout(self, seq: int) -> Generator[Any, Any, None]:
        delay = max(1.0, float(getattr(config, "DISHWASHER_DECLINED_RETRY_SIM_MINUTES", 15.0)))
        yield self.env.timeout(delay)
        if seq != self._declined_retry_seq:
            return
        if self._state.get("device_state") != "idle":
            return
        if not self._pending:
            return
        if self._negotiation_in_progress:
            return
        self.env.process(self._evaluate_pending_pipeline())

    def _try_evaluate_pending(self) -> Generator[Any, Any, None]:
        if self._eval_busy:
            return
        self._eval_busy = True
        try:
            if self._state.get("device_state") != "idle":
                return
            if self._negotiation_in_progress:
                return
            if not self._pending:
                return
            age_min = self._pending_oldest_age_minutes()
            override_after = float(getattr(config, "DISHWASHER_APPROVE_FALSE_OVERRIDE_AFTER_SIM_MIN", 90.0))
            if age_min >= override_after:
                t0 = time.perf_counter()
                decision = self._heuristic_decision()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self._emit_dishwasher_schedule_decision(
                    decision,
                    source="heuristic_pending_age",
                    latency_ms=latency_ms,
                )
                logger.info(
                    "Washing machine %s: pending %.0f sim min — using heuristic schedule (bypass LLM decline streak)",
                    self.agent_id,
                    age_min,
                )
            elif config.DISHWASHER_USE_LLM_SCHEDULE:
                decision = self._llm_or_heuristic_decision()
            else:
                t0 = time.perf_counter()
                decision = self._heuristic_decision()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self._emit_dishwasher_schedule_decision(
                    decision,
                    source="heuristic_no_llm",
                    latency_ms=latency_ms,
                )
                logger.info(
                    "Washing machine %s: schedule gate heuristic (LLM off) pending=%s",
                    self.agent_id,
                    sorted(self._pending.keys()),
                )
            defer = float(decision.get("defer_minutes", 0.0))
            defer = max(0.0, min(float(config.DISHWASHER_DEFER_MINUTES_MAX), defer))
            if not bool(decision.get("approve", False)):
                reason = decision.get("reason", "")
                logger.info(
                    "Washing machine %s: schedule declined (approve=false) — reason=%r pending_age_sim_min=%.1f",
                    self.agent_id,
                    reason,
                    age_min,
                )
                self._schedule_declined_retry()
                return
            participants_snapshot = list(self._pending.keys())
            if not participants_snapshot:
                return
            if config.DISHWASHER_USE_DELAY_NEGOTIATION:
                self._negotiation_in_progress = True
                try:
                    yield from self._negotiation_delay_run(participants_snapshot, defer)
                finally:
                    self._negotiation_in_progress = False
            else:
                yield from self._physical_run_after_defer(defer, participants_snapshot)
        finally:
            self._eval_busy = False

    def _physical_run_after_defer(
        self, defer_minutes: float, served_ids: list[str]
    ) -> Generator[Any, Any, None]:
        if defer_minutes > 0:
            yield self.env.timeout(defer_minutes)
        if self._state["device_state"] != "idle":
            return
        self._go("scheduled")
        yield self.env.timeout(0.0)
        if self._state["device_state"] != "scheduled":
            return
        self._go("running")
        dur = 90.0
        self._energy_kwh += self.energy_cost_per_hour * (dur / 60.0)
        yield self.env.timeout(dur)
        if self._state["device_state"] == "running":
            self._go("complete")
            yield self.env.timeout(5.0)
            self._go("idle")

        if self._state["device_state"] == "idle":
            for pid in served_ids:
                self._pending.pop(pid, None)
            if self._pending and not self._negotiation_in_progress:
                self.env.process(self._evaluate_pending_pipeline())

    def _negotiation_delay_run(
        self, participants_snapshot: list[str], device_delay: float
    ) -> Generator[Any, Any, None]:
        participants = list(participants_snapshot)
        original_values = [0.0] * len(participants)
        iteration = 0
        converged_flag = False
        fallback_used = False
        final_value = float(device_delay)
        nid = str(uuid.uuid4())
        current_values = list(original_values)
        weights = [self._person_weight(p) for p in participants]

        while iteration < config.MAX_ITERATIONS:
            iteration += 1
            proposal = protocol.combined_proposal(
                current_values,
                weights,
                float(device_delay),
                self._dishwasher_device_weight(),
                self._last_carbon,
                clip_lo=0.0,
                clip_hi=float(config.DISHWASHER_DEFER_MINUTES_MAX),
            )
            proposal = max(0.0, min(float(config.DISHWASHER_DEFER_MINUTES_MAX), float(proposal)))
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
                        "attribute": "dishwasher_delay",
                    },
                    self.env.now,
                )
                self.send(pid, m)

            # Never block on indefinite inbox wait: person_cli uses manual_negotiation (no auto-reply),
            # which previously stalled the whole sim. Use finite NEGOTIATION_TIMEOUT; missing → timeout → accept.
            responses, counters = yield from self._yield_collect_negotiation_round_responses(
                nid,
                proposal,
                participants,
                indefinite_wait=False,
            )

            for pid in participants:
                if pid not in responses:
                    responses[pid] = "timeout"

            new_values: list[float] = []
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
                        0.0,
                        float(config.DISHWASHER_DEFER_MINUTES_MAX),
                    )
                )
            else:
                final_value = protocol.combined_proposal(
                    current_values,
                    weights,
                    float(device_delay),
                    self._dishwasher_device_weight(),
                    self._last_carbon,
                    clip_lo=0.0,
                    clip_hi=float(config.DISHWASHER_DEFER_MINUTES_MAX),
                )
        final_value = max(0.0, min(float(config.DISHWASHER_DEFER_MINUTES_MAX), float(final_value)))

        if self._metrics:
            sat = {
                p: protocol.satisfaction_score(final_value, 0.0, preference_range=180.0)
                for p in participants
            }
            self._metrics.log_negotiation(
                NegotiationEvent(
                    timestamp=self.env.now,
                    scenario=self._scenario_name,
                    device_id=self.agent_id,
                    participants=participants,
                    iterations=iteration,
                    converged=converged_flag,
                    final_value=float(final_value),
                    satisfaction_scores=sat,
                    carbon_intensity=self._last_carbon,
                    fallback_used=fallback_used,
                    participant_preferences={p: 0.0 for p in participants},
                    preference_attribute="dishwasher_delay",
                )
            )

        self.broadcast(
            Message.create(
                self.agent_id,
                "broadcast",
                MessageTypes.NegotiationResolved,
                {
                    "final_value": float(final_value),
                    "device_id": self.agent_id,
                    "attribute": "dishwasher_delay",
                    "iterations": iteration,
                    "converged": converged_flag,
                    "fallback_used": fallback_used,
                },
                self.env.now,
            )
        )

        for pid in participants_snapshot:
            self._pending.pop(pid, None)

        yield from self._physical_run_after_defer(float(final_value), [])

    def _emit_dishwasher_schedule_decision(
        self,
        decision: dict[str, Any],
        *,
        source: str,
        latency_ms: float,
    ) -> None:
        """Log + metrics (StreamingMetricsCollector → SSE + llm-inspector timeline)."""
        approve = bool(decision.get("approve", False))
        try:
            defer = float(decision.get("defer_minutes", 0.0))
        except (TypeError, ValueError):
            defer = 0.0
        reason = str(decision.get("reason", "") or "").strip() or "—"
        summary = (
            f"approve={approve} defer_minutes={defer:.2f} reason={reason!r} source={source} "
            f"carbon_gco2kwh={self._last_carbon:.0f} pending={sorted(self._pending.keys())}"
        )
        logger.info("Washing machine %s: schedule decision — %s", self.agent_id, summary)
        if self._metrics is None:
            return
        self._metrics.log_llm_api_call(
            LLMApiCallEvent(
                timestamp=float(self.env.now),
                api_id="dishwasher_schedule",
                success=True,
                observation_summary=summary,
                severity=str(protocol.carbon_band(self._last_carbon)).lower(),
                halo_message_type="DishwasherScheduleDecision",
                latency_ms=float(latency_ms),
            )
        )

    def _llm_or_heuristic_decision(self) -> dict[str, Any]:
        minute_of_day = float(self.env.now % config.MINUTES_PER_DAY)
        band = protocol.carbon_band(self._last_carbon)
        pending_lines = [
            f"- {rid}: urgency={self._pending[rid]['urgency']:.2f}, requested_at_sim_min={self._pending[rid]['at']:.0f}"
            for rid in sorted(self._pending.keys())
        ]
        fc = self._last_carbon_forecast or []
        fc_txt = ", ".join(f"{x:.0f}" for x in fc[:4]) if fc else "n/a"
        prompt = (
            "You are scheduling a residential washing machine to reduce grid carbon impact while respecting requests.\n"
            "Reply with JSON ONLY:\n"
            f'{{"approve": <bool>, "defer_minutes": <number 0-{int(config.DISHWASHER_DEFER_MINUTES_MAX)}>, "reason": "<short string>"}}\n'
            "approve=false: only if you cannot justify any defer_minutes this cycle (use sparingly).\n"
            "defer_minutes: simulated minutes from NOW until the machine may START (0 = as soon as allowed).\n"
            f"Minute-of-day (0=midnight): {minute_of_day:.0f}. Current grid carbon: {self._last_carbon:.0f} gCO2/kWh ({band}). "
            f"Forecast sample gCO2/kWh: {fc_txt}.\n"
            "Pending requests:\n"
            + "\n".join(pending_lines)
            + "\nIf requests are already waiting, prefer approve=true. Use the **smallest** defer_minutes that still "
            "reflects carbon (often 30–90); only approach the max when carbon is extreme. "
            "Use approve=false only when deferring is impossible.\n"
        )
        t0 = time.perf_counter()
        raw = self._llm_client.complete_json(prompt, max_tokens=320, timeout=12.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not isinstance(raw, dict):
            d = self._heuristic_decision()
            self._emit_dishwasher_schedule_decision(d, source="heuristic_fallback", latency_ms=latency_ms)
            return d
        approve = bool(raw.get("approve", False))
        try:
            defer = float(raw.get("defer_minutes", 0.0))
        except (TypeError, ValueError):
            defer = 0.0
        reason = str(raw.get("reason", "")).strip() or "llm"
        d = {"approve": approve, "defer_minutes": defer, "reason": reason}
        self._emit_dishwasher_schedule_decision(d, source="llm", latency_ms=latency_ms)
        return d

    def _heuristic_decision(self) -> dict[str, Any]:
        now = self.env.now
        minute_of_day = now % config.MINUTES_PER_DAY
        band = protocol.carbon_band(self._last_carbon)
        in_spike = config.CARBON_SPIKE_START_MINUTE <= minute_of_day <= config.CARBON_SPIKE_END_MINUTE
        high_carbon = in_spike or band == "high"

        if high_carbon and self.carbon_sensitivity > 0.3:
            # On truly dirty grid: nudge toward the next low-carbon window, capped.
            day_start = (now // config.MINUTES_PER_DAY) * config.MINUTES_PER_DAY
            start = day_start + float(config.DISHWASHER_LOW_CARBON_AFTER_MINUTE)
            if start <= now:
                start += config.MINUTES_PER_DAY
            defer_raw = max(0.0, start - now)
            cap = float(getattr(config, "DISHWASHER_DEFER_DIRTY_GRID_CAP_MINUTES", 90.0))
            defer = min(defer_raw, cap)
            reason = "heuristic_defer_dirty_grid"
        elif band == "medium":
            # Medium carbon: short token delay (5–15 sim min) to demonstrate deferral logic.
            defer = float(self._rng.integers(5, 16))
            reason = "heuristic_medium_carbon"
        else:
            # Low carbon or forecast improving: run as soon as possible.
            defer = 0.0
            reason = "heuristic_low_carbon_run_now"

        # Multiple requesters: one cycle serves everyone; nudge defer down slightly on urgency.
        n = len(self._pending)
        if n >= 2 and defer > 0:
            max_u = max(float(x["urgency"]) for x in self._pending.values())
            factor = 1.0 - min(0.25, 0.06 * float(n - 1) + 0.12 * max(0.0, max_u - 0.5))
            defer = max(0.0, defer * factor)
            reason = f"{reason}_n{n}"

        return {"approve": True, "defer_minutes": defer, "reason": reason}

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
        self._last_auto_preheat_block_notice = -1e12
        self._preheat_busy = False
        self._shower_preferences: dict[str, dict[str, Any]] = {}
        self._negotiation_in_progress = False

    def _shower_indefinite_wait_for_round(self, participants: list[str]) -> bool:
        ids = set(config.NEGOTIATION_INDEFINITE_WAIT_AGENT_IDS)
        if not any(pid in ids for pid in participants):
            return False
        for pid in participants:
            if pid not in ids:
                continue
            info = self._shower_preferences.get(pid)
            if info is None:
                return False
            if not bool(info.get("is_home", True)):
                return False
        return True

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

    def _water_notice(self, detail: str) -> None:
        self.broadcast(
            Message.create(
                self.agent_id,
                "broadcast",
                MessageTypes.WaterServiceNotice,
                {"detail": detail.strip()},
                self.env.now,
            )
        )

    def _maybe_emit_auto_preheat_block(self) -> None:
        gap = float(config.FUSED_AUTO_PREHEAT_BLOCK_NOTICE_COOLDOWN_MIN)
        if self.env.now - self._last_auto_preheat_block_notice < gap:
            return
        self._last_auto_preheat_block_notice = self.env.now
        cap = int(config.CARBON_HIGH_THRESHOLD)
        cur = int(self._last_carbon)
        self._water_notice(f"Auto preheat held — grid {cur} gCO2/kWh (needs under {cap}).")

    def _handle_message(self, msg: Message) -> None:
        if msg.msg_type == MessageTypes.PreferenceDeclaration:
            pl = msg.payload
            pid = str(pl.get("person_id", msg.sender_id))
            prefs = pl.get("preferences", {})
            temp = float(prefs.get("temperature", 21.0))
            explicit_shower = prefs.get("shower_minutes")
            if explicit_shower is not None:
                shower_min = float(
                    np.clip(
                        float(explicit_shower),
                        config.SHOWER_DURATION_MIN_MINUTES,
                        config.SHOWER_DURATION_MAX_MINUTES,
                    )
                )
            else:
                shower_min = float(protocol.shower_minutes_from_comfort_temp(temp))
            self._shower_preferences[pid] = {
                "shower_minutes": shower_min,
                "comfort_weight": float(pl.get("comfort_weight", config.DEFAULT_COMFORT_WEIGHT)),
                "is_home": bool(pl.get("is_home", True)),
            }
            return
        if msg.msg_type == MessageTypes.CarbonIntensityUpdate:
            self._last_carbon = float(msg.payload.get("current", self._last_carbon))
            return
        if msg.msg_type == MessageTypes.ArrivalNotice:
            self.env.process(self._shower_intent_pipeline(msg.sender_id))
            return
        if msg.msg_type == MessageTypes.WaterShowerIntent:
            who = str(msg.payload.get("initiator", msg.sender_id))
            self.env.process(self._shower_intent_pipeline(who))
            return
        if msg.msg_type == MessageTypes.WaterPreheatIntent:
            who = str(msg.payload.get("initiator", msg.sender_id))
            self.env.process(self._preheat_for_person(who))
            return

    def _emit_synthetic_shower_intent(self, initiator: str) -> None:
        m = Message.create(
            initiator,
            self.agent_id,
            MessageTypes.WaterShowerIntent,
            {"initiator": initiator},
            self.env.now,
        )
        self.send(self.agent_id, m)

    def _fused_demo_shower_negotiations(self) -> Generator[Any, Any, None]:
        """Simulated shower requests so fused runs show shower-duration negotiation in the feed."""
        if self._scenario_name != "fused":
            return
        yield self.env.timeout(420.0)
        self._emit_synthetic_shower_intent("person_alice")
        yield self.env.timeout(1380.0)
        self._emit_synthetic_shower_intent("person_bob")
        yield self.env.timeout(1320.0)
        self._emit_synthetic_shower_intent("person_cli")

    def _ensure_participant_pref(self, pid: str) -> None:
        if pid not in self._shower_preferences:
            self._shower_preferences[pid] = {
                "shower_minutes": (float(config.SHOWER_DURATION_MIN_MINUTES) + float(config.SHOWER_DURATION_MAX_MINUTES))
                / 2.0,
                "comfort_weight": float(config.DEFAULT_COMFORT_WEIGHT),
                "is_home": True,
            }

    def _shower_intent_pipeline(self, person_id: str) -> Generator[Any, Any, None]:
        if self._preheat_busy:
            self._water_notice("Shower deferred — preheat ramp active.")
            return
        if self._negotiation_in_progress:
            self._water_notice("Shower deferred — another negotiation is in progress.")
            return
        if self._state.get("device_state") != "idle":
            return
        minutes = yield from self._negotiate_shower_minutes(person_id)
        if minutes <= 0:
            return
        yield from self._execute_shower_cycle(person_id, minutes)

    def _negotiate_shower_minutes(self, initiator: str) -> Generator[Any, Any, float]:
        self._ensure_participant_pref(initiator)
        participants = sorted(
            pid for pid, info in self._shower_preferences.items() if info.get("is_home", True)
        )
        lo = float(config.SHOWER_DURATION_MIN_MINUTES)
        hi = float(config.SHOWER_DURATION_MAX_MINUTES)
        if len(participants) < 2:
            m = float(self._shower_preferences.get(initiator, {}).get("shower_minutes", 15.0))
            return float(max(lo, min(hi, m)))

        mins_list = [float(self._shower_preferences[p]["shower_minutes"]) for p in participants]
        if max(mins_list) - min(mins_list) <= 0.51:
            return float(np.clip(float(np.mean(mins_list)), lo, hi))

        self._negotiation_in_progress = True
        try:
            return float((yield from self._run_shower_negotiation_rounds(participants)))
        finally:
            self._negotiation_in_progress = False

    def _run_shower_negotiation_rounds(self, participants: list[str]) -> Generator[Any, Any, float]:
        clip_lo = float(config.SHOWER_DURATION_MIN_MINUTES)
        clip_hi = float(config.SHOWER_DURATION_MAX_MINUTES)
        device_optimal = float(config.SHOWER_DEVICE_OPTIMAL_MINUTES)

        def eff_weight(pid: str) -> float:
            info = self._shower_preferences[pid]
            return protocol.effective_person_weight(
                float(info["comfort_weight"]),
                bool(info.get("is_home", True)),
            )

        weights = [eff_weight(p) for p in participants]
        original_values = [float(self._shower_preferences[p]["shower_minutes"]) for p in participants]
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
                device_optimal,
                float(self.device_weight),
                self._last_carbon,
                clip_lo=clip_lo,
                clip_hi=clip_hi,
            )
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
                        "attribute": "shower_minutes",
                    },
                    self.env.now,
                )
                self.send(pid, m)

            responses, counters = yield from self._yield_collect_negotiation_round_responses(
                nid,
                proposal,
                participants,
                indefinite_wait=self._shower_indefinite_wait_for_round(participants),
            )

            for pid in participants:
                if pid not in responses:
                    responses[pid] = "timeout"

            new_values = []
            for i, pid in enumerate(participants):
                r = responses[pid]
                if r == "counter" and pid in counters:
                    new_values.append(float(np.clip(counters[pid], clip_lo, clip_hi)))
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
                    np.clip(protocol.unweighted_average(original_values), clip_lo, clip_hi)
                )
            else:
                final_value = protocol.combined_proposal(
                    current_values,
                    weights,
                    device_optimal,
                    float(self.device_weight),
                    self._last_carbon,
                    clip_lo=clip_lo,
                    clip_hi=clip_hi,
                )

        final_value = float(np.clip(float(final_value), clip_lo, clip_hi))

        sat = {
            p: protocol.satisfaction_score(
                final_value,
                float(self._shower_preferences[p]["shower_minutes"]),
                preference_range=12.0,
            )
            for p in participants
        }
        prefs_snapshot = {p: float(self._shower_preferences[p]["shower_minutes"]) for p in participants}

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
                    preference_attribute="shower_minutes",
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
                    "attribute": "shower_minutes",
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
                    {"device_id": self.agent_id, "final_value": final_value, "attribute": "shower_minutes"},
                    self.env.now,
                )
            )

        return final_value

    def _execute_shower_cycle(self, person_id: str, duration_minutes: float) -> Generator[Any, Any, None]:
        if self._state.get("device_state") != "idle":
            return
        dur = float(np.clip(duration_minutes, float(config.SHOWER_DURATION_MIN_MINUTES), float(config.SHOWER_DURATION_MAX_MINUTES)))
        base_cost = float(config.HOT_WATER_DRAIN_PER_SHOWER)
        cost = float(np.clip(base_cost * (dur / 15.0), 0.0, 1.0))
        level = float(self._state.get("hot_water_available", 1.0))
        if level + 1e-12 < cost:
            self._publish_hw(
                notify_feed=True,
                feed_suffix=f"skip — need {int(round(cost * 100))}% tank for {dur:.1f}m shower ({person_id})",
            )
            return
        self._state["device_state"] = "running"
        self._publish_hw(device_activity="running", shower_user_id=person_id)
        yield self.env.timeout(dur)
        level = float(self._state.get("hot_water_available", 1.0))
        self._state["hot_water_available"] = max(0.0, level - cost)
        self._state["device_state"] = "idle"
        pct = int(round(100.0 * float(self._state["hot_water_available"])))
        self._publish_hw(
            device_activity="idle",
            notify_feed=True,
            feed_suffix=f"done · {person_id} · {dur:.1f}m · tank {pct}%",
            shower_user_id=None,
        )
        self._try_fused_auto_preheat_if_eligible()

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

    def _try_fused_auto_preheat_if_eligible(self) -> None:
        """Fused only: refill ramp when tank is below trigger, unit idle, grid carbon under cap."""
        if self._scenario_name != "fused":
            return
        if self._preheat_busy:
            return
        if self._state.get("device_state") != "idle":
            return
        tank = float(self._state.get("hot_water_available", 1.0))
        if tank >= float(config.FUSED_AUTO_PREHEAT_TANK_TRIGGER):
            return
        if self._last_carbon >= float(config.CARBON_HIGH_THRESHOLD):
            self._maybe_emit_auto_preheat_block()
            return
        self.env.process(self._preheat_ramp("auto", source_auto=True))

    def _preheat_ramp(self, tag: str, *, source_auto: bool):
        if self._preheat_busy:
            if not source_auto:
                self._water_notice(f"Preheat held ({tag}) — ramp already running.")
            return
        cap = float(config.CARBON_HIGH_THRESHOLD)
        if self._last_carbon >= cap:
            if source_auto:
                self._maybe_emit_auto_preheat_block()
            else:
                self._water_notice(
                    f"Preheat not started ({tag}) — grid {int(self._last_carbon)} gCO2/kWh, cap {int(cap)}."
                )
            return
        if self._state.get("device_state") != "idle":
            if not source_auto:
                self._water_notice(f"Preheat not started ({tag}) — unit busy (shower or fault).")
            return
        target = float(config.FUSED_AUTO_PREHEAT_TANK_TARGET)
        if float(self._state.get("hot_water_available", 0.0)) >= target - 1e-9:
            if not source_auto:
                pct = int(round(100.0 * target))
                self._water_notice(f"Preheat not needed ({tag}) — tank already at or above ~{pct}%.")
            return
        self._preheat_busy = True
        try:
            self._water_notice(f"Preheat on ({tag}).")
            self._publish_hw(device_activity="preheating", notify_feed=False)
            step = 3.0
            rate = float(config.FUSED_AUTO_PREHEAT_FILL_PER_MINUTE)
            while float(self._state.get("hot_water_available", 0.0)) < target:
                if self._state.get("device_state") != "idle":
                    self._water_notice(f"Preheat cut ({tag}) — unit busy.")
                    return
                yield self.env.timeout(step)
                if self._state.get("device_state") != "idle":
                    self._water_notice(f"Preheat cut ({tag}) — unit busy.")
                    return
                if self._last_carbon >= cap:
                    self._water_notice(f"Preheat stopped ({tag}) — carbon over {int(cap)}.")
                    return
                cur = float(self._state.get("hot_water_available", 0.0))
                self._state["hot_water_available"] = min(1.0, cur + rate * step)
                self._publish_hw(
                    device_activity="preheating",
                    notify_feed=source_auto,
                    feed_suffix=f"preheat · {tag}" if source_auto else "",
                )
            self._water_notice(f"Preheat done ({tag}).")
        finally:
            self._preheat_busy = False
            self._publish_hw(device_activity="idle")

    def _preheat_for_person(self, person_id: str):
        yield from self._preheat_ramp(person_id, source_auto=False)

    def _fused_auto_preheat_poll(self):
        while True:
            yield self.env.timeout(float(config.FUSED_AUTO_PREHEAT_POLL_MINUTES))
            self._try_fused_auto_preheat_if_eligible()

    def run(self):
        if self._scenario_name == "fused":
            self.env.process(self._fused_auto_preheat_poll())
            self.env.process(self._fused_demo_shower_negotiations())
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

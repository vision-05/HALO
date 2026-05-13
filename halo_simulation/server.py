# To run:
# pip install fastapi uvicorn sse-starlette
#
# Option A — from repository root (recommended):
#   PYTHONPATH=. uvicorn halo_simulation.server:app --host 0.0.0.0 --port 8000
#
# Option B — from halo_simulation/ directory:
#   python server.py
#
# Open http://localhost:8000 in your browser

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

# Ensure repo root is on path so `halo_simulation` package resolves when running
# `python server.py` from inside this directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_repo_dotenv() -> None:
    """Load repo-root `.env` into `os.environ` if present (does not override existing vars)."""
    path = _REPO_ROOT / ".env"
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
    except ImportError:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            key, _, val = s.partition("=")
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val


_load_repo_dotenv()

from halo_simulation import config


def _log_llm_gateway_hint() -> None:
    """Visible at default uvicorn log level — helps Cisco LiteLLM setup."""
    proto = config.llm_protocol()
    style = os.getenv("LLM_AUTH_STYLE", "x-api-key").strip() or "x-api-key"
    if proto == "openai":
        u = config.llm_openai_chat_url()
        full = os.getenv("LLM_OPENAI_CHAT_URL", "").strip()
        if u:
            logger.warning(
                "HALO LLM: protocol=openai (LiteLLM OpenAI compat) POST %s — model=%s (override with LLM_MODEL)",
                u if not full else full,
                config.llm_model(),
            )
        else:
            logger.warning(
                "HALO LLM: LLM_PROTOCOL=openai but no URL — set LITELLM_BASE_URL or LLM_OPENAI_CHAT_URL and restart.",
            )
        return

    base = os.getenv("LLM_ANTHROPIC_BASE_URL", os.getenv("LITELLM_BASE_URL", "")).strip()
    full = os.getenv("LLM_MESSAGES_URL", "").strip()
    if full:
        logger.warning("HALO LLM: LLM_MESSAGES_URL is set (full endpoint override). LLM_AUTH_STYLE=%s", style)
    elif base:
        logger.warning(
            "HALO LLM: gateway base URL=%s — requests use %s/v1/messages. LLM_AUTH_STYLE=%s",
            base,
            base.rstrip("/"),
            style,
        )
    else:
        logger.warning(
            "HALO LLM: LLM_ANTHROPIC_BASE_URL / LITELLM_BASE_URL not set — calls go to api.anthropic.com. "
            "For LiteLLM Anthropic passthrough add base URL + CLAUDE_KEY + LLM_AUTH_STYLE=bearer; "
            "for OpenAI-compatible LiteLLM use LLM_PROTOCOL=openai + LITELLM_BASE_URL + CLAUDE_KEY. "
            "Restart uvicorn.",
        )


_log_llm_gateway_hint()

import pandas as pd
from halo_simulation.metrics.collector import (
    FailureEvent,
    LearningEvent,
    LLMApiCallEvent,
    LLMFailureEvent,
    LLMReasoningEvent,
    MetricsCollector,
    NegotiationEvent,
)
from halo_simulation.negotiation import protocol
from halo_simulation.negotiation.message import Message, MessageBus, MessageTypes
from halo_simulation.scenarios.base_scenario import BaseScenario
from halo_simulation.scenarios.carbon_spike import CarbonSpikeScenario
from halo_simulation.scenarios.device_failure import DeviceFailureScenario
from halo_simulation.scenarios.cli_bridge import CliBridgeScenario
from halo_simulation.scenarios.fused import FusedScenario
from halo_simulation.scenarios.temperature_conflict import TemperatureConflictScenario

_UI_DIR = Path(__file__).resolve().parent / "ui"
_NO_SHOWER_UID_IN_TELEMETRY = object()


def sim_minutes_to_clock_str(sim_minutes: float) -> str:
    mod = int(sim_minutes) % (24 * 60)
    return f"{mod // 60:02d}:{mod % 60:02d}"


LLM_DRIVEN_MSG_TYPES: frozenset[str] = frozenset(
    {
        MessageTypes.CostPressureUpdate,
        MessageTypes.ExternalDisruptionEvent,
        MessageTypes.GrocerySignalUpdate,
        MessageTypes.WeatherForecastAlert,
        MessageTypes.LLMObservationUpdate,
    }
)


def _describe_message(msg: Message) -> str:
    pl = msg.payload or {}
    mt = msg.msg_type
    if mt == MessageTypes.PreferenceDeclaration:
        prefs = pl.get("preferences", {})
        return f"Prefers {prefs.get('temperature', '?')}°C, home={pl.get('is_home', True)}"
    if mt == MessageTypes.CarbonIntensityUpdate:
        return f"Carbon {pl.get('current', '?')} gCO2/kWh ({pl.get('band', '?')})"
    if mt == MessageTypes.WeatherUpdate:
        cond = pl.get("condition")
        base = f"Outdoor {pl.get('outdoor_temp_c', pl.get('temperature', '?'))}°C"
        return f"{base}" + (f", {cond}" if cond else "")
    if mt == MessageTypes.NegotiationProposal:
        nid = pl.get("negotiation_id", "")
        nid_s = f" · nid {nid}" if nid else ""
        return f"Propose {pl.get('proposed_value', '?')} ({pl.get('attribute', '')}){nid_s}"
    if mt == MessageTypes.NegotiationResolved:
        return f"Resolved setpoint {pl.get('final_value', '?')}°C"
    if mt == MessageTypes.NegotiationFailed:
        return "Negotiation failed — fallback"
    if mt == MessageTypes.DeviceFailureNotice:
        return f"Failure: {pl.get('device_type', '?')}"
    if mt == MessageTypes.DeviceRecoveryNotice:
        return "Recovered"
    if mt == MessageTypes.DeviceTelemetry:
        frac = pl.get("hot_water_fraction")
        try:
            pct = int(round(100.0 * float(frac))) if frac is not None else None
        except (TypeError, ValueError):
            pct = None
        suf = (pl.get("feed_suffix") or "").strip()
        if pct is None:
            return "Tank update" + (f" — {suf}" if suf else "")
        return ("Hot water " + str(pct) + "%") + (" — " + suf if suf else "")
    if mt == MessageTypes.LLMReasoningNotice:
        rel = pl.get("relevant")
        aid = pl.get("api_id", "")
        if rel:
            return f"LLM reasoning · relevant {aid}"
        return "LLM reasoning · no external fetch"
    if mt == MessageTypes.CostPressureUpdate:
        return f"Cost pressure ({pl.get('severity', '?')}): {pl.get('summary', '')}"
    if mt == MessageTypes.ExternalDisruptionEvent:
        return f"Disruption: {pl.get('summary', '')}"
    if mt == MessageTypes.GrocerySignalUpdate:
        return f"Grocery signal: {pl.get('summary', '')}"
    if mt == MessageTypes.WeatherForecastAlert:
        return f"Weather alert ({pl.get('severity', '?')}): {pl.get('summary', '')}"
    if mt == MessageTypes.LLMObservationUpdate:
        return pl.get("summary") or "LLM observation"
    if mt in (MessageTypes.DepartureNotice, MessageTypes.ArrivalNotice, MessageTypes.SleepNotice):
        return mt.replace("Notice", "").lower()
    return str(mt)


def message_to_public_dict(msg: Message) -> dict[str, Any]:
    pl = msg.payload or {}
    out: dict[str, Any] = {
        "timestamp": msg.timestamp,
        "sender": msg.sender_id,
        "recipient": msg.recipient_id,
        "msg_type": msg.msg_type,
        "description": _describe_message(msg),
        "priority": msg.priority,
    }
    nid = pl.get("negotiation_id")
    if nid is not None:
        out["negotiation_id"] = str(nid)
    return out


def negotiation_to_dict(e: NegotiationEvent) -> dict[str, Any]:
    return {
        "timestamp": e.timestamp,
        "device_id": e.device_id,
        "iterations": e.iterations,
        "converged": e.converged,
        "final_value": e.final_value,
        "satisfaction_scores": dict(e.satisfaction_scores),
        "carbon_intensity": e.carbon_intensity,
        "fallback_used": e.fallback_used,
        "participants": list(e.participants),
        "participant_preferences": dict(e.participant_preferences or {}),
    }


def failure_to_dict(e: FailureEvent) -> dict[str, Any]:
    return {
        "timestamp": e.timestamp,
        "device_id": e.device_id,
        "recovery_attempts": e.recovery_attempts,
        "recovery_succeeded": e.recovery_succeeded,
        "failure_type": e.failure_type,
        "time_in_failed_state": e.time_in_failed_state,
    }


def learning_to_dict(e: LearningEvent) -> dict[str, Any]:
    return {
        "timestamp": e.timestamp,
        "person_id": e.person_id,
        "device_type": e.device_type,
        "ema_value": e.ema_value,
        "bayesian_mu": e.bayesian_mu,
        "bayesian_sigma": e.bayesian_sigma,
        "routine_stable": e.routine_stable,
    }


def llm_reasoning_to_dict(event: LLMReasoningEvent, pending_calls: list[str]) -> dict[str, Any]:
    ts = int(event.timestamp)
    sim_time_str = sim_minutes_to_clock_str(float(ts))
    return {
        "sim_time": ts,
        "sim_time_str": sim_time_str,
        "context_size": len(event.context_snapshot),
        "relevant": event.relevant,
        "api_id": event.api_id,
        "reason": event.reason,
        "pending_calls": list(pending_calls),
        "llm_latency_ms": event.llm_latency_ms,
    }


def llm_observation_from_message(msg: Message) -> dict[str, Any]:
    pl = msg.payload or {}
    ts = float(msg.timestamp)
    return {
        "msg_type": msg.msg_type,
        "api_id": str(pl.get("api_id", "")),
        "summary": str(pl.get("summary", "")),
        "severity": str(pl.get("severity", "low")),
        "timestamp": msg.timestamp,
        "sim_time_str": sim_minutes_to_clock_str(ts),
        "payload": dict(pl),
    }


def llm_api_call_to_dict(e: LLMApiCallEvent) -> dict[str, Any]:
    return {
        "timestamp": e.timestamp,
        "sim_time_str": sim_minutes_to_clock_str(e.timestamp),
        "api_id": e.api_id,
        "success": e.success,
        "observation_summary": e.observation_summary,
        "severity": e.severity,
        "halo_message_type": e.halo_message_type,
        "latency_ms": e.latency_ms,
    }


def llm_failure_to_dict(e: LLMFailureEvent) -> dict[str, Any]:
    return {
        "timestamp": e.timestamp,
        "sim_time_str": sim_minutes_to_clock_str(e.timestamp),
        "phase": e.phase,
        "api_id": e.api_id,
        "message": e.message,
        "detail": e.detail,
    }


def agent_states_from_message(msg: Message) -> list[dict[str, Any]]:
    """Derive UI agent_state events from a routed message (no access to agent internals)."""
    out: list[dict[str, Any]] = []
    ts = msg.timestamp
    mt = msg.msg_type
    pl = msg.payload

    if mt == MessageTypes.PreferenceDeclaration:
        pid = pl.get("person_id", msg.sender_id)
        prefs = pl.get("preferences", {})
        temp = prefs.get("temperature", "")
        home = pl.get("is_home", True)
        out.append(
            {
                "agent_id": pid,
                "agent_type": "person",
                "state_key": "presence",
                "state_value": "home" if home else "away",
                "timestamp": ts,
            }
        )
        out.append(
            {
                "agent_id": pid,
                "agent_type": "person",
                "state_key": "activity",
                "state_value": "awake",
                "timestamp": ts,
            }
        )
        out.append(
            {
                "agent_id": pid,
                "agent_type": "person",
                "state_key": "preferred_temperature",
                "state_value": float(temp) if temp != "" else None,
                "timestamp": ts,
            }
        )
    elif mt == MessageTypes.DepartureNotice:
        out.append(
            {
                "agent_id": msg.sender_id,
                "agent_type": "person",
                "state_key": "presence",
                "state_value": "away",
                "timestamp": ts,
            }
        )
        out.append(
            {
                "agent_id": msg.sender_id,
                "agent_type": "person",
                "state_key": "activity",
                "state_value": "awake",
                "timestamp": ts,
            }
        )
    elif mt == MessageTypes.ArrivalNotice:
        out.append(
            {
                "agent_id": msg.sender_id,
                "agent_type": "person",
                "state_key": "presence",
                "state_value": "home",
                "timestamp": ts,
            }
        )
        out.append(
            {
                "agent_id": msg.sender_id,
                "agent_type": "person",
                "state_key": "activity",
                "state_value": "awake",
                "timestamp": ts,
            }
        )
    elif mt == MessageTypes.SleepNotice:
        out.append(
            {
                "agent_id": msg.sender_id,
                "agent_type": "person",
                "state_key": "activity",
                "state_value": "sleeping",
                "timestamp": ts,
            }
        )
    elif mt == MessageTypes.NegotiationProposal:
        did = pl.get("device_id", msg.sender_id)
        out.append(
            {
                "agent_id": did,
                "agent_type": "device",
                "state_key": "device_state",
                "state_value": "negotiating",
                "timestamp": ts,
            }
        )
    elif mt == MessageTypes.NegotiationResolved:
        did = pl.get("device_id", "")
        fv = pl.get("final_value")
        out.append(
            {
                "agent_id": did,
                "agent_type": "device",
                "state_key": "device_state",
                "state_value": "resolved",
                "timestamp": ts,
            }
        )
        out.append(
            {
                "agent_id": did,
                "agent_type": "device",
                "state_key": "target_temp",
                "state_value": float(fv) if fv is not None else None,
                "timestamp": ts,
            }
        )
    elif mt == MessageTypes.ActuationCommand:
        did = msg.recipient_id
        tv = pl.get("target_temperature")
        out.append(
            {
                "agent_id": did,
                "agent_type": "device",
                "state_key": "target_temp",
                "state_value": float(tv) if tv is not None else None,
                "timestamp": ts,
            }
        )
        if pl.get("outdoor_heating_off") is not None:
            out.append(
                {
                    "agent_id": did,
                    "agent_type": "device",
                    "state_key": "outdoor_heating_off",
                    "state_value": bool(pl.get("outdoor_heating_off")),
                    "timestamp": ts,
                }
            )
        out.append(
            {
                "agent_id": did,
                "agent_type": "device",
                "state_key": "device_state",
                "state_value": "idle",
                "timestamp": ts,
            }
        )
    elif mt == MessageTypes.DeviceFailureNotice:
        did = pl.get("device_id", msg.sender_id)
        out.append(
            {
                "agent_id": did,
                "agent_type": "device",
                "state_key": "device_state",
                "state_value": "failed",
                "timestamp": ts,
            }
        )
    elif mt == MessageTypes.DeviceRecoveryNotice:
        did = pl.get("device_id", msg.sender_id)
        out.append(
            {
                "agent_id": did,
                "agent_type": "device",
                "state_key": "device_state",
                "state_value": "idle",
                "timestamp": ts,
            }
        )
    elif mt == MessageTypes.WeatherUpdate:
        temp = pl.get("outdoor_temp_c")
        out.append(
            {
                "agent_id": "specialist_weather",
                "agent_type": "specialist",
                "state_key": "outdoor_temp",
                "state_value": float(temp) if temp is not None else None,
                "timestamp": ts,
                "data_source": pl.get("source", "synthetic"),
            }
        )
    elif mt == MessageTypes.DeviceTelemetry:
        did = pl.get("device_id", msg.sender_id)
        frac = pl.get("hot_water_fraction")
        try:
            fv = float(frac) if frac is not None else None
        except (TypeError, ValueError):
            fv = None
        if fv is not None:
            out.append(
                {
                    "agent_id": did,
                    "agent_type": "device",
                    "state_key": "hot_water_fraction",
                    "state_value": fv,
                    "timestamp": ts,
                }
            )
        act = pl.get("device_activity")
        if isinstance(act, str):
            out.append(
                {
                    "agent_id": did,
                    "agent_type": "device",
                    "state_key": "device_state",
                    "state_value": act,
                    "timestamp": ts,
                }
            )
        uid_marker = pl.get("shower_user_id", _NO_SHOWER_UID_IN_TELEMETRY)
        if uid_marker is not _NO_SHOWER_UID_IN_TELEMETRY:
            out.append(
                {
                    "agent_id": did,
                    "agent_type": "device",
                    "state_key": "shower_user",
                    "state_value": uid_marker if isinstance(uid_marker, str) else None,
                    "timestamp": ts,
                }
            )

    return out


class StreamingMetricsCollector(MetricsCollector):
    def __init__(self, scenario_name: str, emit: Callable[[str, dict[str, Any]], None]) -> None:
        super().__init__(scenario_name)
        self._emit = emit

    def log_negotiation(self, event: NegotiationEvent) -> None:
        super().log_negotiation(event)
        self._emit("negotiation", negotiation_to_dict(event))

    def log_failure(self, event: FailureEvent) -> None:
        super().log_failure(event)
        self._emit("failure", failure_to_dict(event))

    def log_learning(self, event: LearningEvent) -> None:
        super().log_learning(event)
        self._emit("learning", learning_to_dict(event))

    def log_llm_reasoning(self, event: LLMReasoningEvent, pending_calls: list[str] | None = None) -> None:
        super().log_llm_reasoning(event, pending_calls)
        self._emit("llm_reasoning", llm_reasoning_to_dict(event, pending_calls or []))

    def log_llm_api_call(self, event: LLMApiCallEvent) -> None:
        super().log_llm_api_call(event)
        self._emit("llm_api_call", llm_api_call_to_dict(event))

    def log_llm_failure(self, event: LLMFailureEvent) -> None:
        super().log_llm_failure(event)
        self._emit("llm_pipeline_error", llm_failure_to_dict(event))


class StreamingMessageBus(MessageBus):
    def __init__(
        self,
        env: Any,
        metrics: MetricsCollector | None,
        emit: Callable[[str, dict[str, Any]], None],
    ) -> None:
        super().__init__(env, metrics=metrics)
        self._emit = emit

    def send(self, message: Message) -> None:
        super().send(message)
        self._after_route(message)

    def broadcast(self, message: Message) -> None:
        super().broadcast(message)
        self._after_route(message)

    def _after_route(self, message: Message) -> None:
        if message.msg_type == MessageTypes.DeviceTelemetry:
            for st in agent_states_from_message(message):
                self._emit("agent_state", st)
            pl = message.payload or {}
            if pl.get("notify_feed"):
                self._emit("message", message_to_public_dict(message))
            return

        self._emit("message", message_to_public_dict(message))
        for st in agent_states_from_message(message):
            self._emit("agent_state", st)

        if message.msg_type == MessageTypes.CarbonIntensityUpdate:
            pl = message.payload
            cur = float(pl.get("current", 0))
            band = pl.get("band") or protocol.carbon_band(cur)
            self._emit(
                "carbon",
                {
                    "timestamp": message.timestamp,
                    "value": cur,
                    "level": str(band).upper(),
                    "source": pl.get("source", "synthetic"),
                },
            )
        if message.msg_type == MessageTypes.WeatherUpdate:
            pl = message.payload
            self._emit(
                "weather",
                {
                    "timestamp": message.timestamp,
                    "temperature": pl.get("outdoor_temp_c"),
                    "condition": pl.get("condition"),
                    "source": pl.get("source", "synthetic"),
                },
            )
        if message.msg_type in LLM_DRIVEN_MSG_TYPES:
            self._emit("llm_observation", llm_observation_from_message(message))


def make_emit(loop: asyncio.AbstractEventLoop, q: asyncio.Queue) -> Callable[[str, dict[str, Any]], None]:
    def emit(event: str, data: dict[str, Any]) -> None:
        def _put() -> None:
            try:
                q.put_nowait({"event": event, "data": data})
            except Exception:
                pass

        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:
            pass

    return emit


def enrich_summary(metrics: MetricsCollector) -> dict[str, Any]:
    s = metrics.summary_stats()
    by_person: dict[str, list[float]] = {}
    for e in metrics.negotiation_events:
        for pid, sc in e.satisfaction_scores.items():
            by_person.setdefault(pid, []).append(float(sc))
    s["mean_satisfaction_by_person"] = {
        k: sum(v) / len(v) for k, v in by_person.items() if v
    }
    s["scenario"] = metrics.scenario_name
    # Dishwasher timing is not exposed via messages in the base codebase; optional placeholders.
    s["dishwasher_scheduled_start_sim_minute"] = None
    s["dishwasher_actual_start_sim_minute"] = None
    return s


class StreamingTemperatureConflictScenario(TemperatureConflictScenario):
    def __init__(
        self,
        seed: int,
        days: int,
        emit: Callable[[str, dict[str, Any]], None],
        api_client: Any | None = None,
    ) -> None:
        metrics = StreamingMetricsCollector("temperature_conflict", emit)
        BaseScenario.__init__(self, seed, days, metrics)
        self._api_client = api_client
        self.bus = StreamingMessageBus(self.env, metrics=metrics, emit=emit)


class StreamingCarbonSpikeScenario(CarbonSpikeScenario):
    def __init__(
        self,
        seed: int,
        days: int,
        emit: Callable[[str, dict[str, Any]], None],
        api_client: Any | None = None,
    ) -> None:
        metrics = StreamingMetricsCollector("carbon_spike", emit)
        BaseScenario.__init__(self, seed, days, metrics)
        self._api_client = api_client
        self.bus = StreamingMessageBus(self.env, metrics=metrics, emit=emit)


class StreamingDeviceFailureScenario(DeviceFailureScenario):
    def __init__(
        self,
        seed: int,
        days: int,
        emit: Callable[[str, dict[str, Any]], None],
        api_client: Any | None = None,
    ) -> None:
        metrics = StreamingMetricsCollector("device_failure", emit)
        BaseScenario.__init__(self, seed, days, metrics)
        self._api_client = api_client
        self.bus = StreamingMessageBus(self.env, metrics=metrics, emit=emit)


class StreamingCliBridgeScenario(CliBridgeScenario):
    """``cli_bridge`` with SSE-friendly metrics + ``StreamingMessageBus`` (same ``build()`` as base)."""

    def __init__(
        self,
        seed: int,
        days: int,
        emit: Callable[[str, dict[str, Any]], None],
        inject_queue: queue.Queue,
        api_client: Any | None = None,
    ) -> None:
        metrics = StreamingMetricsCollector("cli_bridge", emit)
        BaseScenario.__init__(self, seed, days, metrics)
        self._inject_queue = inject_queue
        self._api_client = api_client
        self._status_reply = queue.Queue(maxsize=4)
        self.bus = StreamingMessageBus(self.env, metrics=metrics, emit=emit)


class StreamingFusedScenario(FusedScenario):
    """``fused`` with SSE-friendly metrics + ``StreamingMessageBus`` (same ``build()`` as base)."""

    def __init__(
        self,
        seed: int,
        days: int,
        emit: Callable[[str, dict[str, Any]], None],
        inject_queue: queue.Queue,
        api_client: Any | None = None,
    ) -> None:
        metrics = StreamingMetricsCollector("fused", emit)
        BaseScenario.__init__(self, seed, days, metrics)
        self._inject_queue = inject_queue
        self._api_client = api_client
        self._status_reply = queue.Queue(maxsize=4)
        self.bus = StreamingMessageBus(self.env, metrics=metrics, emit=emit)


def create_scenario(
    name: str,
    seed: int,
    days: int,
    emit: Callable[[str, dict[str, Any]], None],
    api_client: Any | None = None,
    inject_queue: queue.Queue | None = None,
) -> BaseScenario:
    if name == "temperature_conflict":
        return StreamingTemperatureConflictScenario(seed, days, emit, api_client=api_client)
    if name == "carbon_spike":
        return StreamingCarbonSpikeScenario(seed, days, emit, api_client=api_client)
    if name == "device_failure":
        return StreamingDeviceFailureScenario(seed, days, emit, api_client=api_client)
    if name == "cli_bridge":
        if inject_queue is None:
            raise ValueError("cli_bridge requires inject_queue")
        return StreamingCliBridgeScenario(seed, days, emit, inject_queue, api_client=api_client)
    if name == "fused":
        if inject_queue is None:
            raise ValueError("fused requires inject_queue")
        return StreamingFusedScenario(seed, days, emit, inject_queue, api_client=api_client)
    raise ValueError(f"Unknown scenario: {name}")


def run_simulation_thread(
    scenario_name: str,
    seed: int,
    days: int,
    emit: Callable[[str, dict[str, Any]], None],
    api_client: Any | None = None,
    stop_requested: threading.Event | None = None,
    inject_queue: queue.Queue | None = None,
    demo_wall_seconds: float = 0.0,
) -> None:
    try:
        try:
            sim_epoch = api_client.sim_epoch_utc if api_client is not None else datetime.now(timezone.utc)
            rs: dict[str, Any] = {"sim_epoch_utc": sim_epoch.isoformat()}
            if demo_wall_seconds > 0:
                rs["demo_wall_seconds"] = float(demo_wall_seconds)
            emit("run_start", rs)
            sc = create_scenario(
                scenario_name,
                seed,
                days,
                emit,
                api_client=api_client,
                inject_queue=inject_queue,
            )
            sc.build()

            from halo_simulation.agents.llm_specialist_agent import LLMSpecialistAgent
            from halo_simulation.external.api_registry import ApiRegistry
            from halo_simulation.external.llm_client import LLMClient

            api_registry = ApiRegistry()
            api_key = config.anthropic_api_key()
            if not api_key:
                logger.warning(
                    "No Anthropic API key — set ANTHROPIC_API_KEY or CLAUDE_KEY in the environment "
                    "or repo-root .env (LLM reasoning uses fallback until then)",
                )
            llm_client = LLMClient(api_key=api_key)
            llm_agent = LLMSpecialistAgent(
                env=sc.env,
                message_bus=sc.bus,
                metrics=sc.metrics,
                api_registry=api_registry,
                llm_client=llm_client,
                context_window=10,
                reasoning_interval=60,
                meal_context=getattr(sc, "_meal_context", None),
            )
            sc._agents.append(llm_agent)

            sc.register_all()
            sc.start_processes()
            rl_model = os.environ.get("HALO_RL_THERMOSTAT_MODEL", "").strip()
            if rl_model:
                try:
                    from halo_simulation.rl.live_inference import attach_rl_thermostat_sidecar

                    attach_rl_thermostat_sidecar(sc, emit, rl_model)
                except Exception:
                    logger.exception(
                        "HALO_RL_THERMOSTAT_MODEL is set (%r) but RL thermostat sidecar failed to start",
                        rl_model,
                    )
            until = float(config.MINUTES_PER_DAY * days)
            chunk = float(config.STREAM_STOP_CHECK_CHUNK_MINUTES)
            stopped_early = False
            while sc.env.now < until - 1e-9:
                if stop_requested is not None and stop_requested.is_set():
                    stopped_early = True
                    break
                next_t = min(sc.env.now + chunk, until)
                if next_t <= sc.env.now:
                    break
                t0 = float(sc.env.now)
                sc.env.run(until=next_t)
                advanced = float(sc.env.now) - t0
                if (
                    demo_wall_seconds > 0.0
                    and until > 1e-9
                    and advanced > 1e-12
                    and not (stop_requested is not None and stop_requested.is_set())
                ):
                    delay = demo_wall_seconds * (advanced / until)
                    if delay > 0:
                        time.sleep(delay)

            if stopped_early:
                emit(
                    "stopped",
                    {
                        "sim_minutes": sc.env.now,
                        "target_minutes": until,
                        "message": "Simulation stopped (client disconnected or Stop)",
                    },
                )
            else:
                paths = sc.metrics.save_outputs()
                summary = enrich_summary(sc.metrics)
                summary["output_paths"] = [str(p) for p in paths]
                emit("done", summary)
        except Exception:
            emit("error", {"traceback": traceback.format_exc()})
    finally:
        if api_client is not None:
            api_client.close()


app = FastAPI(title="HALO Simulation Stream")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index() -> FileResponse:
    path = _UI_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ui/index.html not found")
    return FileResponse(path, media_type="text/html")


@app.get("/stream")
async def stream(
    scenario: str = Query(
        ...,
        description="temperature_conflict | carbon_spike | device_failure | cli_bridge | fused",
    ),
    days: int = Query(14, ge=1, le=365),
    seed: int = Query(42),
    live_data: bool = False,
    demo_wall_seconds: float = Query(
        0.0,
        ge=0.0,
        le=float(config.DEMO_WALL_SECONDS_MAX),
        description="Stretch the full run across this many wall-clock seconds (0 = as fast as possible). "
        "Use ~60 for CLI human-in-the-loop demos.",
    ),
) -> EventSourceResponse:
    if scenario not in ("temperature_conflict", "carbon_spike", "device_failure", "cli_bridge", "fused"):
        raise HTTPException(status_code=400, detail="Invalid scenario")

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    emit = make_emit(loop, q)

    inject_queue: queue.Queue | None = None
    app.state.inject_queue = None
    if scenario in ("cli_bridge", "fused"):
        inject_queue = queue.Queue()
        app.state.inject_queue = inject_queue

    api_client: Any | None = None
    if live_data:
        from halo_simulation.external.api_client import ExternalDataClient

        api_client = ExternalDataClient()
        carbon_data = api_client.get_carbon_intensity()
        weather_data = api_client.get_weather()
        emit(
            "api_status",
            {
                "carbon": api_client.api_status["carbon"],
                "weather": api_client.api_status["weather"],
                "current_carbon": carbon_data["value"],
                "current_temp": weather_data["temperature"],
                "current_condition": weather_data["condition"],
            },
        )

    stop_requested = threading.Event()

    def _run() -> None:
        run_simulation_thread(
            scenario,
            seed,
            days,
            emit,
            api_client=api_client,
            stop_requested=stop_requested,
            inject_queue=inject_queue,
            demo_wall_seconds=float(demo_wall_seconds),
        )

    t = threading.Thread(
        target=_run,
        daemon=True,
        name="halo-simpy",
    )
    t.start()

    async def event_generator():
        try:
            while True:
                item = await q.get()
                ev = item["event"]
                data = item["data"]
                yield {"event": ev, "data": json.dumps(data)}
                if ev in ("done", "error", "stopped"):
                    break
        except asyncio.CancelledError:
            stop_requested.set()
            raise
        finally:
            stop_requested.set()
            app.state.inject_queue = None

    return EventSourceResponse(event_generator())


@app.post("/api/inject")
async def inject_message(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Enqueue a human-bridge command for an active ``cli_bridge`` / ``fused`` stream."""
    from halo_simulation.human_bridge import validate_queue_item

    item = validate_queue_item(body)
    if item is None or item.get("op") == "__status__":
        raise HTTPException(status_code=400, detail="Invalid inject body (see human_bridge contract)")
    q = getattr(app.state, "inject_queue", None)
    if q is None:
        raise HTTPException(
            status_code=503,
            detail="No active human-in-loop stream — start GET /stream?scenario=cli_bridge "
            "or scenario=fused first",
        )
    try:
        q.put_nowait(item)
    except queue.Full:
        raise HTTPException(status_code=503, detail="Inject queue full") from None
    return {"ok": True, "accepted": item}


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    """Quick connectivity check — called by frontend on page load."""
    from halo_simulation.external.api_client import ExternalDataClient

    client = ExternalDataClient()
    try:
        carbon = client.get_carbon_intensity()
        weather = client.get_weather()
        return {
            "carbon_api": client.api_status["carbon"],
            "weather_api": client.api_status["weather"],
            "carbon_value": carbon["value"],
            "carbon_level": carbon["level"],
            "temperature": weather["temperature"],
            "condition": weather["condition"],
            "is_heatwave": weather["is_heatwave"],
            "is_cold_snap": weather["is_cold_snap"],
        }
    finally:
        client.close()


@app.get("/api/weather_series")
async def weather_series(
    forecast_days: int = Query(16, ge=1, le=16, description="Open-Meteo hourly horizon (max 16)"),
) -> dict[str, Any]:
    """Full hourly London forecast for dashboard chart (temperature, wind, WMO codes)."""
    from halo_simulation.external.api_client import fetch_weather_hourly_chart_data

    try:
        return fetch_weather_hourly_chart_data(forecast_days=forecast_days)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Open-Meteo forecast failed: {e}") from e


_SAFE_SCENARIO = re.compile(r"^[a-z_]+$")


@app.get("/results/{scenario}")
async def get_results(scenario: str) -> JSONResponse:
    if not _SAFE_SCENARIO.match(scenario):
        raise HTTPException(status_code=400, detail="Invalid scenario name")
    metrics = MetricsCollector(scenario)
    neg_path = Path(metrics.output_dir) / f"{scenario}_negotiations.csv"
    fail_path = Path(metrics.output_dir) / f"{scenario}_failures.csv"
    if not neg_path.is_file() and not fail_path.is_file():
        raise HTTPException(status_code=404, detail="No outputs yet — run a simulation first")

    def read_csv(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        df = pd.read_csv(path)
        return json.loads(df.to_json(orient="records"))

    return JSONResponse(
        {
            "scenario": scenario,
            "negotiations": read_csv(neg_path),
            "failures": read_csv(fail_path),
        }
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

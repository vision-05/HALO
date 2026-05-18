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
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

# --- LLM inspector (thread-safe; filled while /stream SimPy worker runs) ---
_llm_inspector_lock = threading.Lock()
_llm_inspector_timeline: deque[dict[str, Any]] = deque(maxlen=80)
_api_registry_for_hints: Any = None


def _api_registry_singleton() -> Any:
    global _api_registry_for_hints
    if _api_registry_for_hints is None:
        from halo_simulation.external.api_registry import ApiRegistry

        _api_registry_for_hints = ApiRegistry()
    return _api_registry_for_hints


def _reset_llm_inspector_timeline() -> None:
    with _llm_inspector_lock:
        _llm_inspector_timeline.clear()


def _impact_log_from_timeline(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows for the real-world impact table (LLM-interpreted bus messages); newest first."""
    rows: list[dict[str, Any]] = []
    for ev in timeline:
        if ev.get("kind") != "llm_observation":
            continue
        rows.append(
            {
                "timestamp": ev.get("timestamp"),
                "sim_time_str": ev.get("sim_time_str"),
                "api_id": ev.get("api_id"),
                "msg_type": ev.get("msg_type"),
                "summary": ev.get("summary"),
                "severity": ev.get("severity"),
                "sim_effect": ev.get("sim_effect"),
            }
        )
    rows.reverse()
    return rows


def llm_effects_reference() -> dict[str, str]:
    """Static map: HALO message types emitted after LLM interpretation → who reacts in this codebase."""
    return {
        "ExternalDisruptionEvent": (
            "All PersonAgents (Alice, Bob, person_cli, …) receive the broadcast. "
            "If the summary text matches transport keywords (strike, rail, delay, …), that person "
            "accumulates extra simulated minutes before their next return home. "
            "If it matches energy keywords (power, outage, grid, …), their comfort_weight is reduced slightly."
        ),
        "CostPressureUpdate": (
            "ThermostatDeviceAgent only: boosts device-side negotiation weight for a timed window "
            "after the message (severity sets duration and boost size)."
        ),
        "WeatherForecastAlert": (
            "ThermostatDeviceAgent only: if hourly forecast implies heatwave or cold snap, "
            "temporarily caps or floors the comfort negotiation range for a simulated-time window."
        ),
        "GrocerySignalUpdate": (
            "LLMSpecialistAgent listens and may emit a virtual_shopping ActuationCommand with suggested items; "
            "mainly narrative / feed unless other agents subscribe."
        ),
        "LLMObservationUpdate": (
            "Generic advisory broadcast for dashboards; no built-in subscriber beyond the UI stream."
        ),
        "DishwasherScheduleDecision": (
            "Washing machine (device_dishwasher): internal schedule gate after LLM JSON or heuristic fallback; "
            "see llm_api_call observation_summary (approve, defer_minutes, reason). No extra bus message."
        ),
    }


def _pending_api_effect_hints(api_ids: list[str]) -> list[dict[str, str]]:
    reg = _api_registry_singleton()
    ref = llm_effects_reference()
    out: list[dict[str, str]] = []
    for aid in api_ids:
        d = reg.get(aid)
        halo = str(d.halo_message_type) if d else ""
        eff = ref.get(halo, "Advisory HALO message; effect depends on subscribers.")
        out.append(
            {
                "api_id": aid,
                "halo_message_type": halo,
                "effect_summary": eff,
            }
        )
    return out

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

_LLM_TRANSPORT_KW: tuple[str, ...] = (
    "strike",
    "tube",
    "rail",
    "train",
    "bus",
    "delay",
    "disruption",
    "transport",
)
_LLM_ENERGY_KW: tuple[str, ...] = ("power", "outage", "blackout", "energy", "electricity", "grid")


def _external_disruption_effect(summary: str, severity: str) -> str:
    s = summary.lower()
    sev = str(severity or "low").lower()
    delay = {"low": 15.0, "medium": 45.0, "high": 90.0}.get(sev, 30.0)
    parts: list[str] = []
    if any(k in s for k in _LLM_TRANSPORT_KW):
        parts.append(
            f"PersonAgents: each adds +{delay:g} sim min to return-home delay (transport keyword in summary)."
        )
    if any(k in s for k in _LLM_ENERGY_KW):
        parts.append("PersonAgents: each lowers comfort_weight by 0.2 (min 0.3) when energy keyword matches.")
    if not parts:
        return "Simulation: unchanged (summary matched no transport or energy keywords)."
    return " ".join(parts)


def _cost_pressure_effect(severity: str) -> str:
    sev = str(severity or "low").lower()
    if sev == "high":
        boost, mins = 0.4, 120
    elif sev == "medium":
        boost, mins = 0.2, 60
    else:
        boost, mins = 0.1, 30
    return (
        f"ThermostatDeviceAgent: negotiation device_weight gets +{boost} boost (with cap) for {mins} sim min "
        f"(severity={sev!r})."
    )


def _weather_forecast_effect(pl: dict[str, Any]) -> str:
    hourly = pl.get("hourly")
    temps: list[float] = []
    if isinstance(hourly, dict):
        raw = hourly.get("temperature_2m")
        if isinstance(raw, list):
            temps = [float(x) for x in raw[:24] if x is not None]
    elif isinstance(hourly, list):
        temps = [float(x) for x in hourly[:24]]
    if not temps:
        return "Simulation: unchanged (no hourly temperature series in payload)."
    upcoming_max = max(temps[:6])
    upcoming_min = min(temps[:6])
    if upcoming_max > 30.0:
        return (
            f"ThermostatDeviceAgent: max of next 6 hourly temps = {upcoming_max:.1f}°C (>30) "
            "→ setpoint ceiling 22°C for 180 sim min."
        )
    if upcoming_min < 2.0:
        return (
            f"ThermostatDeviceAgent: min of next 6 hourly temps = {upcoming_min:.1f}°C (<2) "
            "→ setpoint floor 17°C for 180 sim min."
        )
    return (
        f"Simulation: unchanged (next-6h min/max = {upcoming_min:.1f}°C / {upcoming_max:.1f}°C; "
        "thresholds not met)."
    )


def _sim_effect_for_halo_payload(msg_type: str, pl: dict[str, Any]) -> str:
    mt = str(msg_type or "")
    if mt == MessageTypes.ExternalDisruptionEvent:
        return _external_disruption_effect(str(pl.get("summary", "")), str(pl.get("severity", "low")))
    if mt == MessageTypes.CostPressureUpdate:
        return _cost_pressure_effect(str(pl.get("severity", "low")))
    if mt == MessageTypes.WeatherForecastAlert:
        return _weather_forecast_effect(pl)
    if mt == MessageTypes.GrocerySignalUpdate:
        return (
            "LLMSpecialistAgent may emit virtual_shopping ActuationCommand; PersonAgent ignores "
            "ActuationCommand (pass). Thermostat/shower/washing machine (device_dishwasher): no code path for GrocerySignalUpdate."
        )
    if mt == MessageTypes.LLMObservationUpdate:
        return "Simulation: unchanged (no agent handler for LLMObservationUpdate in agents/)."
    return f"Simulation: unknown msg_type {mt!r} for built-in effect mapping."


def _compute_sim_effect_line(entry: dict[str, Any]) -> str:
    kind = entry.get("kind")
    if kind == "llm_pipeline_error":
        ph = str(entry.get("phase") or "").lower()
        if ph == "cooldown":
            return "Simulation: unchanged (API on cooldown; no HTTP request — not an LLM failure)."
        return "Simulation: unchanged (LLM/API error; no observation applied)."
    if kind == "llm_reasoning":
        if not entry.get("relevant"):
            return "Decision: relevant=false → no API queued. Simulation: unchanged."
        ids = entry.get("api_ids")
        api_list: list[str] = []
        if isinstance(ids, list):
            for x in ids:
                s = str(x).strip()
                if s and s.lower() not in ("", "none"):
                    api_list.append(s)
        if not api_list:
            aid = str(entry.get("api_id") or "").strip().lower()
            if aid in ("", "none"):
                return "Decision: relevant but api_id empty/none. Simulation: unchanged."
            api_list = [str(entry.get("api_id")).strip()]
        if len(api_list) == 1:
            q = f"API {api_list[0]!r}"
        else:
            q = f"APIs (in order) {', '.join(repr(a) for a in api_list)}"
        return (
            f"Decision: queue {q}. "
            "Simulation: unchanged until fetch + interpret emit a bus message (see following rows)."
        )
    if kind == "llm_api_call":
        if not entry.get("success"):
            return "Simulation: unchanged (call failed or no interpretation payload)."
        hmt = str(entry.get("halo_message_type") or "")
        if not hmt:
            return "Simulation: unchanged (success without halo_message_type)."
        summ = str(entry.get("observation_summary") or "")
        sev = str(entry.get("severity") or "low")
        if hmt == MessageTypes.ExternalDisruptionEvent:
            return _external_disruption_effect(summ, sev)
        if hmt == MessageTypes.CostPressureUpdate:
            return _cost_pressure_effect(sev)
        if hmt == MessageTypes.WeatherForecastAlert:
            return (
                "WeatherForecastAlert emitted. Thermostat effect uses hourly temps in the llm_observation "
                "row (same run); API row alone may not include hourly JSON."
            )
        if hmt == MessageTypes.GrocerySignalUpdate:
            return _sim_effect_for_halo_payload(hmt, {"summary": summ, "severity": sev})
        if hmt == MessageTypes.LLMObservationUpdate:
            return _sim_effect_for_halo_payload(hmt, {"summary": summ, "severity": sev})
        if hmt == "DishwasherScheduleDecision":
            return f"Washing machine (device_dishwasher): applied schedule gate — {summ}"
        return f"Emitted {hmt!r}; no specialized sim_effect text for this type."
    if kind == "llm_observation":
        mt = str(entry.get("msg_type") or "")
        pl = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        return _sim_effect_for_halo_payload(mt, pl)
    return f"Unknown timeline kind {kind!r}."


def _append_llm_timeline(entry: dict[str, Any]) -> None:
    row = dict(entry)
    row.setdefault("sim_effect", _compute_sim_effect_line(row))
    with _llm_inspector_lock:
        _llm_inspector_timeline.append(row)


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
        attr = str(pl.get("attribute", "") or "")
        attr_d = "washing machine delay" if attr == "dishwasher_delay" else attr
        return f"Propose {pl.get('proposed_value', '?')} ({attr_d}){nid_s}"
    if mt == MessageTypes.NegotiationCounter:
        attr = str(pl.get("attribute", "temperature") or "temperature")
        attr_d = "washing machine delay" if attr == "dishwasher_delay" else attr
        return f"Counter {pl.get('counter_value', '?')} ({attr_d})"
    if mt == MessageTypes.NegotiationAccept:
        return "Accept negotiation"
    if mt == MessageTypes.NegotiationReject:
        return f"Reject ({pl.get('reason', '')})"
    if mt == MessageTypes.NegotiationResolved:
        attr = str(pl.get("attribute", "temperature") or "temperature")
        fv = pl.get("final_value", "?")
        if attr == "shower_minutes":
            return f"Resolved shower duration {fv} min"
        if attr == "dishwasher_delay":
            return f"Resolved washing machine start delay {fv} min"
        return f"Resolved setpoint {fv}°C"
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
    if mt == MessageTypes.WaterServiceNotice:
        return str(pl.get("detail") or "Water")
    if mt in (MessageTypes.WaterShowerIntent, MessageTypes.WaterPreheatIntent):
        return "Shower request" if mt == MessageTypes.WaterShowerIntent else "Preheat request"
    if mt == MessageTypes.DishwasherRunRequest:
        rid = pl.get("requester_id", "?")
        u = pl.get("urgency", "")
        return f"Washing machine run request ({rid}, urgency={u})"
    if mt in (MessageTypes.DepartureNotice, MessageTypes.ArrivalNotice, MessageTypes.SleepNotice):
        return mt.replace("Notice", "").lower()
    return str(mt)


def message_to_public_dict(msg: Message) -> dict[str, Any]:
    pl = msg.payload or {}
    mt = msg.msg_type
    out: dict[str, Any] = {
        "timestamp": msg.timestamp,
        "sender": msg.sender_id,
        "recipient": msg.recipient_id,
        "msg_type": mt,
        "description": _describe_message(msg),
        "priority": msg.priority,
    }
    nid = pl.get("negotiation_id")
    if nid is not None:
        out["negotiation_id"] = str(nid)
    if mt == MessageTypes.NegotiationProposal:
        if pl.get("device_id") is not None:
            out["device_id"] = str(pl["device_id"])
        if pl.get("attribute") is not None:
            out["attribute"] = str(pl["attribute"])
        pv = pl.get("proposed_value")
        if pv is not None:
            try:
                out["proposed_value"] = float(pv)
            except (TypeError, ValueError):
                out["proposed_value"] = pv
    if mt == MessageTypes.NegotiationCounter:
        if pl.get("device_id") is not None:
            out["device_id"] = str(pl["device_id"])
        if pl.get("attribute") is not None:
            out["attribute"] = str(pl["attribute"])
        cv = pl.get("counter_value")
        if cv is not None:
            try:
                out["counter_value"] = float(cv)
            except (TypeError, ValueError):
                out["counter_value"] = cv
    if mt == MessageTypes.NegotiationAccept:
        if pl.get("device_id") is not None:
            out["device_id"] = str(pl["device_id"])
        if pl.get("attribute") is not None:
            out["attribute"] = str(pl["attribute"])
    if mt == MessageTypes.NegotiationReject:
        if pl.get("device_id") is not None:
            out["device_id"] = str(pl["device_id"])
        if pl.get("reason") is not None:
            out["reason"] = str(pl["reason"])
    return out


def negotiation_to_dict(e: NegotiationEvent) -> dict[str, Any]:
    out: dict[str, Any] = {
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
    if e.preference_attribute:
        out["preference_attribute"] = e.preference_attribute
    return out


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
    snap = [dict(x) for x in (event.context_snapshot or [])]
    pc = list(pending_calls)
    api_ids = list(event.api_ids or [])
    if not api_ids and str(event.api_id or "").strip().lower() not in ("", "none"):
        api_ids = [str(event.api_id).strip()]
    return {
        "sim_time": ts,
        "sim_time_str": sim_time_str,
        "context_size": len(snap),
        "context_snapshot": snap,
        "relevant": event.relevant,
        "api_id": event.api_id,
        "api_ids": api_ids,
        "reason": event.reason,
        "pending_calls": pc,
        "pending_effect_hints": _pending_api_effect_hints(pc),
        "llm_latency_ms": event.llm_latency_ms,
    }


def _llm_inspector_registry_api_ids() -> list[str]:
    from halo_simulation.external.api_registry import ApiRegistry

    return [a.api_id for a in ApiRegistry().all() if a.enabled]


def _timeline_row_api_targets(ev: dict[str, Any]) -> list[str]:
    kind = str(ev.get("kind") or "")
    if kind == "llm_reasoning":
        ids = ev.get("api_ids")
        if isinstance(ids, list) and ids:
            out: list[str] = []
            for x in ids:
                s = str(x).strip()
                if s and s.lower() not in ("none", ""):
                    out.append(s)
            return out
        one = ev.get("api_id")
        if one and str(one).strip().lower() not in ("none", ""):
            return [str(one).strip()]
        return []
    if kind in ("llm_api_call", "llm_pipeline_error"):
        one = ev.get("api_id")
        return [str(one).strip()] if one and str(one).strip() else []
    if kind == "llm_observation":
        pl = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        one = ev.get("api_id") or pl.get("api_id")
        return [str(one).strip()] if one and str(one).strip() else []
    return []


def _timeline_grouped_by_api(
    timeline: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """Split timeline rows into per-registry-api columns (newest-first within each)."""
    columns = _llm_inspector_registry_api_ids()
    grouped: dict[str, list[dict[str, Any]]] = {c: [] for c in columns}
    grouped["_other"] = []
    col_set = set(columns)
    for ev in timeline:
        targets = _timeline_row_api_targets(ev)
        if not targets:
            grouped["_other"].append(ev)
            continue
        for aid in targets:
            key = aid if aid in col_set else "_other"
            grouped[key].append(ev)
    ordered: dict[str, list[dict[str, Any]]] = {}
    for c in columns:
        ordered[c] = list(reversed(grouped.get(c, [])))
    ordered["_other"] = list(reversed(grouped.get("_other", [])))
    return columns, ordered


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
        if fv is not None:
            hw_row: dict[str, Any] = {
                "agent_id": did,
                "agent_type": "device",
                "state_key": "hot_water_fraction",
                "state_value": fv,
                "timestamp": ts,
            }
            # Echo activity on the same row so the UI can mark preheat/running on every tank sample
            # (recharge telemetry often omits device_activity; preheat steps always include it here).
            if isinstance(act, str):
                hw_row["device_activity"] = act
            out.append(hw_row)
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
        pc = pending_calls or []
        payload = llm_reasoning_to_dict(event, pc)
        row = {"kind": "llm_reasoning", **payload}
        row["sim_effect"] = _compute_sim_effect_line(row)
        self._emit("llm_reasoning", row)
        _append_llm_timeline(row)

    def log_llm_api_call(self, event: LLMApiCallEvent) -> None:
        super().log_llm_api_call(event)
        base = llm_api_call_to_dict(event)
        row = {"kind": "llm_api_call", **base}
        row["sim_effect"] = _compute_sim_effect_line(row)
        self._emit("llm_api_call", row)
        _append_llm_timeline(row)

    def log_llm_failure(self, event: LLMFailureEvent) -> None:
        super().log_llm_failure(event)
        base = llm_failure_to_dict(event)
        row = {"kind": "llm_pipeline_error", **base}
        row["sim_effect"] = _compute_sim_effect_line(row)
        self._emit("llm_pipeline_error", row)
        _append_llm_timeline(row)


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
            obs = llm_observation_from_message(message)
            row = {"kind": "llm_observation", **obs}
            row["sim_effect"] = _compute_sim_effect_line(row)
            self._emit("llm_observation", row)
            _append_llm_timeline(row)


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
            if scenario_name == "fused":
                rs["shower_model"] = {
                    "auto_poll_min": float(config.FUSED_AUTO_PREHEAT_POLL_MINUTES),
                    "tank_below": float(config.FUSED_AUTO_PREHEAT_TANK_TRIGGER),
                    "preheat_target": float(config.FUSED_AUTO_PREHEAT_TANK_TARGET),
                    "preheat_fill_per_min": float(config.FUSED_AUTO_PREHEAT_FILL_PER_MINUTE),
                    "preheat_step_min": 3.0,
                    "carbon_cap_gco2kwh": float(config.CARBON_HIGH_THRESHOLD),
                    "auto_block_cooldown_min": float(config.FUSED_AUTO_PREHEAT_BLOCK_NOTICE_COOLDOWN_MIN),
                    "drain_per_shower": float(config.HOT_WATER_DRAIN_PER_SHOWER),
                    "recharge_per_min_base": float(config.HOT_WATER_RECHARGE_PER_MINUTE_BASE),
                    "grid_clean_recharge_mult": float(config.HOT_WATER_RECHARGE_GRID_CLEAN_MULTIPLIER),
                    "shower_cycle_min": 15.0,
                }
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
                    else:
                        # Even at Pace 0, yield GIL so asyncio loop can flush SSE events to the browser.
                        time.sleep(0.001)
                else:
                    # No demo pacing — still yield the GIL briefly so queued SSE events reach the client.
                    time.sleep(0.001)

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


@app.get("/api/llm-inspector")
async def api_llm_inspector() -> JSONResponse:
    """JSON snapshot of recent LLM reasoning, API calls, observations, pipeline errors, and impact log (same process as /stream)."""
    with _llm_inspector_lock:
        timeline = list(_llm_inspector_timeline)
    impact_log = _impact_log_from_timeline(timeline)
    column_ids, by_api = _timeline_grouped_by_api(timeline)
    return JSONResponse(
        {
            "timeline": timeline,
            "timeline_count": len(timeline),
            "by_api": by_api,
            "by_api_column_ids": column_ids,
            "impact_log": impact_log,
            "impact_log_count": len(impact_log),
            "effects_reference": llm_effects_reference(),
            "note": "Buffer is cleared when a new /stream starts. Open this page or poll this URL while a stream is running.",
        }
    )


@app.get("/llm-inspector")
async def llm_inspector_page() -> FileResponse:
    path = _UI_DIR / "llm_inspector.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ui/llm_inspector.html not found")
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

    _reset_llm_inspector_timeline()

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

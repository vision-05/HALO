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
import re
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

# Ensure repo root is on path so `halo_simulation` package resolves when running
# `python server.py` from inside this directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from halo_simulation import config
from halo_simulation.metrics.collector import (
    FailureEvent,
    LearningEvent,
    MetricsCollector,
    NegotiationEvent,
)
from halo_simulation.negotiation import protocol
from halo_simulation.negotiation.message import Message, MessageBus, MessageTypes
from halo_simulation.scenarios.base_scenario import BaseScenario
from halo_simulation.scenarios.carbon_spike import CarbonSpikeScenario
from halo_simulation.scenarios.device_failure import DeviceFailureScenario
from halo_simulation.scenarios.temperature_conflict import TemperatureConflictScenario

_UI_DIR = Path(__file__).resolve().parent / "ui"


def _describe_message(msg: Message) -> str:
    mt = msg.msg_type
    pl = msg.payload
    if mt == MessageTypes.PreferenceDeclaration:
        prefs = pl.get("preferences", {})
        return f"Prefers {prefs.get('temperature', '?')}°C, home={pl.get('is_home', True)}"
    if mt == MessageTypes.CarbonIntensityUpdate:
        return f"Carbon {pl.get('current', '?')} gCO2/kWh ({pl.get('band', '?')})"
    if mt == MessageTypes.WeatherUpdate:
        return f"Outdoor {pl.get('outdoor_temp_c', pl.get('temperature', '?'))}°C"
    if mt == MessageTypes.NegotiationProposal:
        return f"Propose {pl.get('proposed_value', '?')} ({pl.get('attribute', '')})"
    if mt == MessageTypes.NegotiationResolved:
        return f"Resolved setpoint {pl.get('final_value', '?')}°C"
    if mt == MessageTypes.NegotiationFailed:
        return "Negotiation failed — fallback"
    if mt == MessageTypes.DeviceFailureNotice:
        return f"Failure: {pl.get('device_type', '?')}"
    if mt == MessageTypes.DeviceRecoveryNotice:
        return "Recovered"
    if mt in (MessageTypes.DepartureNotice, MessageTypes.ArrivalNotice, MessageTypes.SleepNotice):
        return mt.replace("Notice", "").lower()
    return mt


def message_to_public_dict(msg: Message) -> dict[str, Any]:
    return {
        "timestamp": msg.timestamp,
        "sender": msg.sender_id,
        "recipient": msg.recipient_id,
        "msg_type": msg.msg_type,
        "description": _describe_message(msg),
        "priority": msg.priority,
    }


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
        self._emit("message", message_to_public_dict(message))
        for st in agent_states_from_message(message):
            self._emit("agent_state", st)

        if message.msg_type == MessageTypes.CarbonIntensityUpdate:
            pl = message.payload
            cur = float(pl.get("current", 0))
            band = pl.get("band") or protocol.carbon_band(cur)
            self._emit(
                "carbon",
                {"timestamp": message.timestamp, "value": cur, "level": str(band).upper()},
            )


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
    def __init__(self, seed: int, days: int, emit: Callable[[str, dict[str, Any]], None]) -> None:
        metrics = StreamingMetricsCollector("temperature_conflict", emit)
        BaseScenario.__init__(self, seed, days, metrics)
        self.bus = StreamingMessageBus(self.env, metrics=metrics, emit=emit)


class StreamingCarbonSpikeScenario(CarbonSpikeScenario):
    def __init__(self, seed: int, days: int, emit: Callable[[str, dict[str, Any]], None]) -> None:
        metrics = StreamingMetricsCollector("carbon_spike", emit)
        BaseScenario.__init__(self, seed, days, metrics)
        self.bus = StreamingMessageBus(self.env, metrics=metrics, emit=emit)


class StreamingDeviceFailureScenario(DeviceFailureScenario):
    def __init__(self, seed: int, days: int, emit: Callable[[str, dict[str, Any]], None]) -> None:
        metrics = StreamingMetricsCollector("device_failure", emit)
        BaseScenario.__init__(self, seed, days, metrics)
        self.bus = StreamingMessageBus(self.env, metrics=metrics, emit=emit)


def create_scenario(name: str, seed: int, days: int, emit: Callable[[str, dict[str, Any]], None]) -> BaseScenario:
    if name == "temperature_conflict":
        return StreamingTemperatureConflictScenario(seed, days, emit)
    if name == "carbon_spike":
        return StreamingCarbonSpikeScenario(seed, days, emit)
    if name == "device_failure":
        return StreamingDeviceFailureScenario(seed, days, emit)
    raise ValueError(f"Unknown scenario: {name}")


def run_simulation_thread(
    scenario_name: str,
    seed: int,
    days: int,
    emit: Callable[[str, dict[str, Any]], None],
) -> None:
    try:
        sc = create_scenario(scenario_name, seed, days, emit)
        sc.build()
        sc.register_all()
        sc.start_processes()
        until = float(config.MINUTES_PER_DAY * days)
        sc.env.run(until=until)
        paths = sc.metrics.save_outputs()
        summary = enrich_summary(sc.metrics)
        summary["output_paths"] = [str(p) for p in paths]
        emit("done", summary)
    except Exception:
        emit("error", {"traceback": traceback.format_exc()})


app = FastAPI(title="HALO Simulation Stream")


@app.get("/")
async def index() -> FileResponse:
    path = _UI_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ui/index.html not found")
    return FileResponse(path, media_type="text/html")


@app.get("/stream")
async def stream(
    scenario: str = Query(..., description="temperature_conflict | carbon_spike | device_failure"),
    days: int = Query(14, ge=1, le=365),
    seed: int = Query(42),
) -> EventSourceResponse:
    if scenario not in ("temperature_conflict", "carbon_spike", "device_failure"):
        raise HTTPException(status_code=400, detail="Invalid scenario")

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    emit = make_emit(loop, q)

    t = threading.Thread(
        target=run_simulation_thread,
        args=(scenario, seed, days, emit),
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
                if ev in ("done", "error"):
                    break
        except asyncio.CancelledError:
            raise

    return EventSourceResponse(event_generator())


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

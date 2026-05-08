"""
LLMSpecialistAgent — Context-Aware Specialist Agent for HALO.

Reasoning cycles, external API execution, LLM interpretation, and HALO bus emits.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Generator

import httpx
import simpy

from halo_simulation.agents.base_agent import BaseAgent
from halo_simulation.external.api_registry import ApiDefinition, ApiRegistry
from halo_simulation.external.llm_client import LLMClient
from halo_simulation.household_meals import HouseholdMealContext
from halo_simulation.metrics.collector import (
    LLMApiCallEvent,
    LLMFailureEvent,
    LLMReasoningEvent,
    MetricsCollector,
)
from halo_simulation.negotiation.message import Message, MessageBus, MessageTypes

logger = logging.getLogger(__name__)

_CONTEXT_SKIP_TYPES: frozenset[str] = frozenset(
    {
        MessageTypes.LLMReasoningNotice,
        MessageTypes.CostPressureUpdate,
        MessageTypes.ExternalDisruptionEvent,
        MessageTypes.GrocerySignalUpdate,
        MessageTypes.WeatherForecastAlert,
        MessageTypes.LLMObservationUpdate,
    }
)


class SpecialistAgent(BaseAgent):
    """Base class for specialist-role agents (shared ``specialist`` agent_type)."""

    def __init__(
        self,
        agent_id: str,
        env: simpy.Environment,
        message_bus: MessageBus,
        metrics: MetricsCollector | None,
    ) -> None:
        super().__init__(agent_id, "specialist", env, message_bus, metrics)


def _schema_lines_for_prompt(schema: dict[str, Any]) -> str:
    lines: list[str] = []
    for k, v in schema.items():
        nm = getattr(v, "__name__", str(v))
        lines.append(f'    "{k}": "{nm}"')
    return "{\n" + ",\n".join(lines) + "\n}"


class LLMSpecialistAgent(SpecialistAgent):
    def __init__(
        self,
        env: simpy.Environment,
        message_bus: MessageBus,
        metrics: MetricsCollector | None,
        api_registry: ApiRegistry,
        llm_client: LLMClient,
        agent_id: str = "specialist_llm",
        context_window: int = 10,
        reasoning_interval: int = 60,
        meal_context: HouseholdMealContext | None = None,
    ) -> None:
        super().__init__(agent_id, env, message_bus, metrics)
        self._api_registry = api_registry
        self._llm_client = llm_client
        self._max_context = context_window
        self._reasoning_interval = float(reasoning_interval)
        self._meal_context = meal_context

        self._context_window: list[dict[str, Any]] = []
        self._last_reasoning_at: float = 0.0
        self._decisions: list[dict[str, Any]] = []
        self._pending_api_calls: list[str] = []
        self._reasoning_grocery_terms: str = ""

    def run(self) -> Generator[Any, Any, None]:
        self.env.process(self._inbox_listener())
        self.env.process(self._reasoning_loop())
        while True:
            yield self.env.timeout(1e12)

    def _inbox_listener(self) -> Generator[Any, Any, None]:
        while True:
            msg: Message = yield self.inbox.get()
            self._update_context(msg)
            if msg.msg_type == MessageTypes.GrocerySignalUpdate:
                self._broadcast_virtual_shopping_command(msg.payload or {})

    def _reasoning_loop(self) -> Generator[Any, Any, None]:
        while True:
            yield self.env.timeout(self._reasoning_interval)
            self._run_reasoning_cycle()

    def _sim_time_str(self) -> str:
        mod = int(self.env.now) % (24 * 60)
        return f"{mod // 60:02d}:{mod % 60:02d}"

    def _update_context(self, message: Message) -> None:
        if message.msg_type in _CONTEXT_SKIP_TYPES:
            return
        if message.sender_id == self.agent_id:
            return
        summary = self._summarise_message(message)
        entry: dict[str, Any] = {
            "sim_time": int(self.env.now),
            "sim_time_str": self._sim_time_str(),
            "sender": message.sender_id,
            "msg_type": message.msg_type,
            "summary": summary,
        }
        self._context_window.append(entry)
        while len(self._context_window) > self._max_context:
            self._context_window.pop(0)

    def _summarise_message(self, message: Message) -> str:
        mt = message.msg_type
        pl = message.payload or {}
        sender = message.sender_id

        if mt == MessageTypes.CarbonIntensityUpdate:
            cur = pl.get("current", "?")
            band = pl.get("band", "?")
            return f"Grid carbon is {cur} gCO2/kWh ({band})"
        if mt == MessageTypes.WeatherUpdate:
            temp = pl.get("outdoor_temp_c", pl.get("temperature", "?"))
            cond = pl.get("condition", "?")
            return f"Outdoor temperature is {temp}°C, condition: {cond}"
        if mt == MessageTypes.NegotiationResolved:
            val = pl.get("final_value", "?")
            it = pl.get("iterations", "?")
            return f"Thermostat negotiation resolved at {val}°C after {it} rounds"
        if mt == MessageTypes.NegotiationFailed:
            return "Thermostat negotiation failed, fallback applied"
        if mt == MessageTypes.DeviceFailureNotice:
            did = pl.get("device_id", "?")
            return f"Device {did} has failed"
        if mt == MessageTypes.DeviceRecoveryNotice:
            did = pl.get("device_id", "?")
            return f"Device {did} has recovered"
        if mt == MessageTypes.DepartureNotice:
            who = pl.get("name", sender)
            return f"Occupant {who} has left home"
        if mt == MessageTypes.ArrivalNotice:
            who = pl.get("name", sender)
            return f"Occupant {who} has arrived home"
        if mt == MessageTypes.SleepNotice:
            who = pl.get("name", sender)
            return f"Occupant {who} went to sleep"
        if mt == MessageTypes.PreferenceDeclaration:
            prefs = pl.get("preferences") or {}
            val = prefs.get("temperature", "?")
            who = pl.get("person_id", sender)
            fm = pl.get("favorite_meals")
            extra = ""
            if isinstance(fm, list) and fm:
                extra = f" · favorite meals: {', '.join(str(x) for x in fm[:5])}"
            return f"Occupant {who} declared preferred temp {val}°C{extra}"
        if mt == MessageTypes.GrocerySignalUpdate:
            return f"Grocery signal: {pl.get('summary', 'update')}"
        return f"{mt} from {sender}"

    def _broadcast_virtual_shopping_command(self, pl: dict[str, Any]) -> None:
        raw_products = pl.get("products")
        products: list[Any] = raw_products if isinstance(raw_products, list) else []
        items: list[str] = []
        for p in products[:3]:
            if isinstance(p, dict):
                name = p.get("product_name") or p.get("product_name_en") or p.get("name") or ""
                items.append(str(name).strip())
            else:
                items.append(str(p).strip())
        items = [x for x in items if x]
        summary = str(pl.get("summary", ""))
        extra_items = pl.get("ordered_staples") or pl.get("items")
        if isinstance(extra_items, list):
            for x in extra_items[:5]:
                if isinstance(x, str) and x.strip():
                    items.append(x.strip())
        shopping_msg = Message.create(
            self.agent_id,
            "broadcast",
            MessageTypes.ActuationCommand,
            {
                "device_type": "virtual_shopping",
                "action": "suggest_order",
                "items": items[:8],
                "reason": summary,
                "source": "llm_grocery_agent",
            },
            self.env.now,
        )
        self.broadcast(shopping_msg)
        logger.info(
            "LLMSpecialistAgent: virtual shopping ActuationCommand (%s items)",
            len(items),
        )

    def relay_shopping_command_for_grocery_payload(self, pl: dict[str, Any]) -> None:
        self._broadcast_virtual_shopping_command(pl)

    def _meal_block(self) -> str:
        if self._meal_context is None:
            return ""
        return self._meal_context.summary_for_prompt()

    def _build_reasoning_prompt(self) -> str:
        context_lines = "\n".join(
            [
                f"[{e['sim_time_str']}] {e['msg_type']} — {e['summary']}"
                for e in self._context_window
            ]
        )
        registry_summary = self._api_registry.get_summary_for_prompt()
        meal_block = self._meal_block()
        meal_section = (
            f"\nHousehold meals (favorites + recent dinners):\n{meal_block}\n"
            if meal_block.strip()
            else ""
        )
        return f"""You are a specialist agent in a smart home simulation.
Your job is to monitor simulation events and decide if any external real-world 
data would help the home's agents make better decisions right now.

Current simulation time: {self._sim_time_str()}
Recent simulation events (newest last):
{context_lines}
{meal_section}
Available external data sources you may request:
{registry_summary}

When choosing grocery_prices, suggest search_terms that reflect staple ingredients shared across
occupants' favorite meals they have not eaten recently (perishable staples like meat/dairy produce).

Based on the recent events, answer these questions:
1. Is any external data source relevant right now? (yes/no)
2. If yes, which one? (use the exact api_id from the list above, or "none")
3. Why? (one sentence explanation)
4. If api_id is grocery_prices, provide grocery_search_terms: a short search phrase for Open Food Facts (else empty string).

Respond in this exact JSON format and nothing else:
{{"relevant": true/false, "api_id": "string or none", "reason": "string", "grocery_search_terms": "string"}}"""

    def _normalize_llm_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        rel = raw.get("relevant")
        relevant = bool(rel) if isinstance(rel, bool) else str(rel).lower() in ("true", "yes", "1")
        api_id = str(raw.get("api_id", "none")).strip()
        reason = str(raw.get("reason", "")).strip()
        groc = str(raw.get("grocery_search_terms", "")).strip()
        return {"relevant": relevant, "api_id": api_id, "reason": reason, "grocery_search_terms": groc}

    def _record_failure(
        self,
        phase: str,
        api_id: str,
        message: str,
        detail: str | None = None,
    ) -> None:
        if self._metrics:
            self._metrics.log_llm_failure(
                LLMFailureEvent(
                    timestamp=float(self.env.now),
                    phase=phase,
                    api_id=api_id,
                    message=message,
                    detail=detail,
                )
            )

    def execute_api_call(self, api_id: str) -> dict[str, Any] | None:
        api_def = self._api_registry.get(api_id)
        if not api_def:
            self._record_failure("config", api_id, f"Unknown api_id '{api_id}'", None)
            return None

        current_sim_time = float(self.env.now)
        if self._api_registry.is_on_cooldown(api_id, current_sim_time):
            self._record_failure(
                "cooldown",
                api_id,
                f"{api_id} on cooldown — skipped",
                None,
            )
            return None

        params = dict(api_def.params)
        if api_id == "news_disruptions":
            key = os.getenv("NEWSAPI_KEY", "")
            params["apiKey"] = key
            if not key:
                self._record_failure(
                    "config",
                    api_id,
                    "NEWSAPI_KEY not set — news skipped",
                    None,
                )
                return None
        if api_id == "grocery_prices":
            q = self._reasoning_grocery_terms or "staples groceries"
            params["search_terms"] = q[:120]

        url = api_def.base_url.rstrip("/") + api_def.endpoint
        t0 = time.perf_counter()
        try:
            response = httpx.get(url, params=params, timeout=8.0)
            response.raise_for_status()
        except Exception as e:
            self._record_failure("fetch", api_id, f"HTTP request failed: {e}", str(e))
            logger.warning("LLMSpecialistAgent: API %s failed: %s", api_id, e)
            return None

        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_body: dict[str, Any]
        if api_id == "fuel_prices":
            raw_body = {"raw_html": response.text[:8000]}
        else:
            try:
                raw_body = response.json()
            except Exception as e:
                self._record_failure("fetch", api_id, "Response not JSON", str(e))
                return None

        self._api_registry.mark_called(api_id, current_sim_time)

        observation = self._interpret_api_response(api_def, raw_body)
        if not observation:
            return None

        observation["api_id"] = api_id
        observation["latency_ms"] = round(latency_ms, 1)
        observation["source"] = "live"
        return observation

    def _interpret_api_response(self, api_def: ApiDefinition, raw_response: Any) -> dict[str, Any] | None:
        raw_str = json.dumps(raw_response, default=str)[:2000]
        schema_str = _schema_lines_for_prompt(api_def.result_schema)
        prompt = f"""You are interpreting a raw API response for a smart home simulation.

API name: {api_def.name}
API description: {api_def.description}
This result will be emitted as a HALO message of type: {api_def.halo_message_type}

Raw API response (may be truncated):
{raw_str}

Extract the relevant information and return ONLY a JSON object matching this schema (field names and types as described):
{schema_str}

Additional fields to always include:
- "summary": one plain English sentence describing what this data means for the household
- "severity": "low" | "medium" | "high" — how much should this affect agent behaviour
- "halo_message_type": "{api_def.halo_message_type}"
- For grocery_prices: include "products" as a list of objects with product_name when possible
- For fuel HTML: estimate petrol_pence_per_litre and diesel_pence_per_litre if you can infer from text, else use 0

Return ONLY valid JSON. No explanation, no markdown fences."""

        parsed = self._llm_client.complete_json(prompt, max_tokens=400, timeout=10.0)
        if not parsed:
            self._record_failure("interpret", api_def.api_id, "LLM interpretation returned nothing", None)
            return None
        return parsed

    def _emit_halo_message(self, observation: dict[str, Any]) -> None:
        msg_type = str(observation.get("halo_message_type", MessageTypes.LLMObservationUpdate))
        pr = 2 if str(observation.get("severity", "low")).lower() == "high" else 1
        msg = Message.create(
            self.agent_id,
            "broadcast",
            msg_type,
            observation,
            self.env.now,
            priority=pr,
        )
        self.broadcast(msg)
        logger.info(
            "LLMSpecialistAgent: emitted %s — %s",
            msg_type,
            observation.get("summary", ""),
        )
        if msg_type == MessageTypes.GrocerySignalUpdate:
            self.relay_shopping_command_for_grocery_payload(observation)

    def _run_reasoning_cycle(self) -> None:
        if not self._context_window:
            return

        prompt = self._build_reasoning_prompt()
        t0 = time.perf_counter()
        raw = self._llm_client.reason(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = self._normalize_llm_result(raw if isinstance(raw, dict) else {})
        self._reasoning_grocery_terms = result.get("grocery_search_terms", "") or ""
        self._last_reasoning_at = float(self.env.now)

        snapshot = [dict(e) for e in self._context_window]
        event = LLMReasoningEvent(
            timestamp=float(self.env.now),
            context_snapshot=snapshot,
            relevant=result["relevant"],
            api_id=result["api_id"],
            reason=result["reason"],
            llm_latency_ms=float(latency_ms),
        )

        if result["relevant"] and result["api_id"].lower() not in ("none", ""):
            self._pending_api_calls.append(result["api_id"])
            logger.info(
                "LLM queued external API id=%s (pending=%s)",
                result["api_id"],
                self._pending_api_calls,
            )

        pending_snapshot = list(self._pending_api_calls)
        if self._metrics:
            self._metrics.log_llm_reasoning(event, pending_snapshot)

        self._decisions.append(
            {
                "sim_time": self.env.now,
                "result": result,
                "latency_ms": latency_ms,
            }
        )

        notice_payload: dict[str, Any] = {
            "sim_time": int(self.env.now),
            "context_size": len(self._context_window),
            "relevant": result["relevant"],
            "api_id": result["api_id"],
            "reason": result["reason"],
            "pending_calls": pending_snapshot,
        }
        self.broadcast(
            Message.create(
                self.agent_id,
                "broadcast",
                MessageTypes.LLMReasoningNotice,
                notice_payload,
                self.env.now,
            )
        )

        seen: set[str] = set()
        ordered: list[str] = []
        for aid in self._pending_api_calls:
            if aid not in seen:
                seen.add(aid)
                ordered.append(aid)

        for aid in ordered:
            observation = self.execute_api_call(aid)
            if observation:
                self._emit_halo_message(observation)
                if self._metrics:
                    self._metrics.log_llm_api_call(
                        LLMApiCallEvent(
                            timestamp=float(self.env.now),
                            api_id=aid,
                            success=True,
                            observation_summary=str(observation.get("summary", "")),
                            severity=str(observation.get("severity", "low")),
                            halo_message_type=str(observation.get("halo_message_type", "")),
                            latency_ms=float(observation.get("latency_ms", 0.0)),
                        )
                    )
            else:
                if self._metrics:
                    self._metrics.log_llm_api_call(
                        LLMApiCallEvent(
                            timestamp=float(self.env.now),
                            api_id=aid,
                            success=False,
                            observation_summary="",
                            severity="low",
                            halo_message_type="",
                            latency_ms=0.0,
                        )
                    )

        self._pending_api_calls.clear()

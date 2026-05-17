"""Human-in-the-loop bridge: queue → SimPy thread → MessageBus (never from HTTP/stdin thread).

Stdin (``spawn_stdin_command_thread``): ``send-counter <value> <nid> [device_id] [attribute]``,
``send-accept <nid> [device_id] [attribute]``, ``send-reject <nid> [device_id] [reason]``.
Omit optional fields for thermostat ``temperature``; for shower use e.g.
``send-counter 12 <nid> device_shower shower_minutes``.

Contract for queue items (each is a ``dict``):

- ``{"op": "set_pref", "value": <float>}`` — update ``person_cli`` preferred temperature and broadcast
  ``PreferenceDeclaration`` (sender ``person_cli``, recipient ``broadcast``).

- ``{"op": "set_favorite_meals", "meals": [<str>, ...]}`` — update ``person_cli`` favorite dinner dishes
  (2–5 strings stored; used for meal memory + LLM grocery context in ``fused``) and broadcast
  ``PreferenceDeclaration``.

- ``{"op": "simulate_sleep"}`` — broadcast ``SleepNotice`` and record evening meal when meal context is active
  (see ``CliPersonAgent.simulate_sleep``).

- ``{"op": "request_shower"}`` / ``{"op": "request_preheat"}`` — send ``WaterShowerIntent`` / ``WaterPreheatIntent``
  to ``device_shower`` as ``person_cli`` (fused / scenarios with that agent).

- ``{"op": "leave"}`` / ``{"op": "return"}`` — same presence side-effects as ``PersonAgent`` (state + notice).
  For ``CliPersonAgent`` (default **manual_schedule**) there is **no scripted** wake / leave /
  return / sleep clock — inject these (and ``set_pref``) when **you** want the human to move or
  re-declare comfort.

- ``{"op": "send_counter", "value": <float>, "negotiation_id": "<uuid>", "device_id": "device_shower"?, "attribute": "shower_minutes"?}`` —
  send ``NegotiationCounter`` from ``person_cli`` (default target ``device_thermostat`` / ``temperature``; optional fields route to shower duration negotiation).
- ``{"op": "send_accept", "negotiation_id": "<uuid>", "device_id": "device_shower"?, "attribute": "shower_minutes"?}`` — send ``NegotiationAccept``.

- ``{"op": "send_reject", "negotiation_id": "<uuid>", "reason": "..."?, "device_id": "device_shower"? }`` —
  optional ``device_id`` routes to the shower agent when rejecting a shower negotiation.

Allowed ``MessageTypes`` strings for *injected* messages (must match ``negotiation.message.MessageTypes``):

- ``PreferenceDeclaration``, ``DepartureNotice``, ``ArrivalNotice``, ``SleepNotice`` (person → broadcast)
- ``NegotiationAccept``, ``NegotiationCounter``, ``NegotiationReject`` (``person_cli`` → ``device_thermostat`` or ``device_shower``)
- ``WaterShowerIntent``, ``WaterPreheatIntent`` (``person_cli`` → ``device_shower``; from ``request_*`` queue ops)

Stable advocator id: ``person_cli``. Thermostat id in ``cli_bridge`` scenario: ``device_thermostat``.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from typing import Any, Callable

import simpy

from halo_simulation.agents.cli_person import CliPersonAgent
from halo_simulation.negotiation.message import Message, MessageTypes

logger = logging.getLogger(__name__)

CLI_PERSON_ID = "person_cli"
THERMOSTAT_ID = "device_thermostat"
SHOWER_ID = "device_shower"


def status_snapshot(cli: CliPersonAgent) -> dict[str, Any]:
    return {
        "agent_id": cli.agent_id,
        "preferred_temperature": float(cli.state_snapshot.get("preferred_temperature", 0.0)),
        "is_home": bool(cli.state_snapshot.get("is_home", True)),
        "comfort_weight": float(cli.state_snapshot.get("comfort_weight", 0.0)),
        "pending_negotiation": cli.pending_negotiation,
        "sim_now": float(cli.env.now),
    }

# Wall-clock-ish responsiveness: poll queue every this many *simulated* minutes.
DEFAULT_INJECTOR_POLL_SIM_MINUTES = 0.05


def validate_queue_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Return normalized item or None if invalid."""
    if not isinstance(item, dict):
        return None
    op = item.get("op")
    if op == "set_pref":
        try:
            v = float(item["value"])
        except (KeyError, TypeError, ValueError):
            return None
        return {"op": "set_pref", "value": v}
    if op in ("leave", "return"):
        return {"op": op}
    if op == "send_counter":
        try:
            out: dict[str, Any] = {
                "op": "send_counter",
                "value": float(item["value"]),
                "negotiation_id": str(item["negotiation_id"]),
            }
        except (KeyError, TypeError, ValueError):
            return None
        if item.get("device_id") is not None:
            out["device_id"] = str(item["device_id"])
        if item.get("attribute") is not None:
            out["attribute"] = str(item["attribute"])
        return out
    if op == "send_accept":
        try:
            out = {"op": "send_accept", "negotiation_id": str(item["negotiation_id"])}
        except (KeyError, TypeError, ValueError):
            return None
        if item.get("device_id") is not None:
            out["device_id"] = str(item["device_id"])
        if item.get("attribute") is not None:
            out["attribute"] = str(item["attribute"])
        return out
    if op == "send_reject":
        try:
            out = {
                "op": "send_reject",
                "negotiation_id": str(item["negotiation_id"]),
                "reason": str(item.get("reason", "user_reject")),
            }
        except (KeyError, TypeError, ValueError):
            return None
        if item.get("device_id") is not None:
            out["device_id"] = str(item["device_id"])
        return out
    if op == "simulate_sleep":
        return {"op": "simulate_sleep"}
    if op == "set_favorite_meals":
        raw = item.get("meals")
        if not isinstance(raw, list):
            return None
        meals = [str(x).strip() for x in raw if str(x).strip()][:5]
        if not meals:
            return None
        return {"op": "set_favorite_meals", "meals": meals}
    if op == "__status__":
        return {"op": "__status__"}
    if op in ("request_shower", "request_preheat"):
        return {"op": op}
    return None


class BridgeInjector:
    """
    SimPy process: runs only on the simulation thread, drains ``threading.Queue``,
    applies side-effects and sends/broadcasts messages.
    """

    register_on_bus = False
    agent_id = "__bridge_injector"

    def __init__(
        self,
        env: simpy.Environment,
        message_bus: Any,
        inbound: queue.Queue,
        cli_person: CliPersonAgent,
        poll_interval: float = DEFAULT_INJECTOR_POLL_SIM_MINUTES,
        status_reply: queue.Queue | None = None,
    ) -> None:
        self.env = env
        self.bus = message_bus
        self._inbound = inbound
        self._cli = cli_person
        self._poll = poll_interval
        self._status_reply = status_reply

    def run(self):
        # Drain once at t=0 so queued HTTP/CLI commands apply before the first poll interval.
        self._drain_queue()
        while True:
            yield self.env.timeout(self._poll)
            self._drain_queue()

    def _drain_queue(self) -> None:
        while True:
            try:
                raw = self._inbound.get_nowait()
            except queue.Empty:
                break
            item = validate_queue_item(raw) if isinstance(raw, dict) else None
            if item is None:
                logger.warning("BridgeInjector: ignored invalid queue item: %r", raw)
                continue
            self._apply(item)

    def _apply(self, item: dict[str, Any]) -> None:
        op = item["op"]
        if op == "__status__":
            if self._status_reply is not None:
                try:
                    self._status_reply.put_nowait(status_snapshot(self._cli))
                except queue.Full:
                    pass
            return
        if op == "set_pref":
            self._cli.set_preferred_temperature(float(item["value"]))
            self._cli.broadcast_preferences()
        elif op == "set_favorite_meals":
            self._cli.set_favorite_meals(list(item["meals"]))
            self._cli.broadcast_preferences()
        elif op == "simulate_sleep":
            self._cli.simulate_sleep()
        elif op == "leave":
            self._cli.simulate_leave()
        elif op == "return":
            self._cli.simulate_return()
        elif op == "send_counter":
            dev = str(item.get("device_id", THERMOSTAT_ID))
            attr = str(
                item.get("attribute", "temperature" if dev == THERMOSTAT_ID else "shower_minutes")
            )
            self._send_negotiation(
                MessageTypes.NegotiationCounter,
                {
                    "negotiation_id": item["negotiation_id"],
                    "counter_value": float(item["value"]),
                    "device_id": dev,
                    "attribute": attr,
                },
            )
        elif op == "send_accept":
            dev = str(item.get("device_id", THERMOSTAT_ID))
            pl: dict[str, Any] = {
                "negotiation_id": item["negotiation_id"],
                "device_id": dev,
            }
            if item.get("attribute") is not None:
                pl["attribute"] = str(item["attribute"])
            self._send_negotiation(MessageTypes.NegotiationAccept, pl)
        elif op == "send_reject":
            dev = str(item.get("device_id", THERMOSTAT_ID))
            self._send_negotiation(
                MessageTypes.NegotiationReject,
                {
                    "negotiation_id": item["negotiation_id"],
                    "reason": item.get("reason", "user_reject"),
                    "device_id": dev,
                },
            )
        elif op == "request_shower":
            self._to_shower(MessageTypes.WaterShowerIntent)
        elif op == "request_preheat":
            self._to_shower(MessageTypes.WaterPreheatIntent)

    def _to_shower(self, msg_type: str) -> None:
        m = Message.create(
            CLI_PERSON_ID,
            SHOWER_ID,
            msg_type,
            {"initiator": CLI_PERSON_ID},
            self.env.now,
        )
        self.bus.send(m)

    def _send_negotiation(self, msg_type: str, payload: dict[str, Any]) -> None:
        dev = str(payload.get("device_id", THERMOSTAT_ID))
        m = Message.create(
            CLI_PERSON_ID,
            dev,
            msg_type,
            payload,
            self.env.now,
        )
        self.bus.send(m)


def spawn_stdin_command_thread(
    inbound: queue.Queue,
    stop: threading.Event,
    status_reply: queue.Queue | None = None,
    print_banner: Callable[[], None] | None = None,
) -> threading.Thread:
    """Background thread: read stdin lines, push dict ops onto ``inbound`` (never touches bus)."""

    def _loop() -> None:
        if print_banner:
            print_banner()
        while not stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":
                break
            parts = line.strip().split()
            if not parts:
                continue
            cmd = parts[0].lower()
            try:
                if cmd in ("quit", "exit", "q"):
                    stop.set()
                    break
                if cmd == "set-pref" and len(parts) >= 2:
                    inbound.put({"op": "set_pref", "value": float(parts[1])})
                elif cmd == "send-counter" and len(parts) >= 3:
                    d: dict[str, Any] = {
                        "op": "send_counter",
                        "value": float(parts[1]),
                        "negotiation_id": parts[2],
                    }
                    if len(parts) >= 4:
                        d["device_id"] = parts[3]
                    if len(parts) >= 5:
                        d["attribute"] = parts[4]
                    inbound.put(d)
                elif cmd == "send-accept" and len(parts) >= 2:
                    d = {"op": "send_accept", "negotiation_id": parts[1]}
                    if len(parts) >= 3:
                        d["device_id"] = parts[2]
                    if len(parts) >= 4:
                        d["attribute"] = parts[3]
                    inbound.put(d)
                elif cmd == "send-reject" and len(parts) >= 2:
                    d: dict[str, Any] = {"op": "send_reject", "negotiation_id": parts[1]}
                    if len(parts) >= 3:
                        p2 = parts[2]
                        if p2 in ("device_thermostat", "device_shower"):
                            d["device_id"] = p2
                            d["reason"] = parts[3] if len(parts) >= 4 else "user_reject"
                        else:
                            d["reason"] = p2
                    else:
                        d["reason"] = "user_reject"
                    inbound.put(d)
                elif cmd == "leave":
                    inbound.put({"op": "leave"})
                elif cmd == "return":
                    inbound.put({"op": "return"})
                elif cmd == "set-favorite-meals":
                    raw_rest = line.strip()
                    sp = raw_rest.find(" ")
                    payload = raw_rest[sp + 1 :].strip() if sp >= 0 else ""
                    if not payload:
                        print(
                            "set-favorite-meals: pass dishes — comma-separated for multi-word "
                            "(e.g. fish tacos, dal) or space-separated tokens (max 5)",
                        )
                    elif "," in payload:
                        meals = [x.strip() for x in payload.split(",") if x.strip()][:5]
                        if meals:
                            inbound.put({"op": "set_favorite_meals", "meals": meals})
                    else:
                        meals = [str(x).strip() for x in payload.split()[:5] if str(x).strip()]
                        if meals:
                            inbound.put({"op": "set_favorite_meals", "meals": meals})
                        else:
                            print("set-favorite-meals: need at least one dish name")
                elif cmd in ("simulate-sleep", "sleep"):
                    inbound.put({"op": "simulate_sleep"})
                elif cmd == "shower":
                    inbound.put({"op": "request_shower"})
                elif cmd == "preheat":
                    inbound.put({"op": "request_preheat"})
                elif cmd == "status":
                    if status_reply is None:
                        print("status: not available")
                    else:
                        inbound.put({"op": "__status__"})
                        try:
                            snap = status_reply.get(timeout=3.0)
                            print(snap)
                        except queue.Empty:
                            print("status: timeout (sim may not be running yet)")
                else:
                    print(
                        "Unknown command. Try: set-pref, set-favorite-meals, simulate-sleep, shower, "
                        "preheat, send-counter, send-accept, send-reject, leave, return, status, quit"
                    )
            except (IndexError, ValueError) as e:
                print(f"Bad arguments: {e}")

    t = threading.Thread(target=_loop, name="halo-cli-stdin", daemon=True)
    t.start()
    return t

"""Human-in-the-loop bridge: queue → SimPy thread → MessageBus (never from HTTP/stdin thread).

Contract for queue items (each is a ``dict``):

- ``{"op": "set_pref", "value": <float>}`` — update ``person_cli`` preferred temperature and broadcast
  ``PreferenceDeclaration`` (sender ``person_cli``, recipient ``broadcast``).

- ``{"op": "leave"}`` / ``{"op": "return"}`` — same presence side-effects as ``PersonAgent`` (state + notice).

- ``{"op": "send_counter", "value": <float>, "negotiation_id": "<uuid>"}`` — send ``NegotiationCounter`` from
  ``person_cli`` to ``device_thermostat`` (must match an active proposal).

- ``{"op": "send_accept", "negotiation_id": "<uuid>"}`` — send ``NegotiationAccept``.

- ``{"op": "send_reject", "negotiation_id": "<uuid>", "reason": "..." }`` — optional reason; default
  ``below_min_safe`` only if you mirror device min checks externally.

Allowed ``MessageTypes`` strings for *injected* messages (must match ``negotiation.message.MessageTypes``):

- ``PreferenceDeclaration``, ``DepartureNotice``, ``ArrivalNotice``, ``SleepNotice`` (person → broadcast)
- ``NegotiationAccept``, ``NegotiationCounter``, ``NegotiationReject`` (person → ``device_thermostat``)

Stable advocator id: ``person_cli``. Thermostat id in ``cli_bridge`` scenario: ``device_thermostat``.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from typing import Any, Callable

import simpy

from agents.cli_person import CliPersonAgent
from negotiation.message import Message, MessageTypes

logger = logging.getLogger(__name__)

CLI_PERSON_ID = "person_cli"
THERMOSTAT_ID = "device_thermostat"


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
DEFAULT_INJECTOR_POLL_SIM_MINUTES = 0.25


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
            return {
                "op": "send_counter",
                "value": float(item["value"]),
                "negotiation_id": str(item["negotiation_id"]),
            }
        except (KeyError, TypeError, ValueError):
            return None
    if op == "send_accept":
        try:
            return {"op": "send_accept", "negotiation_id": str(item["negotiation_id"])}
        except (KeyError, TypeError, ValueError):
            return None
    if op == "send_reject":
        try:
            return {
                "op": "send_reject",
                "negotiation_id": str(item["negotiation_id"]),
                "reason": str(item.get("reason", "user_reject")),
            }
        except (KeyError, TypeError, ValueError):
            return None
    if op == "__status__":
        return {"op": "__status__"}
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
        elif op == "leave":
            self._cli.simulate_leave()
        elif op == "return":
            self._cli.simulate_return()
        elif op == "send_counter":
            self._send_negotiation(
                MessageTypes.NegotiationCounter,
                {
                    "negotiation_id": item["negotiation_id"],
                    "counter_value": float(item["value"]),
                    "device_id": THERMOSTAT_ID,
                    "attribute": "temperature",
                },
            )
        elif op == "send_accept":
            self._send_negotiation(
                MessageTypes.NegotiationAccept,
                {
                    "negotiation_id": item["negotiation_id"],
                    "device_id": THERMOSTAT_ID,
                },
            )
        elif op == "send_reject":
            self._send_negotiation(
                MessageTypes.NegotiationReject,
                {
                    "negotiation_id": item["negotiation_id"],
                    "reason": item.get("reason", "user_reject"),
                    "device_id": THERMOSTAT_ID,
                },
            )

    def _send_negotiation(self, msg_type: str, payload: dict[str, Any]) -> None:
        m = Message.create(
            CLI_PERSON_ID,
            THERMOSTAT_ID,
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
                    inbound.put(
                        {"op": "send_counter", "value": float(parts[1]), "negotiation_id": parts[2]}
                    )
                elif cmd == "send-accept" and len(parts) >= 2:
                    inbound.put({"op": "send_accept", "negotiation_id": parts[1]})
                elif cmd == "send-reject" and len(parts) >= 2:
                    reason = parts[2] if len(parts) >= 3 else "user_reject"
                    inbound.put({"op": "send_reject", "negotiation_id": parts[1], "reason": reason})
                elif cmd == "leave":
                    inbound.put({"op": "leave"})
                elif cmd == "return":
                    inbound.put({"op": "return"})
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
                        "Unknown command. Try: set-pref, send-counter, send-accept, send-reject, "
                        "leave, return, status, quit"
                    )
            except (IndexError, ValueError) as e:
                print(f"Bad arguments: {e}")

    t = threading.Thread(target=_loop, name="halo-cli-stdin", daemon=True)
    t.start()
    return t

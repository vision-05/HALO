"""Abstract base for all HALO simulation agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Generator, TYPE_CHECKING, Tuple

import simpy

from negotiation.message import Message

if TYPE_CHECKING:
    from metrics.collector import MetricsCollector
    from negotiation.message import MessageBus

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    All inter-agent coordination goes through the message bus.
    Internal state uses the `_state` dict; use the `state` property for a read-only copy only.
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        env: simpy.Environment,
        message_bus: MessageBus,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.env = env
        self.message_bus = message_bus
        self._metrics = metrics
        self._state: dict[str, Any] = {}
        self.inbox = simpy.Store(env)

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @abstractmethod
    def run(self):
        ...

    def send(self, recipient_id: str, message: Message) -> None:
        self.message_bus.send(message)

    def broadcast(self, message: Message) -> None:
        self.message_bus.broadcast(message)

    def receive(self) -> simpy.events.Process:
        return self.inbox.get()

    def wait_inbox_or_timeout(
        self, timeout_duration: float
    ) -> Generator[Any, Any, Tuple[Any, Any]]:
        """
        Wait for the next inbox message or until timeout. If the timeout wins,
        cancel the pending Store get — SimPy does not cancel it automatically, and
        an orphaned Get would consume the next put without waking this process.
        """
        get_ev = self.inbox.get()
        timeout_ev = self.env.timeout(timeout_duration)
        res = yield get_ev | timeout_ev
        if get_ev not in res and not get_ev.triggered:
            get_ev.cancel()
        return res, get_ev

    def drain_inbox_burst(self, handler):
        """
        After a minute-boundary timeout wins over inbox.get(), other processes may
        still post to the inbox at the same SimPy time. Yield once so those puts run,
        then drain with timeout(0) until no co-timestamp delivery remains.
        """
        yield self.env.timeout(0)
        while True:
            res, get_ev = yield from self.wait_inbox_or_timeout(0.0)
            if get_ev in res:
                handler(res[get_ev])
            else:
                break

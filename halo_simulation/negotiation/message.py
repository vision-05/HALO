"""Message types and P2P message bus (routing only, no decisions)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from metrics.collector import MetricsCollector

import config

logger = logging.getLogger(__name__)


class MessageTypes:
    PreferenceDeclaration = "PreferenceDeclaration"
    DepartureNotice = "DepartureNotice"
    ArrivalNotice = "ArrivalNotice"
    SleepNotice = "SleepNotice"
    NegotiationProposal = "NegotiationProposal"
    NegotiationAccept = "NegotiationAccept"
    NegotiationCounter = "NegotiationCounter"
    NegotiationReject = "NegotiationReject"
    NegotiationResolved = "NegotiationResolved"
    NegotiationFailed = "NegotiationFailed"
    CarbonIntensityUpdate = "CarbonIntensityUpdate"
    WeatherUpdate = "WeatherUpdate"
    SpecialistUnavailable = "SpecialistUnavailable"
    DeviceFailureNotice = "DeviceFailureNotice"
    DeviceRecoveryNotice = "DeviceRecoveryNotice"
    ActuationCommand = "ActuationCommand"


@dataclass
class Message:
    msg_id: str
    sender_id: str
    recipient_id: str  # or "broadcast"
    msg_type: str
    payload: dict[str, Any]
    timestamp: float
    priority: int = 1

    @classmethod
    def create(
        cls,
        sender_id: str,
        recipient_id: str,
        msg_type: str,
        payload: dict[str, Any],
        env_now: float,
        priority: int = 1,
    ) -> Message:
        return cls(
            msg_id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            msg_type=msg_type,
            payload=payload,
            timestamp=env_now,
            priority=priority,
        )


class RegisterableAgent(Protocol):
    """Minimal interface for bus registration."""

    agent_id: str

    def receive_for_bus(self) -> Any: ...


class MessageBus:
    """Routes messages to agent inboxes; logs to MetricsCollector when provided."""

    def __init__(
        self,
        env: Any,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.env = env
        self._metrics = metrics
        self._registry: dict[str, Any] = {}

    def register(self, agent: Any) -> None:
        self._registry[agent.agent_id] = agent

    def unregister(self, agent_id: str) -> None:
        self._registry.pop(agent_id, None)

    def _put(self, agent: Any, message: Message) -> Any:
        """SimPy Store.put must be yielded from a process."""
        yield self.env.timeout(config.MESSAGE_BUS_SEND_DELAY)
        yield agent.inbox.put(message)

    def send(self, message: Message) -> None:
        if message.recipient_id == "broadcast":
            self.broadcast(message)
            return
        target = self._registry.get(message.recipient_id)
        if target is None:
            logger.warning("No agent registered for recipient %s", message.recipient_id)
            return
        self.env.process(self._put(target, message))
        self._log_route(message)

    def broadcast(self, message: Message) -> None:
        for aid, agent in self._registry.items():
            if aid == message.sender_id:
                continue
            self.env.process(self._put(agent, message))
        self._log_route(message)

    def _log_route(self, message: Message) -> None:
        if self._metrics is None:
            return
        self._metrics.log_message_routed(
            {
                "msg_id": message.msg_id,
                "sender_id": message.sender_id,
                "recipient_id": message.recipient_id,
                "msg_type": message.msg_type,
                "timestamp": message.timestamp,
            }
        )

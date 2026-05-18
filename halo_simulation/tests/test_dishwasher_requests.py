"""Dishwasher run requests, LLM/heuristic gate, and delay negotiation."""

import numpy as np
import simpy

from halo_simulation.agents.device_agent import DishwasherDeviceAgent
from halo_simulation.metrics.collector import MetricsCollector
from halo_simulation.negotiation.message import Message, MessageBus, MessageTypes


def test_dishwasher_run_request_triggers_cycle_heuristic(monkeypatch):
    def _no_llm_json(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "halo_simulation.agents.device_agent.LLMClient.complete_json",
        _no_llm_json,
    )

    metrics = MetricsCollector("test_dw_req")
    env = simpy.Environment()
    bus = MessageBus(env, metrics=metrics)
    rng = np.random.default_rng(0)
    dish = DishwasherDeviceAgent("device_dishwasher", env, bus, rng, metrics, scenario_name="test_dw")
    bus.register(dish)
    env.process(dish.run())
    env.run(until=0.01)
    bus.send(
        Message.create(
            "specialist_carbon",
            "broadcast",
            MessageTypes.CarbonIntensityUpdate,
            {"current": 120.0, "band": "medium", "forecast_4h": [118.0, 115.0, 90.0, 88.0]},
            1.0,
        )
    )
    env.run(until=2.0)
    bus.send(
        Message.create(
            "person_alice",
            "device_dishwasher",
            MessageTypes.DishwasherRunRequest,
            {"requester_id": "person_alice", "urgency": 0.7},
            3.0,
        )
    )
    env.run(until=600.0)
    assert dish._energy_kwh > 0.0
    assert dish._state["device_state"] == "idle"


def test_dishwasher_bypasses_llm_after_pending_age(monkeypatch):
    """After repeated approve=false, old enough pending uses heuristic so the run does not stall."""
    monkeypatch.setattr("halo_simulation.config.DISHWASHER_USE_LLM_SCHEDULE", True)
    monkeypatch.setattr(
        "halo_simulation.config.DISHWASHER_APPROVE_FALSE_OVERRIDE_AFTER_SIM_MIN",
        40.0,
    )
    monkeypatch.setattr(
        "halo_simulation.config.DISHWASHER_DECLINED_RETRY_SIM_MINUTES",
        5.0,
    )

    def fake_llm(*args, **kwargs):
        return {"approve": False, "defer_minutes": 0.0, "reason": "test_decline"}

    monkeypatch.setattr(
        "halo_simulation.agents.device_agent.LLMClient.complete_json",
        fake_llm,
    )

    metrics = MetricsCollector("test_dw_esc")
    env = simpy.Environment()
    bus = MessageBus(env, metrics=metrics)
    rng = np.random.default_rng(1)
    dish = DishwasherDeviceAgent("device_dishwasher", env, bus, rng, metrics, scenario_name="test_dw_esc")
    bus.register(dish)
    env.process(dish.run())
    env.run(until=0.01)
    bus.send(
        Message.create(
            "specialist_carbon",
            "broadcast",
            MessageTypes.CarbonIntensityUpdate,
            {"current": 120.0, "band": "medium", "forecast_4h": [118.0, 115.0, 90.0, 88.0]},
            0.0,
        )
    )
    env.run(until=1.0)
    bus.send(
        Message.create(
            "person_alice",
            "device_dishwasher",
            MessageTypes.DishwasherRunRequest,
            {"requester_id": "person_alice", "urgency": 0.7},
            5.0,
        )
    )
    env.run(until=200.0)
    assert dish._energy_kwh > 0.0


def test_dishwasher_two_requesters_batch_heuristic_no_llm():
    """Default path skips LLM; two pending ids are served in one cycle (negotiation off by default)."""
    metrics = MetricsCollector("test_dw_2req")
    env = simpy.Environment()
    bus = MessageBus(env, metrics=metrics)
    rng = np.random.default_rng(2)
    dish = DishwasherDeviceAgent("device_dishwasher", env, bus, rng, metrics, scenario_name="test_dw_2")
    bus.register(dish)
    env.process(dish.run())
    env.run(until=0.01)
    bus.send(
        Message.create(
            "specialist_carbon",
            "broadcast",
            MessageTypes.CarbonIntensityUpdate,
            {"current": 120.0, "band": "medium", "forecast_4h": [118.0, 115.0, 90.0, 88.0]},
            1.0,
        )
    )
    env.run(until=2.0)
    bus.send(
        Message.create(
            "person_alice",
            "device_dishwasher",
            MessageTypes.DishwasherRunRequest,
            {"requester_id": "person_alice", "urgency": 0.9},
            3.0,
        )
    )
    bus.send(
        Message.create(
            "person_bob",
            "device_dishwasher",
            MessageTypes.DishwasherRunRequest,
            {"requester_id": "person_bob", "urgency": 0.6},
            3.5,
        )
    )
    env.run(until=600.0)
    assert dish._energy_kwh > 0.0
    assert dish._state["device_state"] == "idle"
    assert not dish._pending

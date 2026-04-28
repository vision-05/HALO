"""Integration-style scenario smoke tests."""

import ast
import inspect
import os

import pytest

from halo_simulation.agents.base_agent import BaseAgent
from halo_simulation.scenarios.carbon_spike import CarbonSpikeScenario
from halo_simulation.scenarios.device_failure import DeviceFailureScenario
from halo_simulation.scenarios.temperature_conflict import TemperatureConflictScenario


def test_temperature_conflict_runs():
    sc = TemperatureConflictScenario(seed=42, days=2)
    stats = sc.run_simulation()
    assert stats["total_negotiations"] >= 0


def test_carbon_spike_runs():
    sc = CarbonSpikeScenario(seed=1, days=2)
    stats = sc.run_simulation()
    assert stats["total_negotiations"] >= 0


def test_device_failure_runs():
    sc = DeviceFailureScenario(seed=2, days=2)
    stats = sc.run_simulation()
    assert stats["total_failures"] >= 0


def test_convergence_rate_14_days():
    sc = TemperatureConflictScenario(seed=99, days=14)
    stats = sc.run_simulation()
    n = stats["total_negotiations"]
    if n == 0:
        pytest.skip("No negotiations in run")
    assert stats["convergence_rate_pct"] > 80.0


def test_no_cross_agent_private_state_access():
    root = os.path.join(os.path.dirname(__file__), "..", "agents")
    offenders: list[str] = []
    for name in os.listdir(root):
        if not name.endswith(".py") or name == "__init__.py":
            continue
        path = os.path.join(root, name)
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.attr.startswith("_") and node.value.id not in ("self", "cls"):
                    offenders.append(f"{path}:{node.lineno}")
    assert not offenders, "Agents should not read other instances' private attributes: " + str(offenders)


def test_base_agent_state_is_copy():
    class Dummy(BaseAgent):
        def run(self):
            if False:
                yield None

    import simpy
    from halo_simulation.negotiation.message import MessageBus

    env = simpy.Environment()
    bus = MessageBus(env)
    a = Dummy("a", "person", env, bus)
    a._state["x"] = 1
    s = a.state
    s["x"] = 999
    assert a._state["x"] == 1

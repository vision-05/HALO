"""Human bridge: injector not on bus; queue validation."""

from __future__ import annotations

import queue

from halo_simulation.human_bridge import THERMOSTAT_ID, validate_queue_item
from halo_simulation.scenarios.cli_bridge import CliBridgeScenario


def test_validate_queue_item() -> None:
    assert validate_queue_item({"op": "set_pref", "value": 21.5}) == {"op": "set_pref", "value": 21.5}
    assert validate_queue_item({"op": "leave"}) == {"op": "leave"}
    assert validate_queue_item({"op": "send_counter", "value": 20.0, "negotiation_id": "abc"}) is not None
    assert validate_queue_item({"op": "nope"}) is None


def test_injector_not_registered_on_bus() -> None:
    iq: queue.Queue = queue.Queue()
    sc = CliBridgeScenario(seed=1, days=1, inject_queue=iq)
    sc.build()
    sc.register_all()
    ids = set(sc.bus._registry.keys())
    assert "__bridge_injector" not in ids
    assert "person_cli" in ids
    assert THERMOSTAT_ID in ids


def test_cli_bridge_runs_headless() -> None:
    iq: queue.Queue = queue.Queue()
    sc = CliBridgeScenario(seed=2, days=1, inject_queue=iq)
    stats = sc.run_simulation()
    assert "total_negotiations" in stats

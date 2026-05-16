from __future__ import annotations

import asyncio
import json
from importlib import import_module
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "agents" / "rl" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Ensure repository root is on sys.path so sibling packages (e.g. discovery, RL) import correctly
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
AGENTS_DIR = ROOT_DIR / "agents"
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

RoutineRLAgent = import_module("rl_agent").RoutineRLAgent


def test_route_prediction_builds_language_agent_payload(tmp_path: Path) -> None:
    agent = object.__new__(RoutineRLAgent)
    agent.name = "RLAgent"
    agent.model_path = tmp_path / "model.zip"
    agent.profile_path = tmp_path / "profile.json"
    agent.activity_log_path = tmp_path / "activity.jsonl"
    agent.activity_log_path.parent.mkdir(parents=True, exist_ok=True)
    agent._last_prediction_payload = None
    agent.state = {}
    agent.outbound_socks = {}
    agent._json_safe = RoutineRLAgent._json_safe.__get__(agent, RoutineRLAgent)
    agent._append_activity_log = RoutineRLAgent._append_activity_log.__get__(agent, RoutineRLAgent)
    agent._build_llm_request = RoutineRLAgent._build_llm_request.__get__(agent, RoutineRLAgent)
    agent._predict_action = lambda: (3, "Recommend Movie Selection", {"current_time_hours": 20.0}, [0.0] * 12)
    agent.send_msg = lambda dest, payload: asyncio.sleep(0)

    record = asyncio.run(agent.route_prediction())

    assert record["action_name"] == "Recommend Movie Selection"
    assert record["message"]["target"] == "LanguageAgent"
    assert record["message"]["action"] == "self_prompt"
    assert record["message"]["params"]["schedule"] == "every_15_minutes"
    assert "predicted_action" in record["message"]["params"]
    assert agent.activity_log_path.exists()

    lines = agent.activity_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["action_name"] == "Recommend Movie Selection"


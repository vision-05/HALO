from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rl_agent import RoutineRLAgent
from RL.PPO import DEFAULT_MODEL_PATH, DEFAULT_PROFILE_PATH
from RL.online_update import DEFAULT_ACTIVITY_LOG_PATH

AGENT_NAME = os.environ.get("AGENT_NAME", "RLAgent")
AGENT_ROLE = os.environ.get("AGENT_ROLE", "Planner")
PROFILE_PATH = Path(os.environ.get("RL_PROFILE_PATH", str(DEFAULT_PROFILE_PATH)))
MODEL_PATH = Path(os.environ.get("RL_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
LOG_PATH = Path(os.environ.get("RL_ACTIVITY_LOG_PATH", str(DEFAULT_ACTIVITY_LOG_PATH)))
PREDICTION_INTERVAL = int(os.environ.get("RL_PREDICTION_INTERVAL_MINUTES", "15"))
MONTHLY_WINDOW_DAYS = int(os.environ.get("RL_MONTHLY_WINDOW_DAYS", "30"))
MONTHLY_TIMESTEPS = int(os.environ.get("RL_MONTHLY_TIMESTEPS", "2000"))
MONTHLY_LR = os.environ.get("RL_MONTHLY_LR", "0.00005")
BOOTSTRAP_TIMESTEPS = int(os.environ.get("RL_BOOTSTRAP_TIMESTEPS", "25000"))
SEED = int(os.environ.get("RL_SEED", "42"))


def _parse_optional_float(value: str) -> float | None:
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


async def main() -> None:
    agent = RoutineRLAgent(
        name=AGENT_NAME,
        role=AGENT_ROLE,
        profile_path=PROFILE_PATH,
        model_path=MODEL_PATH,
        activity_log_path=LOG_PATH,
        prediction_interval_minutes=PREDICTION_INTERVAL,
        monthly_window_days=MONTHLY_WINDOW_DAYS,
        monthly_timesteps=MONTHLY_TIMESTEPS,
        monthly_learning_rate=_parse_optional_float(MONTHLY_LR),
        bootstrap_timesteps=BOOTSTRAP_TIMESTEPS,
        seed=SEED,
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())


"""Load a trained PPO policy and nudge the thermostat on the SimPy clock (live stream)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from halo_simulation import config
from halo_simulation.rl.driver import ACTION_DELTAS
from halo_simulation.rl.observation import build_temperature_rl_observation

logger = logging.getLogger(__name__)


def _sb3_load_path(model_path: str) -> str:
    """Path passed to ``PPO.load``.

    SB3 convention is to call ``PPO.load("foo")`` for a file ``foo.zip``. If both ``foo`` and
    ``foo.zip`` exist (e.g. an extracted checkpoint folder), ``foo`` is a directory and loading
    fails with ``IsADirectoryError`` — in that case pass the explicit ``.zip`` path.
    """
    p = Path(model_path).expanduser().resolve()
    if p.suffix.lower() == ".zip" and p.is_file():
        stem = p.with_suffix("")
        if stem.exists() and stem.is_dir():
            return str(p)
        return str(stem)
    return str(p)


def attach_rl_thermostat_sidecar(
    scenario: Any,
    emit: Callable[[str, dict[str, Any]], None],
    model_path: str,
    step_minutes: float | None = None,
) -> None:
    """Schedule a SimPy process: every *step_minutes*, obs → PPO.predict → ``apply_rl_comfort_delta``."""
    step = float(
        step_minutes
        if step_minutes is not None
        else os.getenv("HALO_RL_THERMOSTAT_STEP_MIN", "15").strip() or "15"
    )
    horizon = float(config.MINUTES_PER_DAY * scenario.days)
    load_path = _sb3_load_path(model_path)
    stochastic = os.getenv("HALO_RL_THERMOSTAT_STOCHASTIC", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    try:
        from stable_baselines3 import PPO
    except ImportError as e:  # pragma: no cover
        exe = sys.executable
        logger.error(
            "stable_baselines3 missing for the Python running this process (%s). "
            "Install into that exact environment, then restart uvicorn.",
            exe,
        )
        raise RuntimeError(
            f"stable_baselines3 not installed for Python {exe!r}. "
            f"Run: {exe} -m pip install -r halo_simulation/rl/requirements-rl.txt "
            "(same interpreter as `python -m uvicorn …`)."
        ) from e

    model = PPO.load(load_path)
    logger.info("RL thermostat sidecar: loaded PPO from %s", load_path)

    def _find_thermostat() -> Any:
        for a in scenario.agents:
            if getattr(a, "agent_id", None) == "device_thermostat":
                return a
        return None

    def sidecar():
        env = scenario.env
        while env.now < horizon - 1e-9:
            thermo = _find_thermostat()
            if thermo is None:
                logger.warning("RL sidecar: no thermostat; stopping sidecar")
                return
            try:
                obs = build_temperature_rl_observation(scenario)
            except Exception:
                logger.exception("RL sidecar: observation build failed")
                rem = min(step, max(0.0, horizon - env.now))
                if rem <= 1e-9:
                    return
                yield env.timeout(rem)
                continue

            action, _states = model.predict(obs, deterministic=not stochastic)
            ai = int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
            ai = max(0, min(ai, len(ACTION_DELTAS) - 1))
            delta = ACTION_DELTAS[ai]
            info = thermo.apply_rl_comfort_delta(delta)
            emit(
                "rl_thermostat",
                {
                    "timestamp": float(env.now),
                    "sim_time": float(env.now),
                    "action_index": ai,
                    "delta_c": float(delta),
                    "apply": info,
                },
            )

            rem = min(step, max(0.0, horizon - env.now))
            if rem <= 1e-9:
                return
            yield env.timeout(rem)

    scenario.env.process(sidecar())
    logger.info(
        "RL thermostat sidecar: scheduled every %.1f sim minutes until t=%.0f",
        step,
        horizon,
    )

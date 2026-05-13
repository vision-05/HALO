#!/usr/bin/env python3
"""Train PPO on :class:`HaloTemperatureRlEnv` (HALO SimPy thermostat hook).

Run from repo root with ``PYTHONPATH=`` set::

    export PYTHONPATH=.
    pip install -r halo_simulation/rl/requirements-rl.txt
    python -m halo_simulation.rl.train_ppo_halo --timesteps 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from halo_simulation.rl.driver import ACTION_DELTAS
from halo_simulation.rl.gym_env import HaloTemperatureRlEnv


def make_halo_env(step_minutes: float = 15.0, seed: int = 0) -> HaloTemperatureRlEnv:
    return HaloTemperatureRlEnv(step_minutes=step_minutes, seed=seed)


def main() -> int:
    p = argparse.ArgumentParser(description="PPO on HaloTemperatureRlEnv (SimPy temperature conflict).")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--step-minutes", type=float, default=15.0)
    p.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "ppo_halo_thermo",
        help="Path prefix for saved SB3 model (no extension).",
    )
    p.add_argument("--skip-env-check", action="store_true")
    args = p.parse_args()

    env = make_halo_env(step_minutes=args.step_minutes, seed=args.seed)
    if not args.skip_env_check:
        check_env(env, warn=True)

    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, ent_coef=0.01)
    print(f"Training PPO for {args.timesteps} timesteps...")
    model.learn(total_timesteps=int(args.timesteps))
    model.save(str(args.model))
    print(f"Saved model to {args.model}.zip (SB3 default)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

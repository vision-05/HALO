#!/usr/bin/env python3
"""Sanity-check the RL driver without installing stable-baselines3."""

from __future__ import annotations

import argparse

import numpy as np

from halo_simulation.rl.driver import ACTION_DELTAS, TemperatureRlDriver


def main() -> int:
    p = argparse.ArgumentParser(description="Run random actions on TemperatureRlDriver (no Gymnasium required).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=96, help="Max steps (96 x 15min = 1 day)")
    p.add_argument("--step-minutes", type=float, default=15.0)
    args = p.parse_args()
    d = TemperatureRlDriver(step_minutes=args.step_minutes)
    obs = d.reset(args.seed)
    total = 0.0
    for i in range(args.steps):
        a = int(np.random.randint(0, len(ACTION_DELTAS)))
        obs, r, term, trunc, info = d.step(a)
        total += float(r)
        if term or trunc:
            print(f"Episode ended at step {i + 1}, total reward {total:.3f}, sim_time={info.get('sim_time')}")
            break
    else:
        print(f"Ran {args.steps} steps without termination; total reward {total:.3f}")
    print("Last obs (9 floats):", obs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

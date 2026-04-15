#!/usr/bin/env python3
"""CLI entry point for HALO simulation scenarios."""

from __future__ import annotations

import argparse
import logging
import sys

from halo_simulation import config
from halo_simulation.scenarios.carbon_spike import CarbonSpikeScenario
from halo_simulation.scenarios.device_failure import DeviceFailureScenario
from halo_simulation.scenarios.temperature_conflict import TemperatureConflictScenario


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _run_one(name: str, seed: int, days: int | None) -> dict:
    if name == "temperature_conflict":
        d = days if days is not None else 14
        sc = TemperatureConflictScenario(seed=seed, days=d)
    elif name == "carbon_spike":
        d = days if days is not None else 7
        sc = CarbonSpikeScenario(seed=seed, days=d)
    elif name == "device_failure":
        d = days if days is not None else 7
        sc = DeviceFailureScenario(seed=seed, days=d)
    else:
        raise ValueError(f"Unknown scenario: {name}")
    return sc.run_simulation()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HALO SimPy simulation")
    parser.add_argument(
        "--scenario",
        choices=["temperature_conflict", "carbon_spike", "device_failure", "all"],
        default="temperature_conflict",
    )
    parser.add_argument("--days", type=int, default=None, help="Override default days for scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args(argv)

    _configure_logging(args.debug)

    names = (
        ["temperature_conflict", "carbon_spike", "device_failure"]
        if args.scenario == "all"
        else [args.scenario]
    )

    for name in names:
        stats = _run_one(name, args.seed, args.days)
        print(f"\n=== Scenario: {name} ===")
        print(f"Total negotiations: {stats['total_negotiations']}")
        print(f"Convergence rate: {stats['convergence_rate_pct']:.1f}%")
        print(f"Mean iterations (converged): {stats['mean_iterations']:.2f}")
        print(f"Mean satisfaction: {stats['mean_satisfaction']:.3f}")
        print(f"Total failures: {stats['total_failures']}")
        print(f"Recovery rate: {stats['recovery_rate_pct']:.1f}%")
        print(f"Outputs: {stats.get('output_paths', ())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

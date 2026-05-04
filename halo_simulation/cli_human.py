#!/usr/bin/env python3
"""Interactive CLI human bridge for ``cli_bridge`` (stdin → ``threading.Queue`` → SimPy ``BridgeInjector``).
PYTHONPATH=. python -m halo_simulation.cli_human --days 2 --seed 42 --demo-wall-seconds 60


**Dashboard / SSE:** start the FastAPI server, pick scenario ``cli_bridge``, then POST JSON commands
to ``/api/inject`` (same keys as the queue: ``op`` + fields). Example::

    curl -s -X POST http://127.0.0.1:8000/api/inject \\
      -H 'Content-Type: application/json' \\
      -d '{\"op\":\"set_pref\",\"value\":21.5}'

``GET /stream?scenario=cli_bridge&...`` must be running so the inject queue is active.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time

import config
from human_bridge import spawn_stdin_command_thread
from scenarios.cli_bridge import CliBridgeScenario


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _banner() -> None:
    print(
        """
HALO human bridge — type commands (one per line):
  set-pref <deg>              — preferred °C + broadcast PreferenceDeclaration
  leave                       — away + DepartureNotice
  return                      — home + ArrivalNotice + preferences
  send-counter <deg> <nid>    — NegotiationCounter (nid from `status`)
  send-accept <nid>
  send-reject <nid> [reason]
  status                      — snapshot (includes pending negotiation id)
  quit                        — stop simulation thread
"""
    )


def run_interactive(seed: int, days: int, debug: bool, warmup_sec: float, demo_wall_seconds: float = 0.0) -> int:
    _configure_logging(debug)
    inbound: queue.Queue = queue.Queue()
    status_reply: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()

    sc = CliBridgeScenario(seed, days, inbound, status_reply=status_reply)

    _banner()
    # Stdin thread must start before any `input()` on the main thread — otherwise the first
    # command line (e.g. "status") is consumed as the prompt answer and never reaches the queue.
    spawn_stdin_command_thread(inbound, stop, status_reply=status_reply, print_banner=None)
    if warmup_sec > 0 and sys.stdin.isatty():
        print(
            f"\n>>> Command reader is live. Simulated time starts in {warmup_sec:.0f}s (wall clock) — "
            "type `set-pref`, `leave`, or `return` now; they run when the sim starts.\n"
            ">>> `status` needs the sim running: use it **after** you see `>>> Starting simulation…`.\n",
            flush=True,
        )
        time.sleep(float(warmup_sec))
        print(">>> Starting simulation…\n", flush=True)
    elif warmup_sec > 0:
        time.sleep(float(warmup_sec))

    def sim_worker() -> None:
        sc.build()
        sc.register_all()
        sc.start_processes()
        until = float(config.MINUTES_PER_DAY * days)
        chunk = float(config.STREAM_STOP_CHECK_CHUNK_MINUTES)
        try:
            while sc.env.now < until - 1e-9:
                if stop.is_set():
                    print("Simulation stop requested.")
                    break
                nxt = min(sc.env.now + chunk, until)
                if nxt <= sc.env.now:
                    break
                t0 = float(sc.env.now)
                sc.env.run(until=nxt)
                advanced = float(sc.env.now) - t0
                if demo_wall_seconds > 0.0 and until > 1e-9 and advanced > 1e-12 and not stop.is_set():
                    delay = demo_wall_seconds * (advanced / until)
                    if delay > 0:
                        time.sleep(delay)
            if not stop.is_set():
                paths = sc.metrics.save_outputs()
                stats = sc.metrics.summary_stats()
                stats["output_paths"] = paths
                print("\n=== Run finished ===")
                for k, v in stats.items():
                    if k != "output_paths":
                        print(f"  {k}: {v}")
                print(f"  outputs: {paths}")
        except Exception as exc:
            print(f"Simulation error: {exc}", file=sys.stderr)
            raise

    t = threading.Thread(target=sim_worker, name="halo-simpy-cli", daemon=False)
    t.start()

    try:
        while t.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop.set()
    t.join(timeout=5.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HALO CLI human bridge (cli_bridge scenario)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--days", type=int, default=2, help="Simulated days (keep small for interactive demo)")
    p.add_argument(
        "--warmup",
        type=float,
        default=8.0,
        metavar="SEC",
        help="Real seconds before SimPy starts so you can type commands into the queue (stdin). Use 0 to skip.",
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="Same as --warmup 0 (for scripts; sim starts immediately).",
    )
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--demo-wall-seconds",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Stretch the full simulated run across this many wall-clock seconds (0 = as fast as possible). "
        "Example: 60 for an audience demo.",
    )
    args = p.parse_args(argv)
    w = 0.0 if args.no_warmup else max(0.0, float(args.warmup))
    return run_interactive(
        args.seed,
        args.days,
        args.debug,
        warmup_sec=w,
        demo_wall_seconds=max(0.0, min(float(args.demo_wall_seconds), float(config.DEMO_WALL_SECONDS_MAX))),
    )


if __name__ == "__main__":
    raise SystemExit(main())

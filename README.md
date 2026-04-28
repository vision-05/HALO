# HALO
Holistic Living Agent Orchestration, an agentic AI powered home automation system

Usage:
Testing agents:
docker compose up language-test

Running agent
docker compose up language

Run documentation
docker compose up docs

**Holistic Living Agent Orchestration** — a multi-agent simulation of a home where people, devices, and specialists (grid carbon, weather) coordinate through a message bus. Negotiations, preferences, and schedules run under [SimPy](https://simpy.readthedocs.io/).

## Requirements

- Python 3.10+ (see project environment)
- Dependencies for the simulation and dashboard: `fastapi`, `uvicorn`, `sse-starlette`, `simpy`, `numpy`, `pandas`, `matplotlib`, and others as used by `halo_simulation/`

Install from the repo root as you normally do for this project (e.g. `pip install -e .` or your chosen `requirements` workflow).

## Run the live dashboard

From the repository root:

```bash
PYTHONPATH=. uvicorn halo_simulation.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser. Use **single worker** (default `uvicorn` without `--workers`) so the SSE stream and inject API share one process.

### Scenarios

| Scenario | Role |
|----------|------|
| `temperature_conflict` | Two scripted occupants + thermostat + specialists |
| `carbon_spike` | Includes dishwasher timing vs carbon |
| `device_failure` | Adds device failure / recovery paths |
| **`cli_bridge`** | **CLI Human Bridge** — Bob is scripted; **`person_cli`** is controlled by you (UI or CLI) |

### Human-in-the-loop (`cli_bridge`)

1. Choose **CLI Human Bridge**, set **Days** (short runs are easier to follow), optional **Pace (s)** to stretch the full run over that many wall-clock seconds (good for demos).
2. Click **Run** so `GET /stream?scenario=cli_bridge&...` is active.
3. Use the **Human inject** strip (same contract as `POST /api/inject`).

**Inject contract:** JSON body with an `op` field; see the module docstring in `halo_simulation/human_bridge.py` for allowed operations (`set_pref`, `leave`, `return`, `send_counter`, `send_accept`, `send_reject`) and fields. The server validates with `validate_queue_item()`.

**Stream query parameters (among others):** `scenario`, `days`, `seed`, `live_data`, `demo_wall_seconds` — when `demo_wall_seconds` is greater than zero, the simulation worker sleeps between chunks so the full run lasts about that many real-time seconds.

### CLI human bridge (no browser)

```bash
PYTHONPATH=. python -m halo_simulation.cli_human --days 2 --seed 42 --demo-wall-seconds 60
```

Stdin commands are documented in `halo_simulation/cli_human.py` (e.g. `set-pref`, `leave`, `return`, `send-counter`, `send-accept`, `status`, `quit`).

## Tests

```bash
PYTHONPATH=. pytest halo_simulation/tests/
```

## Layout

| Path | Purpose |
|------|---------|
| `halo_simulation/server.py` | FastAPI app: static UI, `GET /stream` (SSE), `POST /api/inject`, results API |
| `halo_simulation/ui/index.html` | Live dashboard |
| `halo_simulation/human_bridge.py` | Queue contract, `BridgeInjector` (SimPy-side queue drain → bus) |
| `halo_simulation/scenarios/cli_bridge.py` | `cli_bridge` scenario wiring |
| `halo_simulation/agents/cli_person.py` | `person_cli` agent (manual negotiation path) |
| `halo_simulation/cli_human.py` | Interactive stdin → queue → same scenario as UI |
| `halo_simulation/config.py` | Tunables (time, negotiation, pacing cap, etc.) |

# HALO — `smart_simulation` branch

Holistic Living Agent Orchestration: a SimPy house (people, thermostat, dishwasher/shower in some scenarios, grid carbon and weather specialists, negotiations over the message bus) with a **browser dashboard** served by FastAPI. This branch folds in **LLM-assisted specialist behaviour** and an **optional PPO thermostat policy** that can run during the same live SSE stream you already use for demos.

If you only skim one thing: the simulation runs in a **worker thread**; the browser talks **HTTP + Server-Sent Events**; human injects and RL nudges both land on that same SimPy world, but they enter through different paths (queue vs sidecar process).

---

## What runs where

When you hit **Run** in the UI (or open `GET /stream`), the server starts a **daemon thread** that builds a scenario object, registers agents on a shared `simpy.Environment`, and steps time in chunks so a disconnect can stop the run. Each interesting change is pushed to an `asyncio` queue; the SSE handler drains that queue and forwards JSON as named events (`message`, `negotiation`, `agent_state`, …).

**Human-in-the-loop** (`cli_bridge` or `fused`): the thread owns a `queue.Queue`. `POST /api/inject` validates JSON and enqueues commands; a small SimPy-side drainer posts to the bus. That only works while a stream is active and the server is single-process (one worker) so the queue pointer on `app.state` matches the running thread.

**RL thermostat sidecar** (optional): if `HALO_RL_THERMOSTAT_MODEL` is set at server import time (shell env or repo-root `.env`), the same thread—after `start_processes()`—schedules an extra SimPy process that wakes every *N* simulated minutes, builds the same 9-float observation used in training, calls `PPO.predict`, and applies a comfort setpoint nudge. Failures to load SB3 or the checkpoint are logged; the stream keeps going without RL.

---

## Install

From repo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Add RL stack only if you train or use live PPO:

```bash
pip install -r halo_simulation/rl/requirements-rl.txt
```

`requirements.txt` pulls in the web stack (`fastapi`, `uvicorn[standard]`, `sse-starlette`) alongside `simpy`, `httpx`, `numpy`, etc. The RL extras pull `gymnasium` and `stable-baselines3` (and thus Torch).

**Practical note:** install and run the server with the **same** `python` (e.g. always `python -m uvicorn …` inside the venv). Mixing conda’s `uvicorn` with a venv where only SB3 was installed is an easy way to get `ModuleNotFoundError` at runtime even though “it worked in a notebook once.”

Python 3.10+; we’ve been running 3.13 locally.

---

## Configuration

**`.env`** lives next to this README. It is read when `halo_simulation.server` imports (`python-dotenv`), and it **does not override** variables already exported in your shell—so a blank `export FOO=` can accidentally block a `.env` value.

**LLM** keys: `ANTHROPIC_API_KEY`, `CLAUDE_API_KEY`, or `CLAUDE_KEY` (first non-empty wins, see `config.anthropic_api_key()`). Gateway selection is all in `halo_simulation/config.py`: `LLM_PROTOCOL` switches between native Anthropic Messages (`/v1/messages`) and OpenAI-shaped chat (`/v1/chat/completions`) for proxies like LiteLLM. You need a reachable base URL for the openai mode (`LITELLM_BASE_URL` or an explicit `LLM_OPENAI_CHAT_URL`). Bearer vs `x-api-key` is controlled with `LLM_AUTH_STYLE`.

**RL live inference:**

| Variable | Purpose |
|----------|---------|
| `HALO_RL_THERMOSTAT_MODEL` | Filesystem path to SB3 save (`.zip` ok). |
| `HALO_RL_THERMOSTAT_STEP_MIN` | Sim minutes between sidecar wakes (default 15). |
| `HALO_RL_THERMOSTAT_STOCHASTIC` | If `1` / `true` / `yes`, use stochastic `predict` (demo variety only; not a substitute for a good checkpoint). |

If you have both `checkpoint.zip` and an extracted folder `checkpoint/` next to it, SB3’s usual “stem without .zip” load path would point at a directory; `live_inference._sb3_load_path` detects that and passes the `.zip` explicitly.

More RL-focused commands and training defaults: `halo_simulation/rl/README.md`.

---

## Run the dashboard

```bash
export PYTHONPATH=.
python -m uvicorn halo_simulation.server:app --host 127.0.0.1 --port 8000
```

Open the printed host. **Do not** run multiple Uvicorn workers for this app: the inject queue and stream state assume a single process.

Stream URL shape:

`GET /stream?scenario=<name>&days=<n>&seed=<n>&live_data=<bool>&demo_wall_seconds=<float>`

With `demo_wall_seconds > 0`, the worker sleeps between SimPy slices so the full horizon stretches to roughly that many wall-clock seconds (capped in `config.DEMO_WALL_SECONDS_MAX`). Handy for HITL demos; set `0` for full speed.

### Scenarios (query param `scenario=`)

| Value | What you get |
|-------|----------------|
| `temperature_conflict` | Alice + Bob + thermostat + weather/carbon specialists. **This is what the thermostat PPO was trained on.** |
| `carbon_spike` | Adds dishwasher vs carbon storyline. |
| `device_failure` | Failure and recovery paths. |
| `cli_bridge` | Bob scripted; `person_cli` is you (inject / CLI). |
| `fused` | Full stack: Alice, Bob, `person_cli`, thermostat, dishwasher, shower, specialists, forced evening carbon spike. Same inject contract as `cli_bridge`. |

Inject body format: `halo_simulation/human_bridge.py` (`validate_queue_item`). `CliPersonAgent` defaults to `manual_schedule=True`, so commute-style behaviour does not advance unless you drive it (or you change that flag where the agent is constructed).

Headless inject driver:

```bash
PYTHONPATH=. python -m halo_simulation.cli_human --scenario fused --days 2 --demo-wall-seconds 60
```

---

## Thermostat RL — train vs live (read this before blaming the UI)

There are **two** RL stories in the repo:

1. **`RL/`** — older **JSON profile** MDP (`RL/environment.py`, `RL/PPO.py`, `SmartHomeEnv`). Different state, different actions, different saved weights. Still there for comparison; it is **not** what feeds `HALO_RL_THERMOSTAT_MODEL` unless you deliberately wire that up.

2. **`halo_simulation/rl/`** — thermostat control on the **real SimPy** temperature-conflict world.

Training uses `HaloTemperatureRlEnv` → `TemperatureRlDriver`: each PPO step applies one of three nudges (−0.5 / 0 / +0.5 °C) to the thermostat **comfort** setpoint (`ThermostatDeviceAgent.apply_rl_comfort_delta`), then runs SimPy forward **15 simulated minutes** (configurable `step_minutes`). One gym episode is **one simulated day** (`TemperatureConflictScenario` with `days=1` in the driver reset). Observation is **9 floats** in `[-1,1]` from `build_temperature_rl_observation` (time encoding, normalized temps, carbon signal, negotiation flag, Alice/Bob home). Reward after each macro-step is **negative weighted mean absolute error** between indoor temperature and at-home occupants’ preferred temps (from the thermostat’s `_preferences`), minus a small term `0.05 * (last carbon / CARBON_HIGH_THRESHOLD)` — see `TemperatureRlDriver._reward` in `driver.py`. PPO only ever sees that scalar plus transitions; nobody “votes” on the reward at runtime.

Train with:

```bash
export PYTHONPATH=.
python -m halo_simulation.rl.train_ppo_halo --timesteps <N> --model ./my_prefix
```

which writes `my_prefix.zip` by SB3 convention. The trainer sets `ent_coef=0.01` by default to reduce the “always pick the same discrete action” collapse you get with zero entropy and short runs.

**Live stream:** the sidecar reuses **`build_temperature_rl_observation`** on whatever scenario is running (`fused`, etc.), loads your zip with `PPO.load`, and emits SSE `rl_thermostat`. The UI shows those as pink-accent **RLThermostatNudge** rows and extends the thermostat chart. There is **no reward** in production—only `predict` + apply. If the policy was trained on `temperature_conflict` but you stream `fused`, expect possible **distribution shift**; the wiring is still doing what you asked.

---

## LLM specialist (short)

`LLMSpecialistAgent` wakes on a SimPy clock, calls `LLMClient.reason` for a structured “should I call an external API?” decision, may `httpx.get` registered endpoints (Open-Meteo, Open Food Facts, gov.uk fuel page HTML, NewsAPI if `NEWSAPI_KEY` is set), then asks `complete_json` to turn responses into HALO broadcast messages. Protocol details and header styles live in `llm_client.py` / `config.py`. SSE exposes reasoning, API calls, and pipeline errors separately so the UI can show failures without breaking the stream.

---

## External HTTP the sim / specialist may hit

- **Carbon:** `https://api.carbonintensity.org.uk` (`/intensity`, `/intensity/date/{date}`, optional `/intensity/pt24h` for forecasts).
- **Weather / charts:** `https://api.open-meteo.com/v1/forecast` (London defaults; `live_data` uses `ExternalDataClient`; `GET /api/weather_series` pulls a longer hourly window for the chart).
- **LLM:** whatever host you configure (Anthropic direct, LiteLLM, etc.).
- **Specialist registry:** gov.uk fuel publication page (HTML), Open Food Facts search JSON, NewsAPI `everything` (keyed), plus Open-Meteo again for “severe weather” style pulls.

The browser also loads Chart.js from cdnjs (see `ui/index.html`).

---

## Tests

```bash
PYTHONPATH=. pytest halo_simulation/tests/
```

---

## Where to edit what

| Area | Start here |
|------|------------|
| FastAPI routes, stream thread, RL attach | `halo_simulation/server.py` |
| Dashboard / SSE client | `halo_simulation/ui/index.html` |
| Tunables, LLM URL helpers | `halo_simulation/config.py` |
| Inject contract | `halo_simulation/human_bridge.py` |
| Fused wiring | `halo_simulation/scenarios/fused.py` |
| LLM behaviour | `halo_simulation/agents/llm_specialist_agent.py` |
| LLM HTTP + JSON parsing | `halo_simulation/external/llm_client.py` |
| Registered external APIs for the specialist | `halo_simulation/external/api_registry.py` |
| Thermostat RL train + live sidecar | `halo_simulation/rl/` |

---

## Rough edges (so you don’t burn an afternoon)

- **Venv vs conda:** SB3 missing in the process that serves uvicorn almost always means “wrong Python,” not “SB3 doesn’t exist on Earth.” Align `which python` with the interpreter you use for `python -m uvicorn`.

- **Collapsed PPO:** a checkpoint that always chooses +0.5°C under `deterministic=True` even on the training driver is a **bad training run**, not proof the sidecar is broken. Train longer, watch entropy, compare checkpoints; use `HALO_RL_THERMOSTAT_STOCHASTIC` only if you need visible variety for a demo.

- **LiteLLM JSON:** models sometimes wrap JSON in prose; the client now extracts the first `{…}` object when strict `json.loads` fails. If you still see parse errors, the model is emitting invalid JSON inside the object.

- **Carbon `pt24h` 400:** seen from the UK API occasionally; code falls back.

- **OpenFoodFacts 503:** their outage, not ours.

- **Inject 503:** no active `cli_bridge`/`fused` stream (or run ended). Short `days` + run still “live” fixes it.

- **Matplotlib cache warnings** in CI or sandboxes: set `MPLCONFIGDIR` to something writable if import noise matters.

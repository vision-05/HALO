# HALO — `smart_simulation` branch

Holistic Living Agent Orchestration: a SimPy house (people, thermostat, dishwasher/shower in some scenarios, grid carbon and weather specialists, negotiations over the message bus) with a **browser dashboard** served by FastAPI. This branch includes **LLM-assisted specialist behaviour** on the same live SSE stream you use for demos.

If you only skim one thing: the simulation runs in a **worker thread**; the browser talks **HTTP + Server-Sent Events**; human injects land on that SimPy world through a queue the stream thread owns.

---

## What runs where

When you hit **Run** in the UI (or open `GET /stream`), the server starts a **daemon thread** that builds a scenario object, registers agents on a shared `simpy.Environment`, and steps time in chunks so a disconnect can stop the run. Each interesting change is pushed to an `asyncio` queue; the SSE handler drains that queue and forwards JSON as named events (`message`, `negotiation`, `agent_state`, …).

**Human-in-the-loop** (`cli_bridge` or `fused`): the thread owns a `queue.Queue`. `POST /api/inject` validates JSON and enqueues commands; a small SimPy-side drainer posts to the bus. That only works while a stream is active and the server is single-process (one worker) so the queue pointer on `app.state` matches the running thread.

---

## Install

From repo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls in the web stack (`fastapi`, `uvicorn[standard]`, `sse-starlette`) alongside `simpy`, `httpx`, `numpy`, etc.

**Practical note:** install and run the server with the **same** `python` (e.g. always `python -m uvicorn …` inside the venv).

Python 3.10+; we’ve been running 3.13 locally.

---

## Configuration

**`.env`** lives next to this README. It is read when `halo_simulation.server` imports (`python-dotenv`), and it **does not override** variables already exported in your shell—so a blank `export FOO=` can accidentally block a `.env` value.

**LLM** keys: `ANTHROPIC_API_KEY`, `CLAUDE_API_KEY`, or `CLAUDE_KEY` (first non-empty wins, see `config.anthropic_api_key()`). Gateway selection is all in `halo_simulation/config.py`: `LLM_PROTOCOL` switches between native Anthropic Messages (`/v1/messages`) and OpenAI-shaped chat (`/v1/chat/completions`) for proxies like LiteLLM. You need a reachable base URL for the openai mode (`LITELLM_BASE_URL` or an explicit `LLM_OPENAI_CHAT_URL`). Bearer vs `x-api-key` is controlled with `LLM_AUTH_STYLE`.

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
| `temperature_conflict` | Alice + Bob + thermostat + weather/carbon specialists. |
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

## LLM specialist (short)

`LLMSpecialistAgent` wakes on a SimPy clock, calls `LLMClient.reason` for a structured “should I call an external API?” decision, may `httpx.get` registered endpoints (Open-Meteo, Open Food Facts, gov.uk fuel page HTML, NewsAPI if `NEWSAPI_KEY` or `NEWS_API` is set), then asks `complete_json` to turn responses into HALO broadcast messages. Protocol details and header styles live in `llm_client.py` / `config.py`. SSE exposes reasoning, API calls, and pipeline errors separately so the UI can show failures without breaking the stream.

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
| FastAPI routes, stream thread | `halo_simulation/server.py` |
| Dashboard / SSE client | `halo_simulation/ui/index.html` |
| Tunables, LLM URL helpers | `halo_simulation/config.py` |
| Inject contract | `halo_simulation/human_bridge.py` |
| Fused wiring | `halo_simulation/scenarios/fused.py` |
| LLM behaviour | `halo_simulation/agents/llm_specialist_agent.py` |
| LLM HTTP + JSON parsing | `halo_simulation/external/llm_client.py` |
| Registered external APIs for the specialist | `halo_simulation/external/api_registry.py` |

---

## Rough edges

- **LiteLLM JSON:** models sometimes wrap JSON in prose; the client now extracts the first `{…}` object when strict `json.loads` fails. If you still see parse errors, the model is emitting invalid JSON inside the object.

- **Carbon `pt24h` 400:** seen from the UK API occasionally; code falls back.

- **OpenFoodFacts 503:** their outage, not ours.

- **Inject 503:** no active `cli_bridge`/`fused` stream (or run ended). Short `days` + run still “live” fixes it.

- **Matplotlib cache warnings** in CI or sandboxes: set `MPLCONFIGDIR` to something writable if import noise matters.

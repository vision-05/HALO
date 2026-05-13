# HALO — **`smart_simulation` branch**

**Holistic Living Agent Orchestration**: SimPy home simulation (people, devices, carbon/weather specialists, negotiations) plus **FastAPI + SSE live UI**, **human-in-the-loop** (`cli_bridge` / `fused`), **LLM specialist** (Anthropic Messages or OpenAI-compatible / LiteLLM), and **optional PPO thermostat control** in the live stream.

---

## Install (repo root)

```bash
cd /path/to/HALO
python -m venv .venv && source .venv/bin/activate   # recommended
pip install -r requirements.txt
pip install -r halo_simulation/rl/requirements-rl.txt   # only if you train PPO or use HALO_RL_THERMOSTAT_MODEL
```

**`requirements.txt`** includes: `simpy`, `fastapi`, `uvicorn[standard]`, `sse-starlette`, `httpx`, `numpy`, `pandas`, `matplotlib`, `pytest`, `python-dotenv`.  
**`halo_simulation/rl/requirements-rl.txt`**: `gymnasium`, `stable-baselines3` (+ PyTorch stack).

Python **3.10+** (project currently tested on **3.13** in dev). Use **one** interpreter for both `pip` and `python -m uvicorn` (see *Current issues*).

---

## Configuration

| Mechanism | Detail |
|-----------|--------|
| **`.env`** | Optional file at **repo root** (`HALO/.env`, sibling of this README). Loaded on **`halo_simulation.server` import** via `python-dotenv`; **does not override** variables already set in the shell. |
| **LLM** | `ANTHROPIC_API_KEY`, `CLAUDE_API_KEY`, or `CLAUDE_KEY` (first non-empty wins for key material). Protocol and URLs: see `halo_simulation/config.py` — `LLM_PROTOCOL` (`anthropic` \| `openai`), `LLM_MODEL`, `LITELLM_BASE_URL` / `LLM_ANTHROPIC_BASE_URL`, `LLM_OPENAI_CHAT_URL`, `LLM_MESSAGES_URL`, `LLM_AUTH_STYLE` (`x-api-key` vs `bearer`), etc. |
| **RL live stream** | `HALO_RL_THERMOSTAT_MODEL` — filesystem path to SB3 `PPO` save (`.zip` allowed). `HALO_RL_THERMOSTAT_STEP_MIN` (default `15` sim minutes). `HALO_RL_THERMOSTAT_STOCHASTIC`=`1`/`true`/`yes` → `predict(..., deterministic=False)`. If both `foo.zip` and extracted folder `foo/` exist, loader uses the **`.zip`** explicitly (avoids `IsADirectoryError`). |

Full RL train / env nuance: **`halo_simulation/rl/README.md`**.

---

## Run the dashboard

```bash
export PYTHONPATH=.
python -m uvicorn halo_simulation.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Use a **single** Uvicorn worker (no `--workers`) so **`GET /stream`** and **`POST /api/inject`** share one process and one `inject_queue`.

**Stream (SSE)** — `GET /stream?scenario=…&days=…&seed=…&live_data=…&demo_wall_seconds=…`  
`demo_wall_seconds>0` scales **wall-clock sleeps** between SimPy chunks so the full horizon lasts ~N seconds (cap in `halo_simulation/config.py` → `DEMO_WALL_SECONDS_MAX`).

### Scenarios

| `scenario` | Notes |
|------------|--------|
| `temperature_conflict` | Scripted occupants + thermostat + specialists (matches **`HaloTemperatureRlEnv`** / PPO training distribution). |
| `carbon_spike` | Dishwasher vs carbon. |
| `device_failure` | Failure / recovery paths. |
| **`cli_bridge`** | **`person_cli`** = human (UI inject or CLI); Bob scripted. |
| **`fused`** | Full demo: Alice + Bob + **`person_cli`**, thermostat, dishwasher, shower, specialists, **evening carbon spike**; same **Human inject** contract as `cli_bridge`. |

**Human inject** — only while `cli_bridge` or `fused` stream is active. `POST /api/inject` JSON validated by `validate_queue_item()` in `halo_simulation/human_bridge.py` (`set_pref`, `leave`, `return`, negotiation ops, `set_favorite_meals`, `simulate_sleep`, …). **`CliPersonAgent`**: default `manual_schedule=True` (no auto commute; your injects drive presence/prefs).

**CLI (no browser)** — `python -m halo_simulation.cli_human --scenario cli_bridge|fused …` (stdin protocol in `cli_human.py`).

---

## LLM + metrics (this branch)

- **`LLMSpecialistAgent`** — periodic reasoning + structured tool use; HTTP via `halo_simulation/external/llm_client.py` (`reason`, `complete_json`).
- **OpenAI/LiteLLM path** — `POST …/v1/chat/completions`; requires **`LITELLM_BASE_URL`** (origin) or full **`LLM_OPENAI_CHAT_URL`** unless you use native Anthropic protocol.
- **SSE event types** include `llm_reasoning`, `llm_observation`, `llm_api_call`, `llm_pipeline_error`, `agent_state`, `message`, `negotiation`, `failure`, `learning`, `carbon`, `weather`, `api_status`, **`rl_thermostat`** (when RL sidecar is attached).

---

## RL (training vs live)

| Topic | Location |
|-------|----------|
| Gym env | `halo_simulation/rl/gym_env.py` — `HaloTemperatureRlEnv` wraps `TemperatureRlDriver`. |
| Obs vector (9 floats, `[-1,1]`) | `halo_simulation/rl/observation.py` — `build_temperature_rl_observation`; training driver uses the same builder. |
| Actions | `ACTION_DELTAS` in `halo_simulation/rl/driver.py`: `−0.5`, `0`, `+0.5` °C on thermostat **comfort** setpoint (`ThermostatDeviceAgent.apply_rl_comfort_delta`; skipped while negotiation in progress). |
| Train PPO | `python -m halo_simulation.rl.train_ppo_halo --timesteps … --model ./prefix` (saves `prefix.zip`). Trainer uses `ent_coef=0.01` by default to reduce trivial collapse. |
| Live sidecar | `run_simulation_thread` in `halo_simulation/server.py`: after `start_processes()`, if `HALO_RL_THERMOSTAT_MODEL` is set, `attach_rl_thermostat_sidecar` schedules SimPy `env.process` → obs → `PPO.predict` → apply → SSE `rl_thermostat`. UI feed type **`RLThermostatNudge`** (pink accent). |

---

## Tests

```bash
PYTHONPATH=. pytest halo_simulation/tests/
```

---

## Layout (high-signal paths)

| Path | Role |
|------|------|
| `halo_simulation/server.py` | FastAPI: static UI, SSE `/stream`, `/api/inject`, APIs; dotenv; scenario thread + RL attach. |
| `halo_simulation/ui/index.html` | Dashboard, EventSource handlers, charts, feed. |
| `halo_simulation/config.py` | Tunables + LLM URL/auth helpers. |
| `halo_simulation/human_bridge.py` | Inject queue contract + validation. |
| `halo_simulation/scenarios/fused.py` | `fused` scenario. |
| `halo_simulation/agents/llm_specialist_agent.py` | LLM cycles + API registry calls. |
| `halo_simulation/external/llm_client.py` | Anthropic vs OpenAI-compatible HTTP + JSON extraction. |
| `halo_simulation/rl/` | Driver, env, train script, **`live_inference.py`** sidecar. |

---

## Current issues (this workspace / recent runs)

1. **Interpreter mismatch** — `stable_baselines3` / `uvicorn` must be installed in the **same** environment as `python -m uvicorn`. A shell one-liner with only `import os` does **not** load `.env`; the server loads repo-root `.env` on import. Prefer `python -m uvicorn` (not a global `uvicorn` from another env) after `source .venv/bin/activate`.

2. **PPO checkpoint quality** — `my_halo_ppo.zip` under deterministic `predict` **collapsed to always action +0.5°C** even on the **training** `TemperatureRlDriver` trajectory; that is a **bad training outcome**, not a thermostat wiring bug. Mitigations: retrain with **more timesteps**, monitor entropy / reward; optional **`HALO_RL_THERMOSTAT_STOCHASTIC=1`** for varied (stochastic) actions in demos only.

3. **LiteLLM / JSON interpretation** — `complete_json` occasionally failed on strict `json.loads` of the full model string; client now parses the **first JSON object** via `JSONDecoder.raw_decode` and supports list-shaped OpenAI `content` parts. Residual failures = truly invalid JSON inside the object.

4. **UK carbon public API** — `GET …/intensity/pt24h` sometimes returns **400**; sim continues with degraded/fallback behaviour.

5. **OpenFoodFacts** — live grocery lookup intermittently **503** (remote service).

6. **`POST /api/inject` 503** — expected when **no** active `cli_bridge`/`fused` stream (or run already finished); start `/stream` with short **days** for HITL demos.

7. **Coexisting `my_halo_ppo.zip` + `my_halo_ppo/` directory** — previously broke SB3 load stem; **fixed** in `_sb3_load_path` (`halo_simulation/rl/live_inference.py`) by preferring the `.zip` when the stem path is a directory.

8. **Matplotlib / font cache warnings** — benign in headless/sandboxed runs; set **`MPLCONFIGDIR`** to a writable dir if imports spam stderr.

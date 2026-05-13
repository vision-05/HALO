# HALO × RL (`smart_simulation` branch)

This folder wires **Proximal Policy Optimization (PPO)** experiments to a **JSON routine profile** (`SmartHomeEnv`).  
For **Option 2** (control the **simulated house**), use the **`halo_simulation.rl`** package instead of `RL/environment.py`.

## 1. Branch

Checkout **`smart_simulation`** — it merges **`fused-sim`** simulation code with the **`RL/`** training scripts from the RL branch.

## 2. Install

```bash
cd /path/to/HALO
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r halo_simulation/rl/requirements-rl.txt
```

## 3. Sanity-check the new RL hook (no PPO)

Runs **random thermostat nudges** on a **one-day temperature conflict** SimPy scenario:

```bash
export PYTHONPATH=.
python halo_simulation/rl/run_driver_demo.py --seed 1
```

## 4. Gymnasium env

```bash
export PYTHONPATH=.
python -c "from halo_simulation.rl.gym_env import HaloTemperatureRlEnv; e=HaloTemperatureRlEnv(); o,_=e.reset(seed=0); o2,r,d,t,i=e.step(1); print(r,d, i)"
```

- **Actions:** `0 = −0.5°C`, `1 = hold`, `2 = +0.5°C` on the thermostat **comfort setpoint** (see `ACTION_DELTAS` in `driver.py`).
- **Observation:** 9 floats in `[-1, 1]` (time-of-day, temps, carbon, negotiation flag, Alice/Bob home).

## 5. Train PPO (HALO sim — ready to run)

```bash
export PYTHONPATH=.
python -m halo_simulation.rl.train_ppo_halo --timesteps 50000 --model ./my_halo_ppo
```

This uses **`HaloTemperatureRlEnv`** (3 thermostat nudges, 9-dim observation). Increase `--timesteps` for stronger policies.

**Alternative:** in **`RL/PPO.py`**, change **`make_env()`** to return **`HaloTemperatureRlEnv()`** instead of **`SmartHomeEnv(profile_path)`**, then run `python RL/PPO.py train ...` as before.

The legacy **`RL/environment.py`** profile MDP remains available for comparison.

## 6. Live simulation stream (UI + SimPy sidecar)

If **`HALO_RL_THERMOSTAT_MODEL`** is set when you start the FastAPI server, each streamed run attaches a SimPy sidecar that loads a **Stable-Baselines3 PPO** checkpoint and periodically nudges **`device_thermostat`** (same 9-float observation and **`ACTION_DELTAS`** as training). The UI receives SSE events **`rl_thermostat`** and updates the thermostat chart and feed.

- **`HALO_RL_THERMOSTAT_MODEL`** — path to the saved model. If the file ends in **`.zip`**, pass that path; loading uses the path **without** the `.zip` suffix (SB3 convention).
- **`HALO_RL_THERMOSTAT_STEP_MIN`** — optional; default **`15`** (sim minutes between sidecar steps).

Example:

```bash
export PYTHONPATH=.
export HALO_RL_THERMOSTAT_MODEL=./my_halo_ppo.zip
# optional: export HALO_RL_THERMOSTAT_STEP_MIN=15
uvicorn halo_simulation.server:app --host 127.0.0.1 --port 8000
```

Then open the UI, pick a scenario that includes **`device_thermostat`**, and click **Run**. The policy was trained on **`TemperatureConflictScenario`** (`HaloTemperatureRlEnv`); other scenarios still run, but observation statistics may differ from training.

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

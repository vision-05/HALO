# PPO Smart Home RL

## Files
- `environment.py` — custom Gymnasium environment built from `routine_learning/user_routine_profile.json`
- `PPO.py` — PPO training and inference entrypoint

## Install
```bash
pip install gymnasium stable-baselines3 numpy
```

## Train
```bash
python3 /Users/laki/PycharmProjects/HALO/RL/PPO.py train --timesteps 100000
```

## Predict
```bash
python3 /Users/laki/PycharmProjects/HALO/RL/PPO.py predict --current-hour 17.5 --day-name Friday --season Winter --rain-intensity 0.0 --low-stock-count 2
```


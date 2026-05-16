# RL Agent

PPO-backed HALO agent that predicts a household action every 15 minutes and forwards it to `LanguageAgent` for routing.

## What it does

- loads the current user routine profile from `routine_learning/user_routine_profile.json`
- loads or bootstraps a PPO model from `RL/smart_home_ppo_agent`
- predicts an action every 15 minutes
- forwards a schema-compliant routing prompt to `LanguageAgent`
- refreshes the routine profile and fine-tunes the PPO model every 30 days from `db/rl_activity_log.jsonl`

## Environment variables

- `AGENT_NAME` default `RLAgent`
- `AGENT_ROLE` default `Planner`
- `RL_PROFILE_PATH`
- `RL_MODEL_PATH`
- `RL_ACTIVITY_LOG_PATH`
- `RL_PREDICTION_INTERVAL_MINUTES`
- `RL_MONTHLY_WINDOW_DAYS`
- `RL_MONTHLY_TIMESTEPS`
- `RL_MONTHLY_LR`
- `RL_BOOTSTRAP_TIMESTEPS`
- `RL_SEED`

## Run

```bash
docker compose up rl-agent
```

For local execution:

```bash
python agents/rl/src/main.py
```


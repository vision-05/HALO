# Habit Learning and RL

## 1. Overview
This module is responsible for analyzing raw user activity data to build a **User Routine Profile** (Habit Learning) and training a **Reinforcement Learning (RL) agent** using Proximal Policy Optimization (PPO) to automate smart home decisions. 

The system acts as a "Decision Engine" that receives state data—such as time, weather, and household stock—to output actions that can be executed by physical actuators or suggested via Telegram/LLM interfaces.

---

## 2. Habit Learning: JSON Profile Schema
The habit learning script (`smart_data_generation.py`) processes CSV event logs into a structured JSON profile (`user_routine_profile.json`). This profile defines the user behavior used to build the RL Environment.

### Key Schema Sections:
* **wake_up_patterns / weekday_occupancy**: Contains Kernel Density Estimation (KDE) profiles for wake, leave, and return times.
* **heating_preferences**: Defines seasonal target temperatures extracted from user history.
* **weather_correlation**: Rules defining how environmental factors like rain intensity shift routines (e.g., delaying a dog walk).
* **markov_transition_matrix**: Probabilities of moving between activity states, such as moving from `WakeUp` to `LeaveHome`.

### Schema Example:
```json
{
  "wake_up_patterns": { "weekday_kde_profile": { "grid_hours": [...], "density": [...] } },
  "weekday_occupancy_pattern": { "usual_leave_hour": 8.5, "usual_return_hour": 17.17 },
  "heating_preferences": { "Winter": "22", "Summer": "Off" },
  "dog_walking_habits": { "weather_shift_rule": { "threshold": 0.6, "expected_shift_hours": -7.5 } },
  "markov_transition_matrix": { "states": [...], "transition_matrix": { ... } }
}

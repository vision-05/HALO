# Habit Learning and RL

## Overview
This module is responsible for analyzing raw user activity data to build a **User Routine Profile** (Habit Learning) and training a **Reinforcement Learning (RL) agent** using Proximal Policy Optimization (PPO) to automate smart home decisions. 

The system acts as a "Decision Engine" that receives state data—such as time, weather, and household stock—to output actions that can be executed by physical actuators or suggested via Telegram/LLM interfaces.

---

## Habit Learning: JSON Profile Schema
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
```
## RL Workflow & Integration
The RL agent uses a Proximal Policy Optimization (PPO) algorithm to make automated decisions based on the learned user routine.

### Integration Pipeline
1.  **Habit Learning**: The system processes raw event logs (CSV) to generate a `user_routine_profile.json` using Kernel Density Estimation (KDE).
2.  **Model Update**: The `online_update.py` script performs a "rolling window" update. It refreshes the user profile based on the last 30 days of data and fine-tunes the RL model to adapt to recent habit changes.
3.  **Inference (Decision Making)**: The main system controller calls the prediction function with the current environmental state to receive the optimal action.

**Example Inference Call:**
```python
from PPO import load_trained_model, predict_next_action, make_env

# Load trained model and environment
model = load_trained_model("smart_home_ppo_agent")
env = make_env("user_routine_profile.json")

# Define current state: (hour, day_name, season, rain_intensity, low_stock_count)
current_state = (17.5, "Friday", "Winter", 0.0, 1)

# Get prediction
action_index, action_name = predict_next_action(model, env, state=current_state)
print(f"Action to Execute: {action_name} (Index: {action_index})")
```

## Action Space Specification
The RL agent outputs a discrete action index ranging from **0 to 9**. These indices correspond to specific smart home interventions and must be mapped accordingly within the physical system or simulation.

| Index | Action Name | Logic and Reward Context |
| :--- | :---: | :--- |
| **0** | Reduce Heating When Out | Triggered when the probability of the user being home is low based on the occupancy profile. |
| **1** | Preheat Before Return Home | Activated when the probability of user return within 1 hour exceeds a learned threshold. |
| **2** | Preheat to Preferred Temp | Sets the thermostat to the user's seasonal preference (e.g., 22°C for Winter). |
| **3** | Recommend Movie Selection | Triggered during high-density "Movie Night" windows, specifically optimized for Fridays. |
| **4** | Set Movie Settings (Dim Lights) | Adjusts lighting and appliance states based on the movie night KDE profile. |
| **5** | Suggest Walking Time | Suggests a dog walk during peak routine hours, provided rain intensity is low. |
| **6** | Alt. Walking Time (Rain) | Suggests an alternative walking time shifted by the learned "weather shift" value. |
| **7** | Shopping List Prompt | Prompted when the user is likely to go grocery shopping and stock is low. |
| **8** | Check Fridge & Suggest List | A high-confidence action triggered when low stock counts are $\ge 2$. |
| **9** | No Action Needed | The agent chooses a "wait" state to avoid over-automation or user annoyance. |

---

## Simulation Data Requirements (CSV)
To generate the **User Routine Profile** and fine-tune the RL agent, the simulation must provide a CSV dataset containing logged user activity. The habit-learning pipeline requires the following schema for profile generation:

### Required Columns
* **`timestamp`**: Date and time of the event in `YYYY-MM-DD HH:MM:SS` format.
* **`event_type`**: The activity label. Required labels for the profile include:
    * `WakeUp`, `LeaveHome`, `ReturnHome`
    * `DogWalk`, `GroceryShopping`, `DanceClass`
    * `TVOn`, `TVOff` (for movie habit tracking)
* **`day_of_week`**: The full string name of the day (e.g., `Monday`, `Sunday`).
* **`rain_intensity`**: A float value ($0.0$ to $1.0$). This is critical for calculating `weather_shift_rule` correlations.
* **`heating_temp`**: The numerical temperature set by the user (or `Off`) to determine seasonal thermal preferences.
* **`location_status`**: Categorical indicator of whether the user is `Home` or `Away`.
* **`low_stock_count`**: An integer ($0$ to $4$) used to drive the RL shopping policy.

### Data Example
```csv
timestamp,day_of_week,is_weekend,season,weather_temp,is_raining,rain_intensity,location_status,event_type,heating_temp,tv_status,dog_walk,grocery_trip,dance_class,family_movie_night,confidence_score
2025-01-01 06:54,Wednesday,0,Winter,1.1,False,0.0,Home,WakeUp,21,Off,False,False,False,False,0.91
2025-01-01 08:44,Wednesday,0,Winter,0.7,False,0.0,Away,LeaveHome,21,Off,False,False,False,False,0.94
```

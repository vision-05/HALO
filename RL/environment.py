from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTIONS = [
    "Reduce Heating When Out",
    "Preheat Before Return Home",
    "Preheat House To Preferred Temperature",
    "Recommend Movie Selection",
    "Set Movie Settings (Dim Lights)",
    "Suggest Walking Time",
    "Suggest Alternative Walking Time (Rain)",
    "Shopping List Prompt",
    "Check Fridge And Suggest Shopping List",
    "No Action Needed",
]

DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
SEASONS = ("Winter", "Spring", "Summer", "Autumn")


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip().lower() == "off":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _cyclic_encoding(value: float, period: float) -> Tuple[float, float]:
    angle = 2.0 * math.pi * (value % period) / period
    return math.sin(angle), math.cos(angle)


def _build_default_density_profile(center_hour: float, width_hours: float = 1.5, grid_points: int = 144) -> dict[str, Any]:
    grid = np.linspace(0.0, 24.0, grid_points, endpoint=False)
    wrapped_distance = np.minimum(np.abs(grid - center_hour), 24.0 - np.abs(grid - center_hour))
    density = np.exp(-0.5 * (wrapped_distance / max(0.1, width_hours)) ** 2)
    dx = 24.0 / float(grid_points)
    area = float(np.sum(density) * dx)
    if area > 0.0:
        density = density / area
    return {
        "grid_hours": [round(float(value), 3) for value in grid],
        "density": [round(float(value), 8) for value in density],
        "peak_hour": round(float(center_hour) % 24.0, 2),
        "mean_hour": round(float(center_hour) % 24.0, 2),
        "sample_count": 0,
        "bandwidth_minutes": round(width_hours * 60.0, 2),
    }


def _ensure_density_profile(section: dict[str, Any], fallback_peak: float) -> dict[str, Any]:
    for key in ("kde_profile", "return_home_kde_profile", "movie_night_kde_profile", "dog_walk_evening_kde_profile", "grocery_kde_profile", "dance_class_kde_profile"):
        profile = section.get(key)
        if isinstance(profile, dict) and profile.get("grid_hours") and profile.get("density"):
            return profile
    return _build_default_density_profile(fallback_peak)


def _density_at_hour(profile: dict[str, Any], hour: float) -> float:
    grid = np.asarray(profile.get("grid_hours", []), dtype=float)
    density = np.asarray(profile.get("density", []), dtype=float)
    if grid.size == 0 or density.size == 0:
        return 0.0

    if grid.size == 1:
        return float(density[0])

    wrapped_grid = np.concatenate([grid, np.array([24.0], dtype=float)])
    wrapped_density = np.concatenate([density, np.array([density[0]], dtype=float)])
    return float(np.interp(hour % 24.0, wrapped_grid, wrapped_density))


def _integrate_density_window(profile: dict[str, Any], start_hour: float, window_hours: float, samples: int = 48) -> float:
    if window_hours <= 0.0:
        return 0.0

    end_hour = start_hour + window_hours
    sample_count = max(8, samples)
    sample_points = np.linspace(start_hour, end_hour, sample_count)
    densities = np.array([_density_at_hour(profile, float(point)) for point in sample_points], dtype=float)
    if densities.size < 2:
        return float(densities[0] * window_hours) if densities.size else 0.0

    dx = window_hours / float(sample_count - 1)
    area = float(np.sum((densities[:-1] + densities[1:]) * 0.5) * dx)
    return max(0.0, area)


def _hour_distance(a: float, b: float) -> float:
    diff = abs((a - b + 12.0) % 24.0 - 12.0)
    return float(diff)


class SmartHomeEnv(gym.Env):
    """Gymnasium environment that turns the routine profile into a smart-home RL simulator."""

    metadata = {"render_modes": []}

    def __init__(self, json_path: str | Path, step_minutes: int = 15) -> None:
        super().__init__()
        self.json_path = Path(json_path)
        self.step_minutes = max(5, int(step_minutes))
        self.step_hours = self.step_minutes / 60.0

        with self.json_path.open("r", encoding="utf-8") as file:
            self.profile = json.load(file)

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        self.current_time_hours = 6.0
        self.day_index = 0
        self.day_name = "Monday"
        self.season = "Winter"
        self.day_type = "Weekday"
        self.rain_intensity = 0.0
        self.is_raining = False
        self.low_stock_count = 0
        self.return_hour = 17.0
        self.movie_hour = 20.0
        self.walk_hour = 17.5
        self.shopping_hour = 15.5
        self.leave_hour = 8.0
        self._episode_steps = 0
        self._max_steps = int((24.0 - 6.0) / self.step_hours)

        self._initialize_profile_targets()

    def _initialize_profile_targets(self) -> None:
        weekday = self.profile.get("weekday_occupancy_pattern", {})
        movie = self.profile.get("family_movie_night", {})
        dog_walk = self.profile.get("dog_walking_habits", {})
        shopping = self.profile.get("grocery_shopping", {})

        self.leave_hour = _safe_float(
            weekday.get("usual_leave_hour_mean", weekday.get("usual_leave_hour", 8.0)),
            8.0,
        )
        self.return_hour = _safe_float(
            weekday.get("usual_return_hour_mean", weekday.get("usual_return_hour", 17.0)),
            17.0,
        )
        self.movie_hour = _safe_float(
            movie.get("most_common_start_hour_mean", movie.get("most_common_start_hour", 20.0)),
            20.0,
        )
        self.walk_hour = _safe_float(
            dog_walk.get("common_evening_walk_hour_mean", dog_walk.get("common_evening_walk_hour", 17.5)),
            17.5,
        )
        self.shopping_hour = _safe_float(
            shopping.get("common_shopping_hour_mean", shopping.get("common_shopping_hour", 15.5)),
            15.5,
        )

        self.return_profile = _ensure_density_profile(weekday, self.return_hour)
        self.movie_profile = _ensure_density_profile(movie, self.movie_hour)
        self.walk_profile = _ensure_density_profile(dog_walk, self.walk_hour)
        self.shopping_profile = _ensure_density_profile(shopping, self.shopping_hour)

        weather_rule = dog_walk.get("weather_shift_rule", {})
        grocery_rule = shopping.get("weather_shift_rule", {})
        self.dog_walk_weather_shift_hours = _safe_float(
            weather_rule.get("expected_shift_hours_if_rain_gt_threshold"),
            -7.5,
        )
        self.grocery_weather_shift_hours = _safe_float(
            grocery_rule.get("expected_shift_hours_if_rain_gt_threshold"),
            1.5,
        )

    def _sample_day_context(self) -> None:
        rng = getattr(self, "np_random", np.random.default_rng())
        self.day_index = int(rng.integers(0, len(DAY_NAMES)))
        self.day_name = DAY_NAMES[self.day_index]
        self.day_type = "Weekend" if self.day_name in ("Saturday", "Sunday") else "Weekday"
        self.season = str(rng.choice(list(SEASONS)))
        self.current_time_hours = 6.0
        self.rain_intensity = self._sample_rain_intensity()
        self.is_raining = self.rain_intensity > 0.0
        self.low_stock_count = self._sample_low_stock_count()
        self._episode_steps = 0

    def _sample_rain_intensity(self) -> float:
        rng = getattr(self, "np_random", np.random.default_rng())
        season_rain_prob = {
            "Winter": 0.45,
            "Spring": 0.35,
            "Summer": 0.20,
            "Autumn": 0.40,
        }
        rain_prob = season_rain_prob.get(self.season, 0.30)
        if self.day_type == "Weekend":
            rain_prob = max(0.05, rain_prob - 0.03)
        if rng.random() >= rain_prob:
            return 0.0
        return float(rng.choice([0.25, 0.45, 0.65, 0.85, 1.0]))

    def _sample_low_stock_count(self) -> int:
        rng = getattr(self, "np_random", np.random.default_rng())
        base = [0, 1, 2, 3, 4]
        if self.day_type == "Weekend":
            weights = np.array([0.05, 0.15, 0.30, 0.30, 0.20], dtype=float)
        else:
            weights = np.array([0.10, 0.24, 0.32, 0.22, 0.12], dtype=float)
        if self.rain_intensity > 0.6:
            weights = weights + np.array([0.05, 0.05, 0.05, 0.03, 0.02], dtype=float)
        weights = weights / np.sum(weights)
        return int(rng.choice(base, p=weights))

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        self._sample_day_context()
        return self._get_obs(), {}

    def _season_index(self) -> float:
        season_map = {"Winter": 0.0, "Spring": 1.0, "Summer": 2.0, "Autumn": 3.0}
        return float(season_map.get(self.season, 0.0)) / 3.0

    def _profile_mass(self, profile: dict[str, Any], start_hour: float, window_hours: float) -> float:
        return float(_integrate_density_window(profile, start_hour, window_hours))

    def _get_obs(self):
        time_sin, time_cos = _cyclic_encoding(self.current_time_hours, 24.0)
        day_sin, day_cos = _cyclic_encoding(float(self.day_index), 7.0)
        return_mass = self._profile_mass(self.return_profile, self.current_time_hours, 1.0)
        movie_mass = self._profile_mass(self.movie_profile, self.current_time_hours, 2.0)
        walk_mass = self._profile_mass(self.walk_profile, self.current_time_hours, 2.0)
        shopping_mass = self._profile_mass(self.shopping_profile, self.current_time_hours, 2.0)

        obs = np.array(
            [
                time_sin,
                time_cos,
                day_sin,
                day_cos,
                2.0 * self._season_index() - 1.0,
                2.0 * float(self.rain_intensity) - 1.0,
                2.0 * (float(self.low_stock_count) / 4.0) - 1.0,
                2.0 * float(np.clip(self.current_time_hours / 24.0, 0.0, 1.0)) - 1.0,
                2.0 * float(np.clip(return_mass, 0.0, 1.0)) - 1.0,
                2.0 * float(np.clip(movie_mass, 0.0, 1.0)) - 1.0,
                2.0 * float(np.clip(walk_mass, 0.0, 1.0)) - 1.0,
                2.0 * float(np.clip(shopping_mass, 0.0, 1.0)) - 1.0,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def _reward_reduce_heating_when_out(self) -> float:
        return_mass = self._profile_mass(self.return_profile, self.current_time_hours, 1.0)
        if self.current_time_hours >= self.return_hour:
            return -4.0
        if return_mass < 0.10:
            return 5.0
        if return_mass > 0.60:
            return -3.0
        return 2.0 * (1.0 - return_mass) - 1.0

    def _reward_preheat_before_return_home(self) -> float:
        return_mass = self._profile_mass(self.return_profile, self.current_time_hours, 1.0)
        if return_mass > 0.60:
            return 10.0
        if return_mass < 0.10:
            return -5.0
        return 20.0 * return_mass - 5.0

    def _reward_preheat_house_to_preferred_temperature(self) -> float:
        season_bonus = {"Winter": 8.0, "Spring": 3.0, "Autumn": 4.0, "Summer": -4.0}
        reward = season_bonus.get(self.season, 0.0)
        if self.current_time_hours < self.return_hour:
            reward += 1.5
        return reward

    def _reward_recommend_movie_selection(self) -> float:
        movie_mass = self._profile_mass(self.movie_profile, self.current_time_hours, 2.0)
        if self.day_name != "Friday":
            return -2.0 + 8.0 * movie_mass
        return 10.0 * movie_mass - 1.5

    def _reward_set_movie_settings(self) -> float:
        movie_mass = self._profile_mass(self.movie_profile, self.current_time_hours, 1.0)
        if movie_mass > 0.50:
            return 12.0 * movie_mass
        return 8.0 * movie_mass - 2.0

    def _reward_suggest_walking_time(self) -> float:
        walk_mass = self._profile_mass(self.walk_profile, self.current_time_hours, 2.0)
        if self.rain_intensity > 0.6:
            return -4.0
        return 8.0 * walk_mass - 1.0

    def _reward_suggest_alternative_walking_time(self) -> float:
        if self.rain_intensity <= 0.6:
            return -4.0
        rainy_target = (self.walk_hour + self.dog_walk_weather_shift_hours) % 24.0
        distance = _hour_distance(self.current_time_hours, rainy_target)
        if distance <= 0.5:
            return 20.0
        if distance <= 1.5:
            return 12.0 - 5.0 * distance
        return -2.0

    def _reward_shopping_list_prompt(self) -> float:
        shopping_mass = self._profile_mass(self.shopping_profile, self.current_time_hours, 2.0)
        if self.low_stock_count == 0:
            return -3.0
        return 6.0 * shopping_mass + float(self.low_stock_count)

    def _reward_check_fridge_and_suggest_shopping_list(self) -> float:
        shopping_mass = self._profile_mass(self.shopping_profile, self.current_time_hours, 2.0)
        if self.low_stock_count < 2:
            return -4.0
        return 8.0 * shopping_mass + 2.5 * float(self.low_stock_count)

    def _reward_no_action_needed(self) -> float:
        signals = [
            self._profile_mass(self.return_profile, self.current_time_hours, 1.0),
            self._profile_mass(self.movie_profile, self.current_time_hours, 1.0),
            self._profile_mass(self.walk_profile, self.current_time_hours, 1.0),
            self._profile_mass(self.shopping_profile, self.current_time_hours, 1.0),
        ]
        strongest_signal = max(signals)
        if strongest_signal < 0.10:
            return 2.5
        return -4.0 * strongest_signal

    def _calculate_reward(self, action: int) -> float:
        if action < 0 or action >= len(ACTIONS):
            raise IndexError(f"Invalid action index: {action}")

        reward_functions = {
            0: self._reward_reduce_heating_when_out,
            1: self._reward_preheat_before_return_home,
            2: self._reward_preheat_house_to_preferred_temperature,
            3: self._reward_recommend_movie_selection,
            4: self._reward_set_movie_settings,
            5: self._reward_suggest_walking_time,
            6: self._reward_suggest_alternative_walking_time,
            7: self._reward_shopping_list_prompt,
            8: self._reward_check_fridge_and_suggest_shopping_list,
            9: self._reward_no_action_needed,
        }
        return float(reward_functions[action]())

    def step(self, action: int):
        action_name = self.action_name(action)
        reward = self._calculate_reward(action)
        info = {
            "action": action,
            "action_name": action_name,
            "season": self.season,
            "day_name": self.day_name,
            "day_type": self.day_type,
            "rain_intensity": self.rain_intensity,
            "low_stock_count": self.low_stock_count,
            "current_time_hours": self.current_time_hours,
            "reward": reward,
        }

        self.current_time_hours += self.step_hours
        self._episode_steps += 1

        terminated = self.current_time_hours >= 24.0
        truncated = self._episode_steps >= self._max_steps
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def action_name(self, action: int) -> str:
        return ACTIONS[int(action)]

    def build_observation_from_state(
        self,
        current_time_hours: float,
        day_name: str,
        season: str,
        rain_intensity: float,
        low_stock_count: int,
    ) -> np.ndarray:
        """Build an observation for inference outside the simulator."""
        self.current_time_hours = float(current_time_hours)
        self.day_name = day_name if day_name in DAY_NAMES else "Monday"
        self.day_index = DAY_NAMES.index(self.day_name)
        self.day_type = "Weekend" if self.day_name in ("Saturday", "Sunday") else "Weekday"
        self.season = season if season in SEASONS else "Winter"
        self.rain_intensity = float(np.clip(rain_intensity, 0.0, 1.0))
        self.is_raining = self.rain_intensity > 0.0
        self.low_stock_count = int(np.clip(low_stock_count, 0, 4))
        return self._get_obs()


def load_profile(json_path: str | Path) -> dict[str, Any]:
    with Path(json_path).open("r", encoding="utf-8") as file:
        return json.load(file)


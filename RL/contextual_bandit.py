from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Action space used by Thompson sampling.
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
]

N_ITERATIONS = 10000
MIN_ACTION_PULLS = 30
EPSILON = 0.05


def load_user_profile() -> dict[str, Any]:
    """Load deterministic routine profile used to generate context."""
    base_dir = Path(__file__).resolve().parents[1]
    profile_path = base_dir / "routine_learning" / "user_routine_profile.json"
    with profile_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def mock_weather_forecast(py_rng: random.Random, season: str, day_type: str) -> bool:
    """Simulate whether it rains during the usual evening walk."""
    rain_prob_by_season = {
        "Winter": 0.45,
        "Spring": 0.35,
        "Summer": 0.20,
        "Autumn": 0.40,
    }
    rain_prob = rain_prob_by_season.get(season, 0.30)
    if day_type == "Weekend":
        rain_prob = max(0.05, rain_prob - 0.03)
    return py_rng.random() < rain_prob


def mock_fridge_inventory(py_rng: random.Random) -> tuple[list[str], list[str]]:
    """Simulate fridge contents and low-stock items with varying depletion."""
    all_items = ["milk", "eggs", "bread", "vegetables", "fruit", "chicken", "rice", "yogurt"]

    low_stock_count = py_rng.choices([0, 1, 2, 3, 4], weights=[0.08, 0.20, 0.34, 0.24, 0.14], k=1)[0]
    low_stock_items = py_rng.sample(all_items, k=low_stock_count)
    fridge_items = [item for item in all_items if item not in low_stock_items]

    if len(fridge_items) > 5:
        fridge_items = py_rng.sample(fridge_items, k=5)

    return fridge_items, low_stock_items


def get_context(
    user_profile: dict[str, Any],
    py_rng: random.Random,
    season: str = "Winter",
    current_hour: float = 17.5,
    day_type: str = "Weekday",
) -> dict[str, Any]:
    """Build context from profile + environment simulators."""
    leave_hour = user_profile["weekday_occupancy_pattern"]["usual_leave_hour"]
    return_hour = user_profile["weekday_occupancy_pattern"]["usual_return_hour"]
    preferred_temp = float(user_profile["heating_preferences"].get(season, 21))
    movie_hour = user_profile["family_movie_night"]["most_common_start_hour"]
    walk_hour = user_profile["dog_walking_habits"]["common_evening_walk_hour"]
    shopping_hour = user_profile["grocery_shopping"]["common_shopping_hour"]

    is_raining = mock_weather_forecast(py_rng, season=season, day_type=day_type)
    fridge_items, low_stock_items = mock_fridge_inventory(py_rng)

    return {
        "season": season,
        "day_type": day_type,
        "current_hour": current_hour,
        "leave_hour": leave_hour,
        "return_hour": return_hour,
        "preferred_temp": preferred_temp,
        "movie_hour": movie_hour,
        "walk_hour": walk_hour,
        "shopping_hour": shopping_hour,
        "is_raining_during_walk": is_raining,
        "fridge_items": fridge_items,
        "low_stock_items": low_stock_items,
    }


def get_context_key(context: dict[str, Any]) -> tuple[Any, ...]:
    """Discretize context into buckets for disjoint contextual Thompson sampling."""
    current_hour = float(context["current_hour"])
    return_hour = float(context["return_hour"])
    movie_hour = float(context["movie_hour"])
    walk_hour = float(context["walk_hour"])
    low_stock_count = len(context["low_stock_items"])

    hour_bucket = int(current_hour // 3)  # 6-8, 9-11, ...
    return_phase = "before_return" if current_hour < return_hour else "after_return"
    near_movie = abs(current_hour - movie_hour) <= 1.5
    near_walk = abs(current_hour - walk_hour) <= 1.5
    stock_bucket = min(low_stock_count, 3)

    return (
        context["season"],
        context["day_type"],
        hour_bucket,
        return_phase,
        near_movie,
        near_walk,
        context["is_raining_during_walk"],
        stock_bucket,
    )


def get_valid_action_indices(context: dict[str, Any]) -> list[int]:
    """Filter actions to context-appropriate candidates."""
    current_hour = float(context["current_hour"])
    return_hour = float(context["return_hour"])
    movie_hour = float(context["movie_hour"])
    walk_hour = float(context["walk_hour"])
    raining = bool(context["is_raining_during_walk"])
    low_stock_count = len(context["low_stock_items"])

    valid_indices: list[int] = []
    for index, action in enumerate(ACTIONS):
        if action == "Reduce Heating When Out" and current_hour >= return_hour:
            continue
        if action == "Preheat Before Return Home" and abs(current_hour - (return_hour - 1)) > 2:
            continue
        if action in {"Recommend Movie Selection", "Set Movie Settings (Dim Lights)"} and abs(current_hour - movie_hour) > 2:
            continue
        if action == "Suggest Walking Time" and (raining or abs(current_hour - walk_hour) > 2):
            continue
        if action == "Suggest Alternative Walking Time (Rain)" and not raining:
            continue
        if action == "Shopping List Prompt" and low_stock_count == 0:
            continue
        if action == "Check Fridge And Suggest Shopping List" and low_stock_count < 2:
            continue
        valid_indices.append(index)

    # Keep a safe fallback when strict gates remove every action.
    return valid_indices or list(range(len(ACTIONS)))


def _bernoulli(np_rng: np.random.Generator, prob: float) -> int:
    """Sample a Bernoulli reward as int {0,1}."""
    return int(np_rng.binomial(1, prob))


def expected_reward_probability(action: str, context: dict[str, Any]) -> float:
    """Return expected success probability for action in a given context."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")

    required_keys = {
        "season",
        "current_hour",
        "return_hour",
        "walk_hour",
        "movie_hour",
        "is_raining_during_walk",
        "low_stock_items",
    }
    missing = required_keys - context.keys()
    if missing:
        raise KeyError(f"Context missing keys: {sorted(missing)}")

    current_hour = context["current_hour"]
    return_hour = context["return_hour"]
    walk_hour = context["walk_hour"]
    movie_hour = context["movie_hour"]
    raining = context["is_raining_during_walk"]
    low_stock_count = len(context["low_stock_items"])

    if action == "Reduce Heating When Out":
        return 0.85 if current_hour < return_hour else 0.40

    if action == "Preheat Before Return Home":
        return 0.90 if abs(current_hour - (return_hour - 1)) < 1 else 0.30

    if action == "Preheat House To Preferred Temperature":
        return 0.88 if context["season"] == "Winter" else 0.55

    if action == "Recommend Movie Selection":
        return 0.80 if abs(current_hour - movie_hour) < 2 else 0.35

    if action == "Set Movie Settings (Dim Lights)":
        return 0.92 if abs(current_hour - movie_hour) < 1 else 0.20

    if action == "Suggest Walking Time":
        return 0.82 if (not raining and abs(current_hour - walk_hour) < 2) else 0.30

    if action == "Suggest Alternative Walking Time (Rain)":
        return 0.90 if raining else 0.15

    if action == "Shopping List Prompt":
        return 0.84 if low_stock_count > 0 else 0.25

    if action == "Check Fridge And Suggest Shopping List":
        return 0.93 if low_stock_count >= 2 else 0.35

    return 0.50


def simulate_reward(action: str, context: dict[str, Any], np_rng: np.random.Generator) -> int:
    """Sample realized reward from expected action success probability."""
    return _bernoulli(np_rng, expected_reward_probability(action, context))


class ContextualThompsonSamplingBandit:
    """Disjoint contextual Bernoulli Thompson sampling with fairness constraints."""

    def __init__(self, action_count: int) -> None:
        self.action_count = action_count
        self.alpha_by_context: dict[tuple[Any, ...], np.ndarray] = {}
        self.beta_by_context: dict[tuple[Any, ...], np.ndarray] = {}
        self.total_pulls = np.zeros(action_count, dtype=int)
        self.total_successes = np.zeros(action_count, dtype=int)

    def _ensure_context(self, context_key: tuple[Any, ...]) -> None:
        if context_key not in self.alpha_by_context:
            self.alpha_by_context[context_key] = np.ones(self.action_count, dtype=float)
            self.beta_by_context[context_key] = np.ones(self.action_count, dtype=float)

    def select_action(
        self,
        context_key: tuple[Any, ...],
        valid_indices: list[int],
        np_rng: np.random.Generator,
        py_rng: random.Random,
        min_action_pulls: int = MIN_ACTION_PULLS,
        epsilon: float = EPSILON,
    ) -> tuple[int, np.ndarray]:
        self._ensure_context(context_key)
        alpha = self.alpha_by_context[context_key]
        beta = self.beta_by_context[context_key]

        underexplored = [idx for idx in valid_indices if self.total_pulls[idx] < min_action_pulls]
        if underexplored:
            return py_rng.choice(underexplored), np.array([])

        if py_rng.random() < epsilon:
            return py_rng.choice(valid_indices), np.array([])

        sampled_theta = np_rng.beta(alpha, beta)
        valid_samples = sampled_theta[valid_indices]
        best_local = int(np.argmax(valid_samples))
        return valid_indices[best_local], sampled_theta

    def update(self, context_key: tuple[Any, ...], action_index: int, reward: int) -> None:
        if not 0 <= action_index < self.action_count:
            raise IndexError(f"Invalid action index: {action_index}")
        if reward not in (0, 1):
            raise ValueError(f"Reward must be 0 or 1, got: {reward}")

        self._ensure_context(context_key)
        alpha = self.alpha_by_context[context_key]
        beta = self.beta_by_context[context_key]

        self.total_pulls[action_index] += 1
        self.total_successes[action_index] += reward

        if reward == 1:
            alpha[action_index] += 1
        else:
            beta[action_index] += 1

    def estimated_scores(self) -> np.ndarray:
        return (self.total_successes + 1.0) / (self.total_pulls + 2.0)


def train(
    n_iterations: int = N_ITERATIONS,
    seed: Optional[int] = None,
    season: str = "Winter",
) -> tuple[ContextualThompsonSamplingBandit, np.ndarray, dict[str, Any]]:
    """Run Thompson sampling training and return final model + scores."""
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    user_profile = load_user_profile()
    bandit = ContextualThompsonSamplingBandit(action_count=len(ACTIONS))

    availability_count = np.zeros(len(ACTIONS), dtype=int)
    available_success_sum = np.zeros(len(ACTIONS), dtype=float)
    chosen_when_available = np.zeros(len(ACTIONS), dtype=int)
    cumulative_regret = 0.0

    print("\nStarting Expanded Thompson Sampling Training...\n")

    for step in range(n_iterations):
        context = get_context(
            user_profile=user_profile,
            py_rng=py_rng,
            season=season,
            current_hour=py_rng.uniform(6, 22),
            day_type=py_rng.choice(["Weekday", "Weekend"]),
        )

        valid_indices = get_valid_action_indices(context)
        context_key = get_context_key(context)

        for idx in valid_indices:
            availability_count[idx] += 1
            available_success_sum[idx] += expected_reward_probability(ACTIONS[idx], context)

        action_index, sampled_theta = bandit.select_action(
            context_key=context_key,
            valid_indices=valid_indices,
            np_rng=np_rng,
            py_rng=py_rng,
        )
        chosen_when_available[action_index] += 1

        chosen_action = ACTIONS[action_index]
        reward = simulate_reward(chosen_action, context, np_rng=np_rng)
        bandit.update(context_key=context_key, action_index=action_index, reward=reward)

        chosen_prob = expected_reward_probability(chosen_action, context)
        oracle_prob = max(expected_reward_probability(ACTIONS[idx], context) for idx in valid_indices)
        cumulative_regret += oracle_prob - chosen_prob

        if step % 100 == 0:
            print(f"\nStep {step}")
            print(f"Current Hour: {context['current_hour']:.2f}")
            print(f"Chosen Action: {chosen_action}")
            print(f"Reward: {reward}")
            if sampled_theta.size:
                print(f"Sampled Theta (chosen): {sampled_theta[action_index]:.3f}")
            print(f"Raining During Walk: {context['is_raining_during_walk']}")
            print(f"Low Stock Items: {context['low_stock_items']}")
            print("-" * 50)

    scores = bandit.estimated_scores()
    metrics = {
        "availability_count": availability_count,
        "available_success_sum": available_success_sum,
        "chosen_when_available": chosen_when_available,
        "avg_regret": cumulative_regret / max(1, n_iterations),
    }
    return bandit, scores, metrics


def print_results(bandit: ContextualThompsonSamplingBandit, scores: np.ndarray, metrics: dict[str, Any]) -> None:
    """Pretty-print final posterior state and best action."""
    print("\nFinal Learned Action Preferences:\n")

    availability_count = metrics["availability_count"]
    available_success_sum = metrics["available_success_sum"]
    chosen_when_available = metrics["chosen_when_available"]

    for index, action in enumerate(ACTIONS):
        pulls = bandit.total_pulls[index]
        empirical_rate = (bandit.total_successes[index] / pulls) if pulls else 0.0
        conditioned_expected = (
            available_success_sum[index] / availability_count[index] if availability_count[index] else 0.0
        )

        print(f"Action: {action}")
        print(f"Estimated Success Rate (Bayesian): {scores[index]:.3f}")
        print(f"Empirical Success Rate (Chosen): {empirical_rate:.3f}")
        print(f"Expected Success Rate (When Valid): {conditioned_expected:.3f}")
        print(f"Chosen / Valid Opportunities: {chosen_when_available[index]} / {availability_count[index]}")
        print(f"Global Successes / Pulls: {bandit.total_successes[index]} / {pulls}")
        print("-" * 40)

    best_action_index = int(np.argmax(scores))
    print("\nBEST LEARNED ACTION:")
    print(ACTIONS[best_action_index])
    print(f"Score: {scores[best_action_index]:.3f}")
    print(f"Average Instant Regret: {metrics['avg_regret']:.4f}")


def main() -> None:
    bandit, scores, metrics = train()
    print_results(bandit, scores, metrics)


if __name__ == "__main__":
    main()

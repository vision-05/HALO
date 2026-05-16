from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from .environment import ACTIONS, SmartHomeEnv
except ImportError:  # pragma: no cover - supports direct execution from RL/
    from environment import ACTIONS, SmartHomeEnv

DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "routine_learning" / "user_routine_profile.json"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "smart_home_ppo_agent"


def run_random_baseline(
    env: SmartHomeEnv,
    n_episodes: int = 100,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Run a random action baseline for comparison."""
    np.random.seed(seed)
    episode_rewards = []
    action_counts = np.zeros(len(ACTIONS), dtype=int)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = int(np.random.randint(0, len(ACTIONS)))
            action_counts[action] += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated

        episode_rewards.append(episode_reward)

    return {
        "name": "Random Baseline",
        "episodes": n_episodes,
        "episode_rewards": episode_rewards,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "action_counts": action_counts.tolist(),
        "action_distribution": (action_counts / float(np.sum(action_counts))).tolist(),
    }


def run_ppo_evaluation(
    env: SmartHomeEnv,
    model,
    n_episodes: int = 100,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Run the trained PPO model for evaluation."""
    if seed is not None:
        np.random.seed(seed)

    episode_rewards = []
    action_counts = np.zeros(len(ACTIONS), dtype=int)
    action_rewards = [[] for _ in range(len(ACTIONS))]

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_counts[action] += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            action_rewards[action].append(float(reward))
            done = terminated or truncated

        episode_rewards.append(episode_reward)

    per_action_mean_rewards = [
        float(np.mean(rewards)) if rewards else 0.0 for rewards in action_rewards
    ]

    return {
        "name": "PPO Agent",
        "episodes": n_episodes,
        "episode_rewards": episode_rewards,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "action_counts": action_counts.tolist(),
        "action_distribution": (action_counts / float(np.sum(action_counts))).tolist(),
        "per_action_mean_rewards": per_action_mean_rewards,
    }


def print_evaluation_report(
    ppo_results: dict[str, Any],
    random_results: dict[str, Any],
) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print("PPO SMART HOME AGENT EVALUATION REPORT")
    print("=" * 70)

    print(f"\n{'Metric':<40} {'PPO Agent':<15} {'Random Baseline':<15}")
    print("-" * 70)
    print(f"{'Episodes':<40} {ppo_results['episodes']:<15} {random_results['episodes']:<15}")
    print(f"{'Mean Episode Reward':<40} {ppo_results['mean_reward']:<15.3f} {random_results['mean_reward']:<15.3f}")
    print(f"{'Std Dev':<40} {ppo_results['std_reward']:<15.3f} {random_results['std_reward']:<15.3f}")
    print(f"{'Min Reward':<40} {ppo_results['min_reward']:<15.3f} {random_results['min_reward']:<15.3f}")
    print(f"{'Max Reward':<40} {ppo_results['max_reward']:<15.3f} {random_results['max_reward']:<15.3f}")

    improvement = ppo_results['mean_reward'] - random_results['mean_reward']
    improvement_pct = (improvement / abs(random_results['mean_reward'])) * 100 if random_results['mean_reward'] != 0 else 0
    print(f"\n{'Improvement over Random':<40} {improvement:<15.3f} ({improvement_pct:.1f}%)")

    print("\n" + "-" * 70)
    print("ACTION DISTRIBUTION")
    print("-" * 70)
    print(f"{'Action':<45} {'PPO Count':<12} {'PPO %':<12}")
    print("-" * 70)

    for idx, action in enumerate(ACTIONS):
        ppo_count = ppo_results["action_counts"][idx]
        ppo_pct = ppo_results["action_distribution"][idx] * 100
        print(f"{action:<45} {ppo_count:<12} {ppo_pct:<12.1f}%")

    print("\n" + "-" * 70)
    print("PER-ACTION PERFORMANCE (PPO)")
    print("-" * 70)
    print(f"{'Action':<45} {'Mean Reward':<15}")
    print("-" * 70)

    for idx, action in enumerate(ACTIONS):
        mean_reward = ppo_results["per_action_mean_rewards"][idx]
        print(f"{action:<45} {mean_reward:<15.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"PPO Agent Mean Reward: {ppo_results['mean_reward']:.3f} ± {ppo_results['std_reward']:.3f}")
    print(f"Random Baseline Mean Reward: {random_results['mean_reward']:.3f} ± {random_results['std_reward']:.3f}")
    print(f"PPO is {improvement_pct:.1f}% better than random baseline")
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained PPO smart-home agent.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH, help="Path to the JSON profile.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained PPO model.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default=None, help="Optional: save results to JSON file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from PPO import load_trained_model, _require_sb3

        _require_sb3()
    except ImportError as exc:
        print(f"Error: {exc}")
        print("Install with: pip install gymnasium stable-baselines3")
        return 1

    env = SmartHomeEnv(args.profile)
    model = load_trained_model(args.model)

    print("Evaluating PPO agent...")
    ppo_results = run_ppo_evaluation(env, model, n_episodes=args.episodes, seed=args.seed)

    print("Evaluating random baseline...")
    random_results = run_random_baseline(env, n_episodes=args.episodes, seed=args.seed)

    print_evaluation_report(ppo_results, random_results)

    if args.output:
        import json

        combined_results = {
            "ppo": ppo_results,
            "random_baseline": random_results,
            "improvement": ppo_results["mean_reward"] - random_results["mean_reward"],
        }
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=2)
        print(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

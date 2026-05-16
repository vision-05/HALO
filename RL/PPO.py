from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Tuple, Union

try:
    from .environment import ACTIONS, SmartHomeEnv
except ImportError:  # pragma: no cover - allows direct script execution from RL/
    from environment import ACTIONS, SmartHomeEnv

DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "routine_learning" / "user_routine_profile.json"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "smart_home_ppo_agent"

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError as exc:  # pragma: no cover - dependency error is runtime-specific
    PPO: Any = None
    check_env: Any = None
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None


def _require_sb3() -> None:
    if PPO is None or check_env is None:
        raise ImportError(
            "stable-baselines3 is required for PPO training. Install gymnasium and stable-baselines3 first."
        ) from _SB3_IMPORT_ERROR


def make_env(profile_path: Union[Path, str] = DEFAULT_PROFILE_PATH) -> SmartHomeEnv:
    return SmartHomeEnv(profile_path)


def train_ppo(
    profile_path: Union[Path, str] = DEFAULT_PROFILE_PATH,
    total_timesteps: int = 100_000,
    model_path: Union[Path, str] = DEFAULT_MODEL_PATH,
    seed: int = 42,
    run_check: bool = True,
):
    """Train a PPO agent on the smart-home environment and save the model."""
    _require_sb3()
    env = make_env(profile_path)
    if run_check:
        check_env_fn: Any = check_env
        check_env_fn(env, warn=True)

    ppo_cls: Any = PPO
    model = ppo_cls("MlpPolicy", env, verbose=1, seed=seed)
    print("Training PPO agent...")
    model.learn(total_timesteps=int(total_timesteps))
    model.save(str(model_path))
    print(f"Saved PPO model to: {model_path}")
    return model, env


def fine_tune_ppo(
    profile_path: Union[Path, str] = DEFAULT_PROFILE_PATH,
    model_path: Union[Path, str] = DEFAULT_MODEL_PATH,
    online_timesteps: int = 2_000,
    learning_rate: Optional[float] = None,
):
    """Load an existing model, attach updated env and continue training (incremental fine-tune).

    - online_timesteps: small number (1k-5k) suitable for edge micro-updates
    - learning_rate: optional lower LR to mitigate catastrophic forgetting
    """
    _require_sb3()

    # 1. Build new environment from updated profile
    env = make_env(profile_path)

    # 2. Load existing model
    print(f"Loading existing PPO model from: {model_path}")
    model = load_trained_model(model_path)

    # 3. Optionally lower learning rate on optimizer param groups
    if learning_rate is not None:
        try:
            model.learning_rate = float(learning_rate)  # keep for bookkeeping
            opt = getattr(model.policy, "optimizer", None)
            if opt is not None:
                for g in opt.param_groups:
                    g["lr"] = float(learning_rate)
        except Exception:
            # best-effort: if optimizer not accessible, continue without failing
            pass

    # 4. Attach the new environment
    model.set_env(env)

    # 5. Fine-tune (do not reset num timesteps so logging remains continuous)
    print(f"Fine-tuning agent for {online_timesteps} timesteps (lr={learning_rate})...")
    model.learn(total_timesteps=int(online_timesteps), reset_num_timesteps=False)

    # 6. Save updated model
    model.save(str(model_path))
    print("Fine-tuning complete and model updated.")
    return model, env


def load_trained_model(model_path: Union[Path, str]):
    """Load a previously trained PPO model."""
    _require_sb3()
    ppo_cls: Any = PPO
    return ppo_cls.load(str(model_path))


def load_or_train_model(
    profile_path: Union[Path, str] = DEFAULT_PROFILE_PATH,
    model_path: Union[Path, str] = DEFAULT_MODEL_PATH,
    total_timesteps: int = 25_000,
    seed: int = 42,
    run_check: bool = False,
):
    """Load a saved PPO model or bootstrap a new one from the current profile."""
    try:
        return load_trained_model(model_path)
    except Exception:
        model, _ = train_ppo(
            profile_path=profile_path,
            total_timesteps=total_timesteps,
            model_path=model_path,
            seed=seed,
            run_check=run_check,
        )
        return model


def predict_next_action(
    model,
    env: SmartHomeEnv,
    state: Optional[Tuple[float, str, str, float, int]] = None,
    deterministic: bool = True,
):
    """Predict the next action from the current env state or a supplied state tuple."""
    if state is not None:
        current_time_hours, day_name, season, rain_intensity, low_stock_count = state
        observation = env.build_observation_from_state(
            current_time_hours=current_time_hours,
            day_name=day_name,
            season=season,
            rain_intensity=rain_intensity,
            low_stock_count=low_stock_count,
        )
    else:
        observation, _ = env.reset()

    action_index, _ = model.predict(observation, deterministic=deterministic)
    action_index = int(action_index)
    return action_index, ACTIONS[action_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run a PPO smart-home agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the PPO agent.")
    train_parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--timesteps", type=int, default=100_000)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--skip-env-check", action="store_true")

    fine_tune_parser = subparsers.add_parser("fine_tune", help="Fine-tune an existing PPO model with updated profile.")
    fine_tune_parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    fine_tune_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    fine_tune_parser.add_argument("--timesteps", type=int, default=2000)
    fine_tune_parser.add_argument("--lr", type=float, default=None, help="Optional lower learning rate for fine-tuning")

    predict_parser = subparsers.add_parser("predict", help="Predict the next action using a trained PPO model.")
    predict_parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict_parser.add_argument("--current-hour", type=float, default=17.5)
    predict_parser.add_argument("--day-name", type=str, default="Friday")
    predict_parser.add_argument("--season", type=str, default="Winter")
    predict_parser.add_argument("--rain-intensity", type=float, default=0.0)
    predict_parser.add_argument("--low-stock-count", type=int, default=1)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "train":
        train_ppo(
            profile_path=args.profile,
            total_timesteps=args.timesteps,
            model_path=args.model,
            seed=args.seed,
            run_check=not args.skip_env_check,
        )
        print("Training complete.")
        return 0

    if args.command == "predict":
        env = make_env(args.profile)
        model = load_trained_model(args.model)
        state = (
            float(args.current_hour),
            str(args.day_name),
            str(args.season),
            float(args.rain_intensity),
            int(args.low_stock_count),
        )
        action_index, action_name = predict_next_action(model, env, state=state)
        print(f"Predicted action index: {action_index}")
        print(f"Predicted action: {action_name}")
        return 0

    if args.command == "fine_tune":
        fine_tune_ppo(
            profile_path=args.profile,
            model_path=args.model,
            online_timesteps=args.timesteps,
            learning_rate=args.lr,
        )
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

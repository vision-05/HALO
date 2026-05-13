from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PPO import DEFAULT_MODEL_PATH, fine_tune_ppo
from routine_learning.smart_data_generation import (
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_JSON,
    build_user_routine_profile,
    load_dataset,
    save_profile,
    truncate_csv_to_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rolling-window profile refresh + incremental PPO fine-tune."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--profile", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--truncate-input", action="store_true")
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.00005)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.truncate_input:
        df = truncate_csv_to_window(args.input, window_days=args.window_days)
    else:
        df = load_dataset(args.input, window_days=args.window_days)

    profile = build_user_routine_profile(df)
    profile["profile_metadata"]["rolling_window_days"] = int(args.window_days)
    save_profile(profile, args.profile)

    print("Profile refreshed from rolling window.")
    print(f"Rows retained: {len(df)}")
    print(f"Unique days retained: {df['date_only'].nunique()}")
    print(f"Profile path: {args.profile}")

    fine_tune_ppo(
        profile_path=args.profile,
        model_path=args.model,
        online_timesteps=args.timesteps,
        learning_rate=args.lr,
    )

    print("Online update pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



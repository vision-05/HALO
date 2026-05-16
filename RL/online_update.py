from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

try:
    from .PPO import DEFAULT_MODEL_PATH, DEFAULT_PROFILE_PATH, fine_tune_ppo, load_or_train_model
    from .environment import _build_default_density_profile
except ImportError:  # pragma: no cover - supports direct execution from RL/
    from PPO import DEFAULT_MODEL_PATH, DEFAULT_PROFILE_PATH, fine_tune_ppo, load_or_train_model
    from environment import _build_default_density_profile

DEFAULT_ACTIVITY_LOG_PATH = Path(__file__).resolve().parents[1] / "db" / "rl_activity_log.jsonl"

PROFILE_ALIASES: dict[tuple[str, str], tuple[str, ...]] = {
    ("weekday_occupancy_pattern", "usual_leave_hour_mean"): (
        "leave_hour",
        "weekday_leave_hour",
        "usual_leave_hour",
        "leave_hour_mean",
    ),
    ("weekday_occupancy_pattern", "usual_return_hour_mean"): (
        "return_hour",
        "weekday_return_hour",
        "usual_return_hour",
        "return_hour_mean",
    ),
    ("family_movie_night", "most_common_start_hour_mean"): (
        "movie_hour",
        "movie_start_hour",
        "most_common_start_hour",
        "movie_night_hour",
    ),
    ("dog_walking_habits", "common_evening_walk_hour_mean"): (
        "walk_hour",
        "dog_walk_hour",
        "common_evening_walk_hour",
    ),
    ("grocery_shopping", "common_shopping_hour_mean"): (
        "shopping_hour",
        "grocery_hour",
        "common_shopping_hour",
    ),
    ("dance_class", "common_start_hour_mean"): (
        "dance_hour",
        "dance_class_hour",
        "common_start_hour",
    ),
}

WEATHER_SHIFT_ALIASES: dict[tuple[str, str], tuple[str, ...]] = {
    ("dog_walking_habits", "weather_shift_rule.expected_shift_hours_if_rain_gt_threshold"): (
        "dog_walk_weather_shift_hours",
        "dog_walk_shift_hours",
        "expected_dog_walk_shift_hours",
    ),
    ("grocery_shopping", "weather_shift_rule.expected_shift_hours_if_rain_gt_threshold"): (
        "grocery_weather_shift_hours",
        "grocery_shift_hours",
        "expected_grocery_shift_hours",
    ),
}


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "off"}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_profile(profile_path: Path) -> dict[str, Any]:
    if not profile_path.exists():
        return {}
    loaded = _load_json_file(profile_path)
    return loaded if isinstance(loaded, dict) else {}


def save_profile(profile: dict[str, Any], profile_path: Path) -> None:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2, sort_keys=True)


def load_activity_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []

    if log_path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(log_path)
        return df.to_dict(orient="records")

    if log_path.suffix.lower() == ".json":
        loaded = _load_json_file(log_path)
        if isinstance(loaded, list):
            return [entry for entry in loaded if isinstance(entry, dict)]
        if isinstance(loaded, dict):
            return [loaded]
        return []

    records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                records.append(entry)
    return records


def _flatten_entry(entry: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    flattened.update(entry)
    for nested_key in ("state_snapshot", "profile_hints", "context", "metrics"):
        nested = entry.get(nested_key)
        if isinstance(nested, dict):
            for key, value in nested.items():
                flattened.setdefault(key, value)
    return flattened


def _extract_numeric_values(entries: Iterable[dict[str, Any]], keys: Iterable[str]) -> list[float]:
    values: list[float] = []
    for entry in entries:
        flattened = _flatten_entry(entry)
        for key in keys:
            if key not in flattened:
                continue
            try:
                values.append(float(flattened[key]))
            except (TypeError, ValueError):
                continue
    return values


def _mean_or_default(values: list[float], default: float) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def _update_density_profile(section: dict[str, Any], center_hour: float, sample_count: int) -> None:
    density_profile = _build_default_density_profile(center_hour)
    density_profile["sample_count"] = int(sample_count)
    section["kde_profile"] = density_profile


def build_profile_from_monthly_log(
    entries: Iterable[dict[str, Any]],
    base_profile: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a monthly profile from rolling activity snapshots.

    The updater is intentionally permissive: if the log contains explicit hour hints
    (leave_hour, return_hour, movie_hour, etc.) they are averaged into the profile.
    When the log is sparse, the previous profile is preserved.
    """
    profile = deepcopy(base_profile) if base_profile else {}
    normalized_entries = [entry for entry in entries if isinstance(entry, dict)]

    def update_section(section_name: str, field_name: str, aliases: tuple[str, ...], fallback: float) -> float:
        section = profile.setdefault(section_name, {})
        values = _extract_numeric_values(normalized_entries, aliases)
        existing = _coerce_float(section.get(field_name), fallback)
        center = _mean_or_default(values, existing)
        section[field_name] = round(center, 2)
        _update_density_profile(section, center, len(values))
        return center

    leave_hour = update_section(
        "weekday_occupancy_pattern",
        "usual_leave_hour_mean",
        PROFILE_ALIASES[("weekday_occupancy_pattern", "usual_leave_hour_mean")],
        8.0,
    )
    return_hour = update_section(
        "weekday_occupancy_pattern",
        "usual_return_hour_mean",
        PROFILE_ALIASES[("weekday_occupancy_pattern", "usual_return_hour_mean")],
        17.0,
    )
    movie_hour = update_section(
        "family_movie_night",
        "most_common_start_hour_mean",
        PROFILE_ALIASES[("family_movie_night", "most_common_start_hour_mean")],
        20.0,
    )
    walk_hour = update_section(
        "dog_walking_habits",
        "common_evening_walk_hour_mean",
        PROFILE_ALIASES[("dog_walking_habits", "common_evening_walk_hour_mean")],
        17.5,
    )
    shopping_hour = update_section(
        "grocery_shopping",
        "common_shopping_hour_mean",
        PROFILE_ALIASES[("grocery_shopping", "common_shopping_hour_mean")],
        15.5,
    )
    dance_hour = update_section(
        "dance_class",
        "common_start_hour_mean",
        PROFILE_ALIASES[("dance_class", "common_start_hour_mean")],
        8.0,
    )

    for (section_name, field_path), aliases in WEATHER_SHIFT_ALIASES.items():
        section = profile.setdefault(section_name, {})
        shift_values = _extract_numeric_values(normalized_entries, aliases)
        if not shift_values:
            continue
        weather_rule = section.setdefault("weather_shift_rule", {})
        weather_rule["expected_shift_hours_if_rain_gt_threshold"] = round(float(sum(shift_values) / len(shift_values)), 2)

    movie_section = profile.setdefault("family_movie_night", {})
    movie_section["usually_happens"] = bool(normalized_entries) or bool(movie_section.get("usually_happens", False))

    profile["profile_metadata"] = {
        **profile.get("profile_metadata", {}),
        "monthly_sample_count": len(normalized_entries),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "derived_fields": {
            "usual_leave_hour_mean": round(leave_hour, 2),
            "usual_return_hour_mean": round(return_hour, 2),
            "most_common_start_hour_mean": round(movie_hour, 2),
            "common_evening_walk_hour_mean": round(walk_hour, 2),
            "common_shopping_hour_mean": round(shopping_hour, 2),
            "common_start_hour_mean": round(dance_hour, 2),
        },
    }

    return profile


def refresh_profile_from_log(
    log_path: Path,
    profile_path: Path,
    *,
    base_profile_path: Optional[Path] = None,
    window_days: int = 30,
) -> dict[str, Any]:
    entries = load_activity_log(log_path)

    if window_days > 0 and entries:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(window_days))
        filtered_entries: list[dict[str, Any]] = []
        for entry in entries:
            timestamp = entry.get("timestamp")
            if not timestamp:
                filtered_entries.append(entry)
                continue
            parsed = pd.to_datetime(timestamp, errors="coerce", utc=True)
            if pd.isna(parsed):
                filtered_entries.append(entry)
                continue
            if parsed.to_pydatetime() >= cutoff:
                filtered_entries.append(entry)
        if filtered_entries:
            entries = filtered_entries

    base_profile: dict[str, Any] = {}
    if base_profile_path is not None and base_profile_path.exists():
        base_profile = load_profile(base_profile_path)
    elif profile_path.exists():
        base_profile = load_profile(profile_path)
    elif DEFAULT_PROFILE_PATH.exists():
        base_profile = load_profile(DEFAULT_PROFILE_PATH)

    updated_profile = build_profile_from_monthly_log(entries, base_profile=base_profile)
    save_profile(updated_profile, profile_path)
    return updated_profile


def run_monthly_update(
    *,
    log_path: Path = DEFAULT_ACTIVITY_LOG_PATH,
    profile_path: Path = DEFAULT_PROFILE_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    base_profile_path: Optional[Path] = None,
    window_days: int = 30,
    online_timesteps: int = 2_000,
    learning_rate: Optional[float] = 0.00005,
    seed: int = 42,
) -> dict[str, Any]:
    profile = refresh_profile_from_log(
        log_path,
        profile_path,
        base_profile_path=base_profile_path,
        window_days=window_days,
    )

    try:
        if model_path.exists():
            model, _ = fine_tune_ppo(
                profile_path=profile_path,
                model_path=model_path,
                online_timesteps=online_timesteps,
                learning_rate=learning_rate,
            )
        else:
            model = load_or_train_model(
                profile_path=profile_path,
                model_path=model_path,
                total_timesteps=max(5_000, int(online_timesteps)),
                seed=seed,
                run_check=False,
            )
    except Exception:
        model = load_or_train_model(
            profile_path=profile_path,
            model_path=model_path,
            total_timesteps=max(5_000, int(online_timesteps)),
            seed=seed,
            run_check=False,
        )

    return {
        "profile": profile,
        "model": model,
        "profile_path": str(profile_path),
        "model_path": str(model_path),
        "log_path": str(log_path),
        "window_days": int(window_days),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh a monthly routine profile and fine-tune the PPO model.")
    parser.add_argument("--log", type=Path, default=DEFAULT_ACTIVITY_LOG_PATH, help="Rolling activity log (JSONL/CSV/JSON).")
    parser.add_argument("--base-profile", type=Path, default=None, help="Optional baseline profile to blend with new month data.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH, help="Path to the profile JSON to update.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the PPO model to load/update.")
    parser.add_argument("--window-days", type=int, default=30, help="Number of days to keep when refreshing the profile.")
    parser.add_argument("--timesteps", type=int, default=2_000, help="Fine-tuning timesteps to run after refreshing the profile.")
    parser.add_argument("--lr", type=float, default=0.00005, help="Optional learning rate for fine-tuning.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when bootstrapping a model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_monthly_update(
        log_path=args.log,
        profile_path=args.profile,
        model_path=args.model,
        base_profile_path=args.base_profile,
        window_days=args.window_days,
        online_timesteps=args.timesteps,
        learning_rate=args.lr,
        seed=args.seed,
    )
    print("Profile refreshed and PPO model updated.")
    print(f"Profile path: {result['profile_path']}")
    print(f"Model path: {result['model_path']}")
    print(f"Log path: {result['log_path']}")
    print(f"Window days: {result['window_days']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

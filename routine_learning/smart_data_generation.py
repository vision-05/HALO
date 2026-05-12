from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

DEFAULT_INPUT_CSV = Path(__file__).resolve().parent / "user_routine_synthetic_data.csv"
DEFAULT_OUTPUT_JSON = Path(__file__).resolve().parent / "user_routine_profile.json"

WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
WEEKEND_NAMES = ("Saturday", "Sunday")
ALL_DAY_NAMES = WEEKDAY_NAMES + WEEKEND_NAMES


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the synthetic routine dataset and normalize timestamp fields."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "event_type" not in df.columns:
        raise ValueError("Dataset must contain at least 'timestamp' and 'event_type' columns.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "event_type"])
    df["event_type"] = df["event_type"].astype(str)
    df["day_of_week"] = df["day_of_week"].astype(str)
    df["hour_float"] = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60.0
        + df["timestamp"].dt.second / 3600.0
    )
    df["date_only"] = df["timestamp"].dt.date
    return df.sort_values(["timestamp", "event_type"]).reset_index(drop=True)


def _cyclic_distance(a: np.ndarray, b: float) -> np.ndarray:
    """Shortest signed distance on a 24h clock."""
    return ((a - b + 12.0) % 24.0) - 12.0


def _linear_mean_hour(hours: np.ndarray) -> float:
    return float(np.mean(hours)) if hours.size else float("nan")


def _circular_mean_hour(hours: np.ndarray) -> float:
    if hours.size == 0:
        return float("nan")
    angles = 2.0 * np.pi * (hours % 24.0) / 24.0
    mean_angle = float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return float((mean_angle / (2.0 * np.pi)) * 24.0)


def build_kde_profile(hours: np.ndarray, grid_points: int = 144) -> dict[str, Any]:
    """Build a circular KDE profile over the 24h day."""
    hours = np.asarray(hours, dtype=float)
    hours = hours[np.isfinite(hours)]

    if hours.size == 0:
        return {
            "grid_hours": [],
            "density": [],
            "peak_hour": None,
            "mean_hour": None,
            "sample_count": 0,
            "bandwidth_minutes": None,
        }

    grid = np.linspace(0.0, 24.0, grid_points, endpoint=False)
    if hours.size == 1:
        bandwidth_hours = 0.5
    else:
        std = float(np.std(hours, ddof=1))
        bandwidth_hours = max(0.25, 1.06 * std * (hours.size ** (-1.0 / 5.0)))

    # Periodic extension to avoid boundary artifacts near midnight.
    augmented = np.concatenate([hours - 24.0, hours, hours + 24.0])
    diff = grid[:, None] - augmented[None, :]
    density = np.exp(-0.5 * (diff / bandwidth_hours) ** 2).sum(axis=1)
    density /= augmented.size * bandwidth_hours * np.sqrt(2.0 * np.pi)

    dx = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
    area = float(np.sum(density) * dx)
    if area > 0.0:
        density = density / area

    peak_hour = float(grid[int(np.argmax(density))])
    mean_hour = _circular_mean_hour(hours)

    return {
        "grid_hours": [round(float(value), 3) for value in grid],
        "density": [round(float(value), 8) for value in density],
        "peak_hour": round(peak_hour, 2),
        "mean_hour": round(mean_hour, 2),
        "sample_count": int(hours.size),
        "bandwidth_minutes": round(bandwidth_hours * 60.0, 2),
    }


def _select_hours(
    df: pd.DataFrame,
    event_type: str,
    day_names: Optional[tuple[str, ...]] = None,
    hour_min: Optional[float] = None,
    hour_max: Optional[float] = None,
) -> np.ndarray:
    mask = df["event_type"].eq(event_type)
    if day_names is not None:
        mask &= df["day_of_week"].isin(day_names)
    if hour_min is not None:
        mask &= df["hour_float"] >= hour_min
    if hour_max is not None:
        mask &= df["hour_float"] < hour_max
    return df.loc[mask, "hour_float"].to_numpy(dtype=float)


def _mode_value(series: pd.Series, default: Any = None) -> Any:
    if series.empty:
        return default
    modes = series.mode(dropna=True)
    if modes.empty:
        return default
    return modes.iloc[0]


def _mode_numeric(series: pd.Series, default: Optional[float] = None) -> Optional[float]:
    if series.empty:
        return default
    return float(_mode_value(series, default=default))


def _build_time_section(
    df: pd.DataFrame,
    event_type: str,
    label: str,
    day_names: Optional[tuple[str, ...]] = None,
    hour_min: Optional[float] = None,
    hour_max: Optional[float] = None,
) -> dict[str, Any]:
    hours = _select_hours(df, event_type, day_names=day_names, hour_min=hour_min, hour_max=hour_max)
    profile = build_kde_profile(hours)
    linear_mean = _linear_mean_hour(hours)
    circular_mean = _circular_mean_hour(hours)

    return {
        f"{label}_mean_hour": round(linear_mean, 2) if np.isfinite(linear_mean) else None,
        f"{label}_circular_mean_hour": round(circular_mean, 2) if np.isfinite(circular_mean) else None,
        f"{label}_kde_peak_hour": profile["peak_hour"],
        f"{label}_kde_profile": profile,
    }


def build_markov_transition_matrix(df: pd.DataFrame) -> dict[str, Any]:
    """Build a Markov transition matrix from event sequences per day."""
    states = sorted(df["event_type"].dropna().astype(str).unique().tolist())
    states = ["__START__"] + states + ["__END__"]
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for _, day_df in df.groupby("date_only", sort=True):
        day_sequence = day_df.sort_values("timestamp")["event_type"].astype(str).tolist()
        if not day_sequence:
            continue

        transition_counts["__START__"][day_sequence[0]] += 1
        for current_state, next_state in zip(day_sequence, day_sequence[1:]):
            transition_counts[current_state][next_state] += 1
        transition_counts[day_sequence[-1]]["__END__"] += 1

    transition_matrix: dict[str, dict[str, float]] = {}
    for state in states:
        row_counts = transition_counts.get(state, Counter())
        total = sum(row_counts.values())
        transition_matrix[state] = {
            next_state: round((row_counts.get(next_state, 0) / total) if total else 0.0, 6)
            for next_state in states
        }

    return {
        "states": states,
        "transition_matrix": transition_matrix,
    }


def _time_shift_rule(
    df: pd.DataFrame,
    event_type: str,
    threshold: float = 0.6,
) -> dict[str, Any]:
    event_df = df.loc[df["event_type"].eq(event_type)].copy()
    if event_df.empty:
        return {
            "threshold": threshold,
            "correlation": None,
            "baseline_peak_hour": None,
            "expected_shift_hours_if_rain_gt_threshold": None,
            "rule_summary": f"No {event_type} samples available to estimate a rain shift rule.",
            "sample_count": 0,
        }

    dry_df = event_df.loc[event_df["rain_intensity"] <= threshold]
    rainy_df = event_df.loc[event_df["rain_intensity"] > threshold]

    baseline_hours = dry_df["hour_float"].to_numpy(dtype=float)
    if baseline_hours.size == 0:
        baseline_hours = event_df["hour_float"].to_numpy(dtype=float)
    baseline_profile = build_kde_profile(baseline_hours)
    baseline_peak = baseline_profile["peak_hour"]

    all_hours = event_df["hour_float"].to_numpy(dtype=float)
    all_rain = event_df["rain_intensity"].astype(float).to_numpy()
    signed_shift = _cyclic_distance(all_hours, baseline_peak)
    if all_rain.size >= 2 and np.std(all_rain) > 0 and np.std(signed_shift) > 0:
        correlation = float(np.corrcoef(all_rain, signed_shift)[0, 1])
    else:
        correlation = None

    rainy_hours = rainy_df["hour_float"].to_numpy(dtype=float)
    if rainy_hours.size:
        rainy_peak = build_kde_profile(rainy_hours)["peak_hour"]
        expected_shift = float(_cyclic_distance(np.array([rainy_peak]), baseline_peak)[0])
    else:
        rainy_peak = None
        expected_shift = 0.0

    if expected_shift >= 0:
        rule_summary = (
            f"If rain_intensity > {threshold}, expected {event_type} time shifts by +{expected_shift:.2f} hours."
        )
    else:
        rule_summary = (
            f"If rain_intensity > {threshold}, expected {event_type} time shifts by {expected_shift:.2f} hours."
        )

    return {
        "threshold": threshold,
        "correlation": None if correlation is None or np.isnan(correlation) else round(correlation, 4),
        "baseline_peak_hour": baseline_peak,
        "rainy_peak_hour": rainy_peak,
        "expected_shift_hours_if_rain_gt_threshold": round(expected_shift, 2),
        "rule_summary": rule_summary,
        "sample_count": int(event_df.shape[0]),
        "rainy_sample_count": int(rainy_df.shape[0]),
    }


def _heating_preferences(df: pd.DataFrame) -> dict[str, str]:
    heating_map: dict[str, str] = {}
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        # Exclude "Off" values from the mode calculation.
        season_df = df.loc[df["season"].eq(season) & df["heating_temp"].ne("Off")]
        if season == "Summer" or season_df.empty:
            heating_map[season] = "Off"
            continue
        preferred = _mode_value(season_df["heating_temp"], default=21)
        heating_map[season] = str(int(round(float(preferred)))) if preferred is not None else "21"
    return heating_map


def build_user_routine_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Build a richer routine profile from the synthetic data."""
    wake_weekday = _build_time_section(df, "WakeUp", "weekday_wake", day_names=WEEKDAY_NAMES)
    wake_saturday = _build_time_section(df, "WakeUp", "saturday_wake", day_names=("Saturday",))
    wake_sunday = _build_time_section(df, "WakeUp", "sunday_wake", day_names=("Sunday",))

    leave_section = _build_time_section(df, "LeaveHome", "leave_home", day_names=WEEKDAY_NAMES)
    return_section = _build_time_section(df, "ReturnHome", "return_home", day_names=WEEKDAY_NAMES)
    movie_section = _build_time_section(df, "TVOn", "movie_night", day_names=("Friday",))
    dance_section = _build_time_section(df, "DanceClass", "dance_class", day_names=("Sunday",))
    grocery_section = _build_time_section(df, "GroceryShopping", "grocery_shopping", day_names=("Sunday",))
    dog_walk_evening = _build_time_section(df, "DogWalk", "dog_walk_evening", hour_min=15.0)

    weekday_wake_df = df.loc[df["event_type"].eq("WakeUp") & df["day_of_week"].isin(WEEKDAY_NAMES)]
    saturday_wake_df = df.loc[df["event_type"].eq("WakeUp") & df["day_of_week"].eq("Saturday")]
    sunday_wake_df = df.loc[df["event_type"].eq("WakeUp") & df["day_of_week"].eq("Sunday")]

    profile: dict[str, Any] = {
        "wake_up_patterns": {
            "weekday_average_hour": wake_weekday["weekday_wake_mean_hour"],
            "weekday_kde_peak_hour": wake_weekday["weekday_wake_kde_peak_hour"],
            "weekday_kde_profile": wake_weekday["weekday_wake_kde_profile"],
            "saturday_average_hour": wake_saturday["saturday_wake_mean_hour"],
            "saturday_kde_peak_hour": wake_saturday["saturday_wake_kde_peak_hour"],
            "saturday_kde_profile": wake_saturday["saturday_wake_kde_profile"],
            "sunday_average_hour": wake_sunday["sunday_wake_mean_hour"],
            "sunday_kde_peak_hour": wake_sunday["sunday_wake_kde_peak_hour"],
            "sunday_kde_profile": wake_sunday["sunday_wake_kde_profile"],
        },
        "weekday_occupancy_pattern": {
            "usual_leave_hour": leave_section["leave_home_kde_peak_hour"],
            "usual_leave_hour_mean": leave_section["leave_home_mean_hour"],
            "usual_return_hour": return_section["return_home_kde_peak_hour"],
            "usual_return_hour_mean": return_section["return_home_mean_hour"],
            "leave_home_kde_profile": leave_section["leave_home_kde_profile"],
            "return_home_kde_profile": return_section["return_home_kde_profile"],
        },
        "heating_preferences": _heating_preferences(df),
        "family_movie_night": {
            "usually_happens": bool(df["event_type"].eq("TVOn").any()),
            "most_common_start_hour": movie_section["movie_night_kde_peak_hour"],
            "most_common_start_hour_mean": movie_section["movie_night_mean_hour"],
            "movie_night_kde_profile": movie_section["movie_night_kde_profile"],
        },
        "dog_walking_habits": {
            "saturday_walk_frequency": int(df.loc[df["event_type"].eq("DogWalk") & df["day_of_week"].eq("Saturday")].shape[0]),
            "sunday_walk_frequency": int(df.loc[df["event_type"].eq("DogWalk") & df["day_of_week"].eq("Sunday")].shape[0]),
            "common_evening_walk_hour": dog_walk_evening["dog_walk_evening_kde_peak_hour"],
            "common_evening_walk_hour_mean": dog_walk_evening["dog_walk_evening_mean_hour"],
            "dog_walk_evening_kde_profile": dog_walk_evening["dog_walk_evening_kde_profile"],
            "weather_shift_rule": _time_shift_rule(df, "DogWalk"),
        },
        "grocery_shopping": {
            "usually_on_sunday": bool(df["event_type"].eq("GroceryShopping").any()),
            "common_shopping_hour": grocery_section["grocery_shopping_kde_peak_hour"],
            "common_shopping_hour_mean": grocery_section["grocery_shopping_mean_hour"],
            "grocery_kde_profile": grocery_section["grocery_shopping_kde_profile"],
            "weather_shift_rule": _time_shift_rule(df, "GroceryShopping"),
        },
        "dance_class": {
            "usually_on_sunday": bool(df["event_type"].eq("DanceClass").any()),
            "common_start_hour": dance_section["dance_class_kde_peak_hour"],
            "common_start_hour_mean": dance_section["dance_class_mean_hour"],
            "dance_class_kde_profile": dance_section["dance_class_kde_profile"],
        },
        "markov_transition_matrix": build_markov_transition_matrix(df),
        "weather_correlation": {
            "DogWalk": _time_shift_rule(df, "DogWalk"),
            "GroceryShopping": _time_shift_rule(df, "GroceryShopping"),
        },
        "profile_metadata": {
            "generated_from": "user_routine_synthetic_data.csv",
            "sample_rows": int(df.shape[0]),
            "unique_days": int(df["date_only"].nunique()),
            "date_start": df["timestamp"].min().date().isoformat() if not df.empty else None,
            "date_end": df["timestamp"].max().date().isoformat() if not df.empty else None,
            "kde_grid_points": 144,
            "transition_states": build_markov_transition_matrix(df)["states"],
        },
    }

    return profile


def save_profile(profile: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a smarter user routine profile from synthetic data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to the synthetic CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to write the user_routine_profile.json file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_dataset(args.input)
    profile = build_user_routine_profile(df)
    save_profile(profile, args.output)

    print("Profile generated successfully.")
    print(f"Input rows: {df.shape[0]}")
    print(f"Output file: {args.output}")
    print(f"Weekday wake-up KDE peak: {profile['wake_up_patterns']['weekday_kde_peak_hour']}")
    print(f"Dog walk rain shift rule: {profile['weather_correlation']['DogWalk']['rule_summary']}")
    print(f"Grocery rain shift rule: {profile['weather_correlation']['GroceryShopping']['rule_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


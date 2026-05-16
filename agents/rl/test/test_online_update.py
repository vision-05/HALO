from __future__ import annotations

import json
from pathlib import Path

from RL.online_update import build_profile_from_monthly_log, load_activity_log, refresh_profile_from_log


def test_build_profile_from_monthly_log_updates_hour_targets() -> None:
    profile = build_profile_from_monthly_log(
        [
            {
                "leave_hour": 7.5,
                "return_hour": 18.25,
                "movie_hour": 20.0,
                "walk_hour": 17.0,
                "shopping_hour": 16.0,
                "dance_hour": 8.0,
            }
        ],
        base_profile={},
    )

    assert profile["weekday_occupancy_pattern"]["usual_leave_hour_mean"] == 7.5
    assert profile["weekday_occupancy_pattern"]["usual_return_hour_mean"] == 18.25
    assert profile["family_movie_night"]["most_common_start_hour_mean"] == 20.0
    assert profile["profile_metadata"]["monthly_sample_count"] == 1


def test_refresh_profile_from_log_reads_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "activity.jsonl"
    profile_path = tmp_path / "profile.json"
    log_path.write_text(json.dumps({"timestamp": "2026-05-01T18:00:00Z", "leave_hour": 8.0}) + "\n", encoding="utf-8")

    assert load_activity_log(log_path)
    profile = refresh_profile_from_log(log_path, profile_path, window_days=30)

    assert profile_path.exists()
    assert profile["weekday_occupancy_pattern"]["usual_leave_hour_mean"] == 8.0
    assert profile["profile_metadata"]["monthly_sample_count"] == 1


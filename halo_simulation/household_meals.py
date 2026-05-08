"""Shared household meal preferences and 14-day dinner history for LLM grocery context."""

from __future__ import annotations

from collections import deque
from typing import Any

from halo_simulation import config


class HouseholdMealContext:
    """
    Tracks favorite meals per person and recent dinners (last 14 dinner entries per person).
    Used by fused scenario + LLMSpecialistAgent prompts.
    """

    def __init__(self) -> None:
        self._favorites: dict[str, list[str]] = {}
        self._history: dict[str, deque[tuple[int, str]]] = {}

    def register_person(self, person_id: str, favorite_meals: list[str]) -> None:
        self._favorites[person_id] = list(favorite_meals)
        if person_id not in self._history:
            self._history[person_id] = deque(maxlen=14)

    def update_favorites(self, person_id: str, favorite_meals: list[str]) -> None:
        self._favorites[person_id] = list(favorite_meals)

    def record_dinner(self, person_id: str, sim_now: float, meal_name: str) -> None:
        day_index = int(sim_now // float(config.MINUTES_PER_DAY))
        dq = self._history.setdefault(person_id, deque(maxlen=14))
        dq.append((day_index, meal_name.strip() or "dinner"))

    def pick_evening_meal(self, person_id: str, rng: Any) -> str:
        """Pick tonight's meal — bias toward favorites least eaten recently."""
        favs = [f.strip() for f in self._favorites.get(person_id, []) if f.strip()]
        if not favs:
            return "simple meal"
        hist = [m.lower() for _, m in self._history.get(person_id, [])]
        scores: list[tuple[int, str]] = []
        for f in favs:
            fl = f.lower()
            c = sum(1 for m in hist if fl in m or m in fl)
            scores.append((c, f))
        min_c = min(s[0] for s in scores)
        candidates = [f for c, f in scores if c == min_c]
        return str(rng.choice(candidates))

    def summary_for_prompt(self) -> str:
        lines: list[str] = []
        for pid, favs in sorted(self._favorites.items()):
            if not favs:
                continue
            lines.append(f"- {pid}: favorites = {', '.join(favs)}")
        for pid in sorted(self._history.keys()):
            hist = list(self._history[pid])
            if not hist:
                continue
            tail = ", ".join(f"day {d}:{m}" for d, m in hist[-7:])
            lines.append(f"- {pid}: recent dinners (newest last, up to 14): {tail}")
        return "\n".join(lines) if lines else "(no meal registry yet)"

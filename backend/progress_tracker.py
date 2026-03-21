"""
backend/progress_tracker.py
----------------------------
User progress tracking with spaced-repetition schedule.

FIX: Expression of type "None" cannot be assigned to parameter of
     type "int" — all optional numeric fields now default to 0.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class ProgressEntry:
    """Holds learning progress for a single skill."""

    def __init__(
        self,
        skill: str,
        completion: int = 0,        # ✅ FIX: default int, never None
        mastery:    str = "Novice",
    ) -> None:
        self.skill:       str      = skill
        # ✅ FIX: explicit int annotation — Pylance satisfied
        self.completion:  int      = max(0, min(100, int(completion)))
        self.mastery:     str      = mastery
        self.last_updated: str     = datetime.now().isoformat()
        self.review_count: int     = 0
        self.next_review:  str     = datetime.now().isoformat()

    def to_dict(self) -> dict[str, object]:
        return {
            "skill":        self.skill,
            "completion":   self.completion,
            "mastery":      self.mastery,
            "last_updated": self.last_updated,
            "review_count": self.review_count,
            "next_review":  self.next_review,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ProgressEntry":
        # ✅ FIX: int() cast so we never pass None to int field
        entry = cls(
            skill      = str(data.get("skill", "")),
            completion = int(data.get("completion", 0) or 0),
            mastery    = str(data.get("mastery", "Novice")),
        )
        entry.last_updated = str(data.get("last_updated", entry.last_updated))
        entry.review_count = int(data.get("review_count", 0) or 0)
        entry.next_review  = str(data.get("next_review", entry.next_review))
        return entry


class ProgressTracker:
    """Manage a collection of ProgressEntry objects."""

    # Spaced repetition intervals (days)
    INTERVALS: list[int] = [1, 3, 7, 14, 30]

    def __init__(self) -> None:
        self._data: dict[str, ProgressEntry] = {}

    # ----------------------------------------------------------
    def update(
        self,
        skill:      str,
        completion: int,           # ✅ FIX: explicit int parameter
        mastery:    str = "Novice",
    ) -> None:
        """Save or update progress for a skill."""
        if skill in self._data:
            entry = self._data[skill]
            entry.completion  = max(0, min(100, int(completion)))
            entry.mastery     = mastery
            entry.last_updated = datetime.now().isoformat()
            entry.review_count += 1
        else:
            self._data[skill] = ProgressEntry(skill, completion, mastery)

        self._schedule_next_review(skill)

    def _schedule_next_review(self, skill: str) -> None:
        entry   = self._data[skill]
        idx     = min(entry.review_count, len(self.INTERVALS) - 1)
        days    = self.INTERVALS[idx]
        next_dt = datetime.now() + timedelta(days=days)
        entry.next_review = next_dt.isoformat()

    # ----------------------------------------------------------
    def get(self, skill: str) -> Optional[ProgressEntry]:
        return self._data.get(skill)

    def all(self) -> dict[str, ProgressEntry]:
        return self._data

    def summary(self) -> dict[str, object]:
        entries = list(self._data.values())
        avg_completion = (
            sum(e.completion for e in entries) / len(entries)
            if entries else 0
        )
        return {
            "total_skills":    len(entries),
            "avg_completion":  round(avg_completion, 1),
            "completed":       sum(1 for e in entries if e.completion == 100),
        }

    # ----------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        return {k: v.to_dict() for k, v in self._data.items()}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ProgressTracker":
        tracker = cls()
        for skill, entry_data in data.items():
            if isinstance(entry_data, dict):
                tracker._data[skill] = ProgressEntry.from_dict(entry_data)
        return tracker
from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np


MatchHistory = Tuple[int, int, int]


DEFAULT_STATISTICS = {
    "points": 1.0,
    "goals_scored": 1.2,
    "goals_conceded": 1.2,
    "goal_difference": 0.0,
    "win_rate": 0.33,
}


def create_history(
    window: int,
) -> Deque[MatchHistory]:
    if window < 1:
        raise ValueError(
            "History window must be at least 1."
        )

    return deque(
        maxlen=window
    )


def append_result(
    history: Deque[MatchHistory],
    goals_scored: int,
    goals_conceded: int,
) -> None:
    if goals_scored > goals_conceded:
        points = 3

    elif goals_scored == goals_conceded:
        points = 1

    else:
        points = 0

    history.append(
        (
            int(goals_scored),
            int(goals_conceded),
            points,
        )
    )


def summarize_history(
    history: Deque[MatchHistory],
) -> dict[str, float]:
    if not history:
        return dict(
            DEFAULT_STATISTICS
        )

    matches = list(
        history
    )

    points = np.mean(
        [
            match[2]
            for match in matches
        ]
    )

    goals_scored = np.mean(
        [
            match[0]
            for match in matches
        ]
    )

    goals_conceded = np.mean(
        [
            match[1]
            for match in matches
        ]
    )

    wins = sum(
        1
        for match in matches
        if match[2] == 3
    )

    return {
        "points": float(
            points
        ),
        "goals_scored": float(
            goals_scored
        ),
        "goals_conceded": float(
            goals_conceded
        ),
        "goal_difference": float(
            goals_scored
            - goals_conceded
        ),
        "win_rate": float(
            wins / len(matches)
        ),
    }
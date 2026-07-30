from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.services.poisson_simulation_service import (
    PoissonSimulationService,
)


RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_matches.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "poisson_evaluation_report.json"
)

TEST_START_DATE = "2024-07-01"
FORM_WINDOW = 8

INITIAL_ELO = 1500.0
K_FACTOR = 25.0
HOME_ADVANTAGE = 60.0

LABEL_ORDER = ["A", "D", "H"]

HistoryItem = Tuple[
    int,
    int,
    int,
]


def points_for_result(
    goals_scored: int,
    goals_conceded: int,
) -> int:
    if goals_scored > goals_conceded:
        return 3

    if goals_scored == goals_conceded:
        return 1

    return 0


def history_summary(
    history: Deque[HistoryItem],
) -> Dict[str, float]:
    if not history:
        return {
            "form_points": 1.0,
            "goals_scored": 1.2,
            "goals_conceded": 1.2,
            "win_rate": 0.33,
        }

    rows = list(history)

    points = np.mean(
        [row[0] for row in rows]
    )

    goals_scored = np.mean(
        [row[1] for row in rows]
    )

    goals_conceded = np.mean(
        [row[2] for row in rows]
    )

    wins = sum(
        1
        for row in rows
        if row[0] == 3
    )

    return {
        "form_points": float(points),
        "goals_scored": float(
            goals_scored
        ),
        "goals_conceded": float(
            goals_conceded
        ),
        "win_rate": float(
            wins / len(rows)
        ),
    }


def expected_home_score(
    home_elo: float,
    away_elo: float,
) -> float:
    return 1.0 / (
        1.0
        + 10.0
        ** (
            (
                away_elo
                - home_elo
                - HOME_ADVANTAGE
            )
            / 400.0
        )
    )


def update_elo(
    ratings: Dict[str, float],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> None:
    home_elo = ratings[home_team]
    away_elo = ratings[away_team]

    expected_home = expected_home_score(
        home_elo=home_elo,
        away_elo=away_elo,
    )

    if home_goals > away_goals:
        actual_home = 1.0

    elif home_goals == away_goals:
        actual_home = 0.5

    else:
        actual_home = 0.0

    goal_margin = abs(
        home_goals - away_goals
    )

    margin_multiplier = (
        1.0 + np.log1p(goal_margin)
        if goal_margin > 0
        else 1.0
    )

    rating_change = (
        K_FACTOR
        * margin_multiplier
        * (
            actual_home
            - expected_home
        )
    )

    ratings[home_team] = (
        home_elo + rating_change
    )

    ratings[away_team] = (
        away_elo - rating_change
    )


def actual_label(
    home_goals: int,
    away_goals: int,
) -> str:
    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def ordered_probability_row(
    home_probability: float,
    draw_probability: float,
    away_probability: float,
) -> List[float]:
    return [
        away_probability,
        draw_probability,
        home_probability,
    ]


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Run python scripts/download_data.py first."
        )

    matches = pd.read_csv(
        RAW_DATA_PATH
    )

    matches["date"] = pd.to_datetime(
        matches["date"],
        utc=True,
        errors="raise",
    )

    matches = matches.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    ratings: Dict[str, float] = defaultdict(
        lambda: INITIAL_ELO
    )

    histories: Dict[
        str,
        Deque[HistoryItem],
    ] = defaultdict(
        lambda: deque(
            maxlen=FORM_WINDOW
        )
    )

    simulation_service = (
        PoissonSimulationService(
            simulations=20_000,
            random_seed=42,
        )
    )

    test_start = pd.Timestamp(
        TEST_START_DATE,
        tz="UTC",
    )

    actual_labels: List[str] = []
    predicted_labels: List[str] = []
    probability_rows: List[
        List[float]
    ] = []

    evaluated_match_ids: List[
        object
    ] = []

    for row in matches.itertuples(
        index=False
    ):
        home_team = str(
            row.home_team
        )

        away_team = str(
            row.away_team
        )

        home_goals = int(
            row.home_goals
        )

        away_goals = int(
            row.away_goals
        )

        home_summary = history_summary(
            histories[home_team]
        )

        away_summary = history_summary(
            histories[away_team]
        )

        home_elo = float(
            ratings[home_team]
        )

        away_elo = float(
            ratings[away_team]
        )

        is_test_match = (
            row.date >= test_start
            and str(row.competition) == "CL"
        )

        if is_test_match:
            simulation = (
                simulation_service
                .simulate(
                    home_summary=home_summary,
                    away_summary=away_summary,
                    home_elo=home_elo,
                    away_elo=away_elo,
                )
            )

            probability_map = {
                "H": (
                    simulation
                    .home_win_probability
                ),
                "D": (
                    simulation
                    .draw_probability
                ),
                "A": (
                    simulation
                    .away_win_probability
                ),
            }

            predicted_label = max(
                probability_map,
                key=probability_map.get,
            )

            actual_labels.append(
                actual_label(
                    home_goals=home_goals,
                    away_goals=away_goals,
                )
            )

            predicted_labels.append(
                predicted_label
            )

            probability_rows.append(
                ordered_probability_row(
                    home_probability=(
                        simulation
                        .home_win_probability
                    ),
                    draw_probability=(
                        simulation
                        .draw_probability
                    ),
                    away_probability=(
                        simulation
                        .away_win_probability
                    ),
                )
            )

            evaluated_match_ids.append(
                row.match_id
            )

        histories[home_team].append(
            (
                points_for_result(
                    goals_scored=home_goals,
                    goals_conceded=away_goals,
                ),
                home_goals,
                away_goals,
            )
        )

        histories[away_team].append(
            (
                points_for_result(
                    goals_scored=away_goals,
                    goals_conceded=home_goals,
                ),
                away_goals,
                home_goals,
            )
        )

        update_elo(
            ratings=ratings,
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
        )

    if not actual_labels:
        raise RuntimeError(
            "No Champions League test matches were found."
        )

    probability_array = np.asarray(
        probability_rows,
        dtype=float,
    )

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels,
    )

    evaluation_log_loss = log_loss(
        actual_labels,
        probability_array,
        labels=LABEL_ORDER,
    )

    report_text = classification_report(
        actual_labels,
        predicted_labels,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=LABEL_ORDER,
    )

    report = {
        "test_start_date": (
            TEST_START_DATE
        ),
        "test_matches": int(
            len(actual_labels)
        ),
        "accuracy": float(
            accuracy
        ),
        "log_loss": float(
            evaluation_log_loss
        ),
        "classification_report": (
            report_text
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "evaluated_match_ids": (
            evaluated_match_ids
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=2,
            default=str,
        )

    print("=" * 70)
    print("POISSON MONTE CARLO EVALUATION")
    print("=" * 70)

    print(
        f"Test matches: "
        f"{len(actual_labels):,}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"Log loss: "
        f"{evaluation_log_loss:.4f}"
    )

    print()
    print("Classification report:")
    print(report_text)

    print("Confusion matrix:")
    print(matrix)

    print()
    print(
        "Validated ML baseline:"
    )

    print(
        "Accuracy: 58.20%"
    )

    print(
        "Log loss: 1.0075"
    )

    print()
    print(
        f"Report saved to:\n"
        f"{REPORT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
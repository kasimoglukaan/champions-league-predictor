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


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.features.feature_builder import (
    FeatureBuilder,
)
from src.models.machine_learning_model import (
    MachineLearningModel,
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

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "match_model.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "hybrid_ensemble_report.json"
)

TEST_START_DATE = "2024-07-01"

FORM_WINDOW = 8

INITIAL_ELO = 1500.0
K_FACTOR = 25.0
HOME_ADVANTAGE = 60.0

LABEL_ORDER = [
    "A",
    "D",
    "H",
]

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


def actual_label(
    home_goals: int,
    away_goals: int,
) -> str:
    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


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
        [
            row[0]
            for row in rows
        ]
    )

    goals_scored = np.mean(
        [
            row[1]
            for row in rows
        ]
    )

    goals_conceded = np.mean(
        [
            row[2]
            for row in rows
        ]
    )

    wins = sum(
        1
        for row in rows
        if row[0] == 3
    )

    return {
        "form_points": float(
            points
        ),
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
    home_elo = ratings[
        home_team
    ]

    away_elo = ratings[
        away_team
    ]

    expected_home = (
        expected_home_score(
            home_elo=home_elo,
            away_elo=away_elo,
        )
    )

    if home_goals > away_goals:
        actual_home = 1.0

    elif home_goals == away_goals:
        actual_home = 0.5

    else:
        actual_home = 0.0

    goal_margin = abs(
        home_goals
        - away_goals
    )

    if goal_margin > 0:
        margin_multiplier = (
            1.0
            + np.log1p(
                goal_margin
            )
        )

    else:
        margin_multiplier = 1.0

    rating_change = (
        K_FACTOR
        * margin_multiplier
        * (
            actual_home
            - expected_home
        )
    )

    ratings[home_team] = (
        home_elo
        + rating_change
    )

    ratings[away_team] = (
        away_elo
        - rating_change
    )


def normalize_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    row_sums = clipped.sum(
        axis=1,
        keepdims=True,
    )

    return clipped / row_sums


def probabilities_to_labels(
    probabilities: np.ndarray,
) -> List[str]:
    best_indices = np.argmax(
        probabilities,
        axis=1,
    )

    return [
        LABEL_ORDER[index]
        for index in best_indices
    ]


def evaluate_probabilities(
    actual_labels: List[str],
    probabilities: np.ndarray,
) -> Dict[str, object]:
    normalized = (
        normalize_probabilities(
            probabilities
        )
    )

    predicted_labels = (
        probabilities_to_labels(
            normalized
        )
    )

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels,
    )

    evaluation_log_loss = log_loss(
        actual_labels,
        normalized,
        labels=LABEL_ORDER,
    )

    report_text = (
        classification_report(
            actual_labels,
            predicted_labels,
            labels=LABEL_ORDER,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=LABEL_ORDER,
    )

    return {
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
            matrix
        ),
        "predicted_labels": (
            predicted_labels
        ),
    }


def build_poisson_probabilities(
    matches: pd.DataFrame,
    test_match_ids: set,
) -> Dict[object, List[float]]:
    ratings: Dict[
        str,
        float,
    ] = defaultdict(
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

    poisson_probabilities: Dict[
        object,
        List[float],
    ] = {}

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

        home_summary = (
            history_summary(
                histories[
                    home_team
                ]
            )
        )

        away_summary = (
            history_summary(
                histories[
                    away_team
                ]
            )
        )

        home_elo = float(
            ratings[
                home_team
            ]
        )

        away_elo = float(
            ratings[
                away_team
            ]
        )

        if row.match_id in test_match_ids:
            simulation = (
                simulation_service
                .simulate(
                    home_summary=(
                        home_summary
                    ),
                    away_summary=(
                        away_summary
                    ),
                    home_elo=home_elo,
                    away_elo=away_elo,
                )
            )

            poisson_probabilities[
                row.match_id
            ] = [
                simulation
                .away_win_probability,

                simulation
                .draw_probability,

                simulation
                .home_win_probability,
            ]

        histories[
            home_team
        ].append(
            (
                points_for_result(
                    goals_scored=(
                        home_goals
                    ),
                    goals_conceded=(
                        away_goals
                    ),
                ),
                home_goals,
                away_goals,
            )
        )

        histories[
            away_team
        ].append(
            (
                points_for_result(
                    goals_scored=(
                        away_goals
                    ),
                    goals_conceded=(
                        home_goals
                    ),
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

    return poisson_probabilities


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Match data was not found. "
            "Run python scripts/download_data.py first."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model was not found: "
            f"{MODEL_PATH}"
        )

    print("=" * 72)
    print("HYBRID ML + POISSON ENSEMBLE EVALUATION")
    print("=" * 72)

    matches = pd.read_csv(
        RAW_DATA_PATH
    )

    matches["date"] = pd.to_datetime(
        matches["date"],
        utc=True,
        errors="raise",
    )

    matches = matches.sort_values(
        [
            "date",
            "match_id",
        ]
    ).reset_index(
        drop=True
    )

    feature_builder = (
        FeatureBuilder(
            initial_elo=(
                INITIAL_ELO
            ),
            k_factor=K_FACTOR,
            home_advantage=(
                HOME_ADVANTAGE
            ),
            form_window=FORM_WINDOW,
        )
    )

    print(
        f"Loaded matches: "
        f"{len(matches):,}"
    )

    print(
        "Generating ML features..."
    )

    dataset = (
        feature_builder.build(
            matches
        )
    )

    dataset["date"] = pd.to_datetime(
        dataset["date"],
        utc=True,
        errors="raise",
    )

    test_start = pd.Timestamp(
        TEST_START_DATE,
        tz="UTC",
    )

    test_data = dataset[
        (
            dataset["date"]
            >= test_start
        )
        & (
            dataset["competition"]
            == "CL"
        )
    ].copy()

    test_data = test_data.sort_values(
        [
            "date",
            "match_id",
        ]
    ).reset_index(
        drop=True
    )

    if test_data.empty:
        raise RuntimeError(
            "No Champions League test "
            "matches were found."
        )

    feature_columns = list(
        FeatureBuilder
        .FEATURE_COLUMNS
    )

    X_test = test_data[
        feature_columns
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    actual_labels = (
        test_data[
            "target"
        ]
        .astype(str)
        .tolist()
    )

    model = MachineLearningModel()

    model.load(
        str(MODEL_PATH)
    )

    ml_probabilities = (
        model.predict_proba(
            X_test
        )
    )

    ml_probabilities = (
        normalize_probabilities(
            np.asarray(
                ml_probabilities,
                dtype=float,
            )
        )
    )

    test_match_ids = set(
        test_data[
            "match_id"
        ].tolist()
    )

    print(
        "Generating Poisson probabilities..."
    )

    poisson_by_match_id = (
        build_poisson_probabilities(
            matches=matches,
            test_match_ids=(
                test_match_ids
            ),
        )
    )

    missing_match_ids = [
        match_id
        for match_id in test_data[
            "match_id"
        ].tolist()
        if match_id
        not in poisson_by_match_id
    ]

    if missing_match_ids:
        raise RuntimeError(
            "Poisson probabilities are "
            "missing for "
            f"{len(missing_match_ids)} "
            "test matches."
        )

    poisson_probabilities = (
        np.asarray(
            [
                poisson_by_match_id[
                    match_id
                ]
                for match_id in test_data[
                    "match_id"
                ].tolist()
            ],
            dtype=float,
        )
    )

    poisson_probabilities = (
        normalize_probabilities(
            poisson_probabilities
        )
    )

    print(
        f"Test matches: "
        f"{len(test_data):,}"
    )

    baseline_ml_result = (
        evaluate_probabilities(
            actual_labels=(
                actual_labels
            ),
            probabilities=(
                ml_probabilities
            ),
        )
    )

    baseline_poisson_result = (
        evaluate_probabilities(
            actual_labels=(
                actual_labels
            ),
            probabilities=(
                poisson_probabilities
            ),
        )
    )

    weights = [
        1.00,
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55,
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20,
        0.15,
        0.10,
        0.05,
        0.00,
    ]

    ensemble_results: Dict[
        str,
        Dict[str, object],
    ] = {}

    print()
    print("ENSEMBLE WEIGHT RESULTS")
    print("-" * 72)

    for ml_weight in weights:
        poisson_weight = (
            1.0
            - ml_weight
        )

        hybrid_probabilities = (
            ml_weight
            * ml_probabilities
            + poisson_weight
            * poisson_probabilities
        )

        hybrid_result = (
            evaluate_probabilities(
                actual_labels=(
                    actual_labels
                ),
                probabilities=(
                    hybrid_probabilities
                ),
            )
        )

        result_name = (
            f"ml_{ml_weight:.2f}"
            f"_poisson_{poisson_weight:.2f}"
        )

        ensemble_results[
            result_name
        ] = {
            "ml_weight": float(
                ml_weight
            ),
            "poisson_weight": float(
                poisson_weight
            ),
            "accuracy": (
                hybrid_result[
                    "accuracy"
                ]
            ),
            "log_loss": (
                hybrid_result[
                    "log_loss"
                ]
            ),
            "classification_report": (
                hybrid_result[
                    "classification_report"
                ]
            ),
            "confusion_matrix": (
                hybrid_result[
                    "confusion_matrix"
                ].tolist()
            ),
        }

        print(
            f"ML {ml_weight:>5.0%} | "
            f"Poisson {poisson_weight:>5.0%} | "
            f"Accuracy "
            f"{hybrid_result['accuracy']:.2%} | "
            f"Log loss "
            f"{hybrid_result['log_loss']:.4f}"
        )

    best_accuracy_name = max(
        ensemble_results,
        key=lambda name: (
            ensemble_results[
                name
            ]["accuracy"],
            -ensemble_results[
                name
            ]["log_loss"],
        ),
    )

    best_log_loss_name = min(
        ensemble_results,
        key=lambda name: (
            ensemble_results[
                name
            ]["log_loss"],
            -ensemble_results[
                name
            ]["accuracy"],
        ),
    )

    best_accuracy_result = (
        ensemble_results[
            best_accuracy_name
        ]
    )

    best_log_loss_result = (
        ensemble_results[
            best_log_loss_name
        ]
    )

    beats_ml_accuracy = (
        best_accuracy_result[
            "accuracy"
        ]
        > baseline_ml_result[
            "accuracy"
        ]
    )

    beats_ml_log_loss = (
        best_log_loss_result[
            "log_loss"
        ]
        < baseline_ml_result[
            "log_loss"
        ]
    )

    report = {
        "test_start_date": (
            TEST_START_DATE
        ),
        "test_matches": int(
            len(test_data)
        ),
        "label_order": (
            LABEL_ORDER
        ),
        "ml_baseline": {
            "accuracy": (
                baseline_ml_result[
                    "accuracy"
                ]
            ),
            "log_loss": (
                baseline_ml_result[
                    "log_loss"
                ]
            ),
            "confusion_matrix": (
                baseline_ml_result[
                    "confusion_matrix"
                ].tolist()
            ),
        },
        "poisson_baseline": {
            "accuracy": (
                baseline_poisson_result[
                    "accuracy"
                ]
            ),
            "log_loss": (
                baseline_poisson_result[
                    "log_loss"
                ]
            ),
            "confusion_matrix": (
                baseline_poisson_result[
                    "confusion_matrix"
                ].tolist()
            ),
        },
        "best_accuracy_ensemble": {
            "name": (
                best_accuracy_name
            ),
            **best_accuracy_result,
        },
        "best_log_loss_ensemble": {
            "name": (
                best_log_loss_name
            ),
            **best_log_loss_result,
        },
        "beats_ml_accuracy": bool(
            beats_ml_accuracy
        ),
        "beats_ml_log_loss": bool(
            beats_ml_log_loss
        ),
        "all_results": (
            ensemble_results
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
        )

    print()
    print("=" * 72)
    print("HYBRID ENSEMBLE SUMMARY")
    print("=" * 72)

    print("ML baseline:")
    print(
        f"Accuracy: "
        f"{baseline_ml_result['accuracy']:.2%}"
    )
    print(
        f"Log loss: "
        f"{baseline_ml_result['log_loss']:.4f}"
    )

    print()
    print("Poisson baseline:")
    print(
        f"Accuracy: "
        f"{baseline_poisson_result['accuracy']:.2%}"
    )
    print(
        f"Log loss: "
        f"{baseline_poisson_result['log_loss']:.4f}"
    )

    print()
    print(
        "Best accuracy ensemble:"
    )
    print(
        f"ML weight: "
        f"{best_accuracy_result['ml_weight']:.0%}"
    )
    print(
        f"Poisson weight: "
        f"{best_accuracy_result['poisson_weight']:.0%}"
    )
    print(
        f"Accuracy: "
        f"{best_accuracy_result['accuracy']:.2%}"
    )
    print(
        f"Log loss: "
        f"{best_accuracy_result['log_loss']:.4f}"
    )

    print()
    print(
        "Best log-loss ensemble:"
    )
    print(
        f"ML weight: "
        f"{best_log_loss_result['ml_weight']:.0%}"
    )
    print(
        f"Poisson weight: "
        f"{best_log_loss_result['poisson_weight']:.0%}"
    )
    print(
        f"Accuracy: "
        f"{best_log_loss_result['accuracy']:.2%}"
    )
    print(
        f"Log loss: "
        f"{best_log_loss_result['log_loss']:.4f}"
    )

    print()
    print(
        "Accuracy improvement over ML: "
        f"{best_accuracy_result['accuracy'] - baseline_ml_result['accuracy']:+.2%}"
    )

    print(
        "Log-loss improvement over ML: "
        f"{baseline_ml_result['log_loss'] - best_log_loss_result['log_loss']:+.4f}"
    )

    print()
    print(
        "Best-accuracy confusion matrix:"
    )

    print(
        np.asarray(
            best_accuracy_result[
                "confusion_matrix"
            ]
        )
    )

    print()
    print(
        "Best-accuracy classification report:"
    )

    print(
        best_accuracy_result[
            "classification_report"
        ]
    )

    print(
        f"Report saved to:\n"
        f"{REPORT_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
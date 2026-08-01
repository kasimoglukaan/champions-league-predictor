from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
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
    / "production_cl_ensemble_report.json"
)

RESULTS_CSV_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "production_cl_ensemble_results.csv"
)


TEST_START_DATE = "2024-07-01"

FORM_WINDOW = 8

INITIAL_ELO = 1500.0
K_FACTOR = 25.0
HOME_ADVANTAGE = 60.0

POISSON_SIMULATIONS = 20_000
RANDOM_SEED = 42

LABEL_ORDER = [
    "A",
    "D",
    "H",
]

ENSEMBLE_CONFIGURATIONS = {
    "ML only": (
        1.00,
        0.00,
    ),
    "Poisson only": (
        0.00,
        1.00,
    ),
    "70 ML / 30 Poisson": (
        0.70,
        0.30,
    ),
    "60 ML / 40 Poisson": (
        0.60,
        0.40,
    ),
    "50 ML / 50 Poisson": (
        0.50,
        0.50,
    ),
    "40 ML / 60 Poisson": (
        0.40,
        0.60,
    ),
}


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


def determine_actual_label(
    home_goals: int,
    away_goals: int,
) -> str:
    if home_goals > away_goals:
        return "H"

    if away_goals > home_goals:
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

    rows = list(
        history
    )

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

    margin_multiplier = (
        1.0
        + np.log1p(
            goal_margin
        )
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

    ratings[
        home_team
    ] = (
        home_elo
        + rating_change
    )

    ratings[
        away_team
    ] = (
        away_elo
        - rating_change
    )


def load_matches() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw production dataset was not found: "
            f"{RAW_DATA_PATH}"
        )

    matches = pd.read_csv(
        RAW_DATA_PATH,
        low_memory=False,
    )

    required_columns = {
        "match_id",
        "date",
        "competition",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    }

    missing_columns = (
        required_columns
        - set(
            matches.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    matches = matches.copy()

    matches["date"] = pd.to_datetime(
        matches["date"],
        utc=True,
        errors="raise",
    )

    matches["home_goals"] = pd.to_numeric(
        matches["home_goals"],
        errors="raise",
    ).astype(int)

    matches["away_goals"] = pd.to_numeric(
        matches["away_goals"],
        errors="raise",
    ).astype(int)

    matches["competition"] = (
        matches["competition"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    matches["winner"] = [
        determine_actual_label(
            home_goals=int(
                home_goals
            ),
            away_goals=int(
                away_goals
            ),
        )
        for home_goals, away_goals
        in zip(
            matches["home_goals"],
            matches["away_goals"],
        )
    ]

    return (
        matches.sort_values(
            by=[
                "date",
                "match_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def normalize_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        probabilities,
        dtype=float,
    )

    values = np.clip(
        values,
        1e-12,
        1.0,
    )

    row_sums = values.sum(
        axis=1,
        keepdims=True,
    )

    return (
        values
        / row_sums
    )


def probabilities_to_labels(
    probabilities: np.ndarray,
) -> np.ndarray:
    best_indexes = np.argmax(
        probabilities,
        axis=1,
    )

    return np.asarray(
        LABEL_ORDER
    )[best_indexes]


def multiclass_brier_score(
    actual_labels: Iterable[str],
    probabilities: np.ndarray,
) -> float:
    labels = list(
        actual_labels
    )

    one_hot = np.zeros(
        (
            len(labels),
            len(LABEL_ORDER),
        ),
        dtype=float,
    )

    label_indexes = {
        label: index
        for index, label
        in enumerate(
            LABEL_ORDER
        )
    }

    for row_index, label in enumerate(
        labels
    ):
        one_hot[
            row_index,
            label_indexes[label],
        ] = 1.0

    squared_errors = (
        probabilities
        - one_hot
    ) ** 2

    return float(
        np.mean(
            np.sum(
                squared_errors,
                axis=1,
            )
        )
    )


def evaluate_probabilities(
    actual_labels: Iterable[str],
    probabilities: np.ndarray,
) -> Dict[str, object]:
    labels = np.asarray(
        list(
            actual_labels
        ),
        dtype=str,
    )

    normalized = normalize_probabilities(
        probabilities
    )

    predicted_labels = (
        probabilities_to_labels(
            normalized
        )
    )

    report = classification_report(
        labels,
        predicted_labels,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predicted_labels,
        labels=LABEL_ORDER,
    )

    return {
        "match_count": int(
            len(labels)
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predicted_labels,
            )
        ),
        "log_loss": float(
            log_loss(
                labels,
                normalized,
                labels=LABEL_ORDER,
            )
        ),
        "brier_score": (
            multiclass_brier_score(
                actual_labels=labels,
                probabilities=normalized,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predicted_labels,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predicted_labels,
                labels=LABEL_ORDER,
                average="weighted",
                zero_division=0,
            )
        ),
        "draw_recall": float(
            recall_score(
                labels,
                predicted_labels,
                labels=[
                    "D",
                ],
                average=None,
                zero_division=0,
            )[0]
        ),
        "away_recall": float(
            report["A"]["recall"]
        ),
        "home_recall": float(
            report["H"]["recall"]
        ),
        "predicted_away_wins": int(
            np.sum(
                predicted_labels
                == "A"
            )
        ),
        "predicted_draws": int(
            np.sum(
                predicted_labels
                == "D"
            )
        ),
        "predicted_home_wins": int(
            np.sum(
                predicted_labels
                == "H"
            )
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "classification_report": (
            report
        ),
    }


def build_poisson_probabilities(
    matches: pd.DataFrame,
    evaluation_match_ids: set,
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
            simulations=(
                POISSON_SIMULATIONS
            ),
            random_seed=(
                RANDOM_SEED
            ),
        )
    )

    probabilities_by_id: Dict[
        object,
        List[float],
    ] = {}

    for row in matches.itertuples(
        index=False
    ):
        match_id = row.match_id

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
            histories[
                home_team
            ]
        )

        away_summary = history_summary(
            histories[
                away_team
            ]
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

        if (
            match_id
            in evaluation_match_ids
        ):
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

            probabilities_by_id[
                match_id
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

    return probabilities_by_id


def build_ml_probabilities(
    matches: pd.DataFrame,
    test_data: pd.DataFrame,
) -> np.ndarray:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Production model was not found: "
            f"{MODEL_PATH}"
        )

    feature_builder = FeatureBuilder(
        initial_elo=INITIAL_ELO,
        k_factor=K_FACTOR,
        home_advantage=HOME_ADVANTAGE,
        form_window=FORM_WINDOW,
    )

    print(
        "Generating production ML features..."
    )

    dataset = feature_builder.build(
        matches
    )

    dataset = dataset.set_index(
        "match_id",
        drop=False,
    )

    test_ids = (
        test_data[
            "match_id"
        ].tolist()
    )

    missing_ids = [
        match_id
        for match_id in test_ids
        if match_id
        not in dataset.index
    ]

    if missing_ids:
        raise RuntimeError(
            "ML features are missing for "
            f"{len(missing_ids)} test matches."
        )

    X_test = (
        dataset.loc[
            test_ids,
            FeatureBuilder.FEATURE_COLUMNS,
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    model = MachineLearningModel()

    model.load(
        str(
            MODEL_PATH
        )
    )

    probabilities = model.predict_proba(
        X_test
    )

    return normalize_probabilities(
        np.asarray(
            probabilities,
            dtype=float,
        )
    )


def print_result_table(
    results: Dict[
        str,
        Dict[str, object],
    ],
) -> None:
    print()
    print("=" * 108)
    print(
        "PRODUCTION CHAMPIONS LEAGUE ENSEMBLE RESULTS"
    )
    print("=" * 108)

    print(
        f"{'Configuration':<25}"
        f"{'Accuracy':>11}"
        f"{'Log loss':>12}"
        f"{'Brier':>11}"
        f"{'Macro F1':>12}"
        f"{'Draw recall':>14}"
        f"{'Matches':>10}"
    )

    print(
        "-" * 108
    )

    sorted_results = sorted(
        results.items(),
        key=lambda item: (
            -item[1][
                "accuracy"
            ],
            item[1][
                "log_loss"
            ],
        ),
    )

    for (
        configuration_name,
        metrics,
    ) in sorted_results:
        print(
            f"{configuration_name:<25}"
            f"{metrics['accuracy']:>10.2%}"
            f"{metrics['log_loss']:>12.4f}"
            f"{metrics['brier_score']:>11.4f}"
            f"{metrics['macro_f1']:>12.2%}"
            f"{metrics['draw_recall']:>14.2%}"
            f"{metrics['match_count']:>10,}"
        )


def create_results_dataframe(
    results: Dict[
        str,
        Dict[str, object],
    ],
) -> pd.DataFrame:
    rows = []

    for (
        configuration_name,
        metrics,
    ) in results.items():
        (
            ml_weight,
            poisson_weight,
        ) = ENSEMBLE_CONFIGURATIONS[
            configuration_name
        ]

        rows.append(
            {
                "configuration": (
                    configuration_name
                ),
                "ml_weight": (
                    ml_weight
                ),
                "poisson_weight": (
                    poisson_weight
                ),
                "match_count": metrics[
                    "match_count"
                ],
                "accuracy": metrics[
                    "accuracy"
                ],
                "log_loss": metrics[
                    "log_loss"
                ],
                "brier_score": metrics[
                    "brier_score"
                ],
                "macro_f1": metrics[
                    "macro_f1"
                ],
                "weighted_f1": metrics[
                    "weighted_f1"
                ],
                "draw_recall": metrics[
                    "draw_recall"
                ],
                "away_recall": metrics[
                    "away_recall"
                ],
                "home_recall": metrics[
                    "home_recall"
                ],
                "predicted_away_wins": (
                    metrics[
                        "predicted_away_wins"
                    ]
                ),
                "predicted_draws": (
                    metrics[
                        "predicted_draws"
                    ]
                ),
                "predicted_home_wins": (
                    metrics[
                        "predicted_home_wins"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    print()
    print("=" * 108)
    print(
        "PRODUCTION CHAMPIONS LEAGUE "
        "ML + POISSON VALIDATION"
    )
    print("=" * 108)

    matches = load_matches()

    test_start = pd.Timestamp(
        TEST_START_DATE,
        tz="UTC",
    )

    test_data = matches[
        (
            matches[
                "date"
            ]
            >= test_start
        )
        & (
            matches[
                "competition"
            ]
            == "CL"
        )
    ].copy()

    test_data = (
        test_data.sort_values(
            by=[
                "date",
                "match_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if test_data.empty:
        raise RuntimeError(
            "No Champions League test "
            "matches were found."
        )

    print(
        f"Loaded production matches: "
        f"{len(matches):,}"
    )

    print(
        f"CL test matches: "
        f"{len(test_data):,}"
    )

    print(
        f"Test date range: "
        f"{test_data['date'].min()} "
        f"to "
        f"{test_data['date'].max()}"
    )

    ml_probabilities = (
        build_ml_probabilities(
            matches=matches,
            test_data=test_data,
        )
    )

    print(
        "Generating production Poisson probabilities..."
    )

    test_match_ids = set(
        test_data[
            "match_id"
        ].tolist()
    )

    poisson_by_match_id = (
        build_poisson_probabilities(
            matches=matches,
            evaluation_match_ids=(
                test_match_ids
            ),
        )
    )

    missing_poisson_ids = [
        match_id
        for match_id in test_data[
            "match_id"
        ].tolist()
        if match_id
        not in poisson_by_match_id
    ]

    if missing_poisson_ids:
        raise RuntimeError(
            "Poisson probabilities are missing "
            f"for {len(missing_poisson_ids)} "
            "test matches."
        )

    poisson_probabilities = np.asarray(
        [
            poisson_by_match_id[
                match_id
            ]
            for match_id
            in test_data[
                "match_id"
            ].tolist()
        ],
        dtype=float,
    )

    poisson_probabilities = (
        normalize_probabilities(
            poisson_probabilities
        )
    )

    actual_labels = (
        test_data[
            "winner"
        ]
        .astype(str)
        .to_numpy()
    )

    results: Dict[
        str,
        Dict[str, object],
    ] = {}

    for (
        configuration_name,
        weights,
    ) in ENSEMBLE_CONFIGURATIONS.items():
        ml_weight = float(
            weights[0]
        )

        poisson_weight = float(
            weights[1]
        )

        combined_probabilities = (
            ml_weight
            * ml_probabilities
            + poisson_weight
            * poisson_probabilities
        )

        results[
            configuration_name
        ] = evaluate_probabilities(
            actual_labels=(
                actual_labels
            ),
            probabilities=(
                combined_probabilities
            ),
        )

    print_result_table(
        results
    )

    best_accuracy_name = max(
        results,
        key=lambda name: (
            results[
                name
            ]["accuracy"],
            -results[
                name
            ]["log_loss"],
        ),
    )

    best_log_loss_name = min(
        results,
        key=lambda name: (
            results[
                name
            ]["log_loss"],
            -results[
                name
            ]["accuracy"],
        ),
    )

    best_brier_name = min(
        results,
        key=lambda name: (
            results[
                name
            ]["brier_score"],
            -results[
                name
            ]["accuracy"],
        ),
    )

    ml_baseline = results[
        "ML only"
    ]

    best_accuracy_result = results[
        best_accuracy_name
    ]

    best_log_loss_result = results[
        best_log_loss_name
    ]

    best_brier_result = results[
        best_brier_name
    ]

    print()
    print("=" * 108)
    print("FINAL SUMMARY")
    print("=" * 108)

    print(
        "Best accuracy configuration: "
        f"{best_accuracy_name}"
    )

    print(
        "Accuracy: "
        f"{best_accuracy_result['accuracy']:.2%}"
    )

    print(
        "Accuracy difference vs ML only: "
        f"{best_accuracy_result['accuracy'] - ml_baseline['accuracy']:+.2%}"
    )

    print()
    print(
        "Best log-loss configuration: "
        f"{best_log_loss_name}"
    )

    print(
        "Log loss: "
        f"{best_log_loss_result['log_loss']:.4f}"
    )

    print(
        "Log-loss improvement vs ML only: "
        f"{ml_baseline['log_loss'] - best_log_loss_result['log_loss']:+.4f}"
    )

    print()
    print(
        "Best Brier configuration: "
        f"{best_brier_name}"
    )

    print(
        "Brier score: "
        f"{best_brier_result['brier_score']:.4f}"
    )

    print()
    print(
        "Best-accuracy confusion matrix [A, D, H]:"
    )

    print(
        np.asarray(
            best_accuracy_result[
                "confusion_matrix"
            ]
        )
    )

    report = {
        "experiment_name": (
            "production_champions_league_ensemble"
        ),
        "data_path": str(
            RAW_DATA_PATH
        ),
        "model_path": str(
            MODEL_PATH
        ),
        "test_start_date": (
            TEST_START_DATE
        ),
        "test_matches": int(
            len(
                test_data
            )
        ),
        "test_start": (
            test_data[
                "date"
            ].min().isoformat()
        ),
        "test_end": (
            test_data[
                "date"
            ].max().isoformat()
        ),
        "best_accuracy_configuration": (
            best_accuracy_name
        ),
        "best_log_loss_configuration": (
            best_log_loss_name
        ),
        "best_brier_configuration": (
            best_brier_name
        ),
        "results": (
            results
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            ensure_ascii=False,
            indent=2,
        )

    results_dataframe = (
        create_results_dataframe(
            results
        )
    )

    results_dataframe.to_csv(
        RESULTS_CSV_PATH,
        index=False,
    )

    print()
    print(
        "JSON report saved to:"
    )

    print(
        REPORT_PATH
    )

    print()
    print(
        "CSV results saved to:"
    )

    print(
        RESULTS_CSV_PATH
    )

    print("=" * 108)


if __name__ == "__main__":
    main()
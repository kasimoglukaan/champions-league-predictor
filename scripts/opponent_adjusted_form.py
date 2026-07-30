import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.features.feature_builder import FeatureBuilder


RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_matches.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "opponent_adjusted_form_report.json"
)


TEST_START_DATE = "2024-07-01"

LABEL_ORDER = ["A", "D", "H"]

INITIAL_ELO = 1500.0
K_FACTOR = 25.0
HOME_ADVANTAGE = 60.0
FORM_WINDOW = 8


# points, goal difference, opponent Elo
AdjustedHistoryItem = Tuple[
    float,
    float,
    float,
]


def create_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def ordered_probabilities(
    model: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    raw_probabilities = model.predict_proba(
        features
    )

    classes = list(model.classes_)

    ordered = np.zeros(
        (
            len(features),
            len(LABEL_ORDER),
        ),
        dtype=float,
    )

    for output_index, label in enumerate(
        LABEL_ORDER
    ):
        if label not in classes:
            raise ValueError(
                f"Expected class is missing: {label}"
            )

        class_index = classes.index(label)

        ordered[:, output_index] = (
            raw_probabilities[:, class_index]
        )

    return ordered


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    targets: pd.Series,
) -> Dict[str, object]:
    predictions = model.predict(features)

    probabilities = ordered_probabilities(
        model=model,
        features=features,
    )

    return {
        "accuracy": float(
            accuracy_score(
                targets,
                predictions,
            )
        ),
        "log_loss": float(
            log_loss(
                targets,
                probabilities,
                labels=LABEL_ORDER,
            )
        ),
        "classification_report": (
            classification_report(
                targets,
                predictions,
                labels=LABEL_ORDER,
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                targets,
                predictions,
                labels=LABEL_ORDER,
            )
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


def points_for_result(
    goals_scored: int,
    goals_conceded: int,
) -> float:
    if goals_scored > goals_conceded:
        return 3.0

    if goals_scored == goals_conceded:
        return 1.0

    return 0.0


def history_features(
    history: Deque[AdjustedHistoryItem],
) -> Dict[str, float]:
    if not history:
        return {
            "adjusted_points": 1.0,
            "adjusted_goal_difference": 0.0,
            "average_opponent_elo": INITIAL_ELO,
        }

    history_rows = list(history)

    adjusted_points = []

    adjusted_goal_differences = []

    opponent_elos = []

    for (
        points,
        goal_difference,
        opponent_elo,
    ) in history_rows:
        opponent_strength = (
            opponent_elo / INITIAL_ELO
        )

        adjusted_points.append(
            points * opponent_strength
        )

        adjusted_goal_differences.append(
            goal_difference
            * opponent_strength
        )

        opponent_elos.append(
            opponent_elo
        )

    return {
        "adjusted_points": float(
            np.mean(adjusted_points)
        ),
        "adjusted_goal_difference": float(
            np.mean(
                adjusted_goal_differences
            )
        ),
        "average_opponent_elo": float(
            np.mean(opponent_elos)
        ),
    }


def build_adjusted_features(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = matches.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        utc=True,
        errors="raise",
    )

    dataframe = dataframe.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    ratings: Dict[str, float] = defaultdict(
        lambda: INITIAL_ELO
    )

    histories: Dict[
        str,
        Deque[AdjustedHistoryItem],
    ] = defaultdict(
        lambda: deque(
            maxlen=FORM_WINDOW
        )
    )

    feature_rows: List[Dict[str, float]] = []

    for row in dataframe.itertuples(
        index=False
    ):
        home_team = str(row.home_team)
        away_team = str(row.away_team)

        home_goals = int(row.home_goals)
        away_goals = int(row.away_goals)

        home_elo_before = float(
            ratings[home_team]
        )

        away_elo_before = float(
            ratings[away_team]
        )

        home_history = history_features(
            histories[home_team]
        )

        away_history = history_features(
            histories[away_team]
        )

        feature_rows.append(
            {
                "match_id": row.match_id,

                "home_adjusted_form": (
                    home_history[
                        "adjusted_points"
                    ]
                ),
                "away_adjusted_form": (
                    away_history[
                        "adjusted_points"
                    ]
                ),
                "adjusted_form_difference": (
                    home_history[
                        "adjusted_points"
                    ]
                    - away_history[
                        "adjusted_points"
                    ]
                ),

                "home_adjusted_goal_difference": (
                    home_history[
                        "adjusted_goal_difference"
                    ]
                ),
                "away_adjusted_goal_difference": (
                    away_history[
                        "adjusted_goal_difference"
                    ]
                ),
                "adjusted_goal_difference_gap": (
                    home_history[
                        "adjusted_goal_difference"
                    ]
                    - away_history[
                        "adjusted_goal_difference"
                    ]
                ),

                "home_average_opponent_elo": (
                    home_history[
                        "average_opponent_elo"
                    ]
                ),
                "away_average_opponent_elo": (
                    away_history[
                        "average_opponent_elo"
                    ]
                ),
                "opponent_quality_difference": (
                    home_history[
                        "average_opponent_elo"
                    ]
                    - away_history[
                        "average_opponent_elo"
                    ]
                ),
            }
        )

        home_points = points_for_result(
            goals_scored=home_goals,
            goals_conceded=away_goals,
        )

        away_points = points_for_result(
            goals_scored=away_goals,
            goals_conceded=home_goals,
        )

        histories[home_team].append(
            (
                home_points,
                float(
                    home_goals
                    - away_goals
                ),
                away_elo_before,
            )
        )

        histories[away_team].append(
            (
                away_points,
                float(
                    away_goals
                    - home_goals
                ),
                home_elo_before,
            )
        )

        update_elo(
            ratings=ratings,
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
        )

    return pd.DataFrame(feature_rows)


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Run this command first:\n"
            "python scripts/download_data.py"
        )

    print("=" * 70)
    print("OPPONENT-ADJUSTED FORM EXPERIMENT")
    print("=" * 70)

    matches = pd.read_csv(
        RAW_DATA_PATH
    )

    print(
        f"Loaded matches: {len(matches):,}"
    )

    feature_builder = FeatureBuilder(
        initial_elo=INITIAL_ELO,
        k_factor=K_FACTOR,
        home_advantage=HOME_ADVANTAGE,
        form_window=FORM_WINDOW,
    )

    baseline_dataset = (
        feature_builder.build(matches)
    )

    adjusted_features = (
        build_adjusted_features(matches)
    )

    dataset = baseline_dataset.merge(
        adjusted_features,
        on="match_id",
        how="left",
        validate="one_to_one",
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

    train_data = dataset[
        dataset["date"] < test_start
    ].copy()

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

    if train_data.empty:
        raise RuntimeError(
            "Training data is empty."
        )

    if test_data.empty:
        raise RuntimeError(
            "Test data is empty."
        )

    baseline_features = list(
        FeatureBuilder.FEATURE_COLUMNS
    )

    adjusted_form_features = [
        "home_adjusted_form",
        "away_adjusted_form",
        "adjusted_form_difference",
    ]

    adjusted_goal_features = [
        "home_adjusted_goal_difference",
        "away_adjusted_goal_difference",
        "adjusted_goal_difference_gap",
    ]

    opponent_quality_features = [
        "home_average_opponent_elo",
        "away_average_opponent_elo",
        "opponent_quality_difference",
    ]

    experiments = {
        "baseline": (
            baseline_features
        ),
        "baseline_plus_adjusted_form": (
            baseline_features
            + adjusted_form_features
        ),
        "baseline_plus_adjusted_goals": (
            baseline_features
            + adjusted_goal_features
        ),
        "baseline_plus_opponent_quality": (
            baseline_features
            + opponent_quality_features
        ),
        "baseline_plus_all_adjusted": (
            baseline_features
            + adjusted_form_features
            + adjusted_goal_features
            + opponent_quality_features
        ),
    }

    y_train = train_data[
        "target"
    ].copy()

    y_test = test_data[
        "target"
    ].copy()

    results: Dict[str, Dict] = {}

    print(
        f"Training matches: {len(train_data):,}"
    )

    print(
        f"Test CL matches: {len(test_data):,}"
    )

    for experiment_name, feature_columns in (
        experiments.items()
    ):
        feature_columns = list(
            dict.fromkeys(
                feature_columns
            )
        )

        X_train = train_data[
            feature_columns
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X_test = test_data[
            feature_columns
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        result = evaluate_model(
            model=model,
            features=X_test,
            targets=y_test,
        )

        results[experiment_name] = {
            "feature_count": len(
                feature_columns
            ),
            "features": feature_columns,
            "accuracy": (
                result["accuracy"]
            ),
            "log_loss": (
                result["log_loss"]
            ),
            "classification_report": (
                result[
                    "classification_report"
                ]
            ),
            "confusion_matrix": (
                result[
                    "confusion_matrix"
                ].tolist()
            ),
        }

        print()
        print(experiment_name)
        print("-" * 60)

        print(
            f"Features: "
            f"{len(feature_columns)}"
        )

        print(
            f"Accuracy: "
            f"{result['accuracy']:.2%}"
        )

        print(
            f"Log loss: "
            f"{result['log_loss']:.4f}"
        )

    best_accuracy_experiment = max(
        results,
        key=lambda name: (
            results[name]["accuracy"],
            -results[name]["log_loss"],
        ),
    )

    best_log_loss_experiment = min(
        results,
        key=lambda name: (
            results[name]["log_loss"],
            -results[name]["accuracy"],
        ),
    )

    report = {
        "training_matches": int(
            len(train_data)
        ),
        "test_matches": int(
            len(test_data)
        ),
        "test_start_date": (
            TEST_START_DATE
        ),
        "best_accuracy_experiment": (
            best_accuracy_experiment
        ),
        "best_log_loss_experiment": (
            best_log_loss_experiment
        ),
        "results": results,
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
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print(
        "Best accuracy experiment: "
        f"{best_accuracy_experiment}"
    )

    print(
        "Accuracy: "
        f"{results[best_accuracy_experiment]['accuracy']:.2%}"
    )

    print(
        "Log loss: "
        f"{results[best_accuracy_experiment]['log_loss']:.4f}"
    )

    print()
    print(
        "Best log-loss experiment: "
        f"{best_log_loss_experiment}"
    )

    print(
        "Accuracy: "
        f"{results[best_log_loss_experiment]['accuracy']:.2%}"
    )

    print(
        "Log loss: "
        f"{results[best_log_loss_experiment]['log_loss']:.4f}"
    )

    print()
    print(
        f"Report saved to:\n"
        f"{REPORT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
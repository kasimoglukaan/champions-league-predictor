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
    / "home_away_form_report.json"
)

TEST_START_DATE = "2024-07-01"
FORM_WINDOW = 8
LABEL_ORDER = ["A", "D", "H"]


# points, goals scored, goals conceded
HistoryItem = Tuple[int, int, int]


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
        class_index = classes.index(label)

        ordered[
            :,
            output_index,
        ] = raw_probabilities[
            :,
            class_index,
        ]

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


def result_points(
    goals_scored: int,
    goals_conceded: int,
) -> int:
    if goals_scored > goals_conceded:
        return 3

    if goals_scored == goals_conceded:
        return 1

    return 0


def history_statistics(
    history: Deque[HistoryItem],
) -> Dict[str, float]:
    if not history:
        return {
            "points": 1.0,
            "goals_scored": 1.2,
            "goals_conceded": 1.2,
            "goal_difference": 0.0,
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
        "points": float(points),
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
            wins / len(rows)
        ),
    }


def build_home_away_features(
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

    home_histories: Dict[
        str,
        Deque[HistoryItem],
    ] = defaultdict(
        lambda: deque(
            maxlen=FORM_WINDOW
        )
    )

    away_histories: Dict[
        str,
        Deque[HistoryItem],
    ] = defaultdict(
        lambda: deque(
            maxlen=FORM_WINDOW
        )
    )

    feature_rows: List[Dict] = []

    for row in dataframe.itertuples(
        index=False
    ):
        home_team = str(row.home_team)
        away_team = str(row.away_team)

        home_stats = history_statistics(
            home_histories[home_team]
        )

        away_stats = history_statistics(
            away_histories[away_team]
        )

        feature_rows.append(
            {
                "match_id": row.match_id,

                "home_specific_points": (
                    home_stats["points"]
                ),
                "away_specific_points": (
                    away_stats["points"]
                ),
                "venue_points_difference": (
                    home_stats["points"]
                    - away_stats["points"]
                ),

                "home_specific_goals_scored": (
                    home_stats[
                        "goals_scored"
                    ]
                ),
                "away_specific_goals_scored": (
                    away_stats[
                        "goals_scored"
                    ]
                ),

                "home_specific_goals_conceded": (
                    home_stats[
                        "goals_conceded"
                    ]
                ),
                "away_specific_goals_conceded": (
                    away_stats[
                        "goals_conceded"
                    ]
                ),

                "home_specific_goal_difference": (
                    home_stats[
                        "goal_difference"
                    ]
                ),
                "away_specific_goal_difference": (
                    away_stats[
                        "goal_difference"
                    ]
                ),
                "venue_goal_difference_gap": (
                    home_stats[
                        "goal_difference"
                    ]
                    - away_stats[
                        "goal_difference"
                    ]
                ),

                "home_specific_win_rate": (
                    home_stats["win_rate"]
                ),
                "away_specific_win_rate": (
                    away_stats["win_rate"]
                ),
                "venue_win_rate_difference": (
                    home_stats["win_rate"]
                    - away_stats["win_rate"]
                ),
            }
        )

        home_goals = int(
            row.home_goals
        )

        away_goals = int(
            row.away_goals
        )

        home_histories[
            home_team
        ].append(
            (
                result_points(
                    home_goals,
                    away_goals,
                ),
                home_goals,
                away_goals,
            )
        )

        away_histories[
            away_team
        ].append(
            (
                result_points(
                    away_goals,
                    home_goals,
                ),
                away_goals,
                home_goals,
            )
        )

    return pd.DataFrame(
        feature_rows
    )


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Run python scripts/download_data.py first."
        )

    print("=" * 70)
    print("HOME / AWAY SPECIFIC FORM EXPERIMENT")
    print("=" * 70)

    matches = pd.read_csv(
        RAW_DATA_PATH
    )

    print(
        f"Loaded matches: {len(matches):,}"
    )

    feature_builder = FeatureBuilder(
        initial_elo=1500,
        k_factor=25,
        home_advantage=60,
        form_window=8,
    )

    baseline_dataset = (
        feature_builder.build(matches)
    )

    venue_features = (
        build_home_away_features(
            matches
        )
    )

    dataset = baseline_dataset.merge(
        venue_features,
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

    baseline_features = list(
        FeatureBuilder.FEATURE_COLUMNS
    )

    points_features = [
        "home_specific_points",
        "away_specific_points",
        "venue_points_difference",
    ]

    goals_features = [
        "home_specific_goals_scored",
        "away_specific_goals_scored",
        "home_specific_goals_conceded",
        "away_specific_goals_conceded",
        "home_specific_goal_difference",
        "away_specific_goal_difference",
        "venue_goal_difference_gap",
    ]

    win_rate_features = [
        "home_specific_win_rate",
        "away_specific_win_rate",
        "venue_win_rate_difference",
    ]

    experiments = {
        "baseline": (
            baseline_features
        ),
        "baseline_plus_venue_points": (
            baseline_features
            + points_features
        ),
        "baseline_plus_venue_goals": (
            baseline_features
            + goals_features
        ),
        "baseline_plus_venue_win_rate": (
            baseline_features
            + win_rate_features
        ),
        "baseline_plus_all_venue": (
            baseline_features
            + points_features
            + goals_features
            + win_rate_features
        ),
    }

    y_train = train_data[
        "target"
    ].copy()

    y_test = test_data[
        "target"
    ].copy()

    results = {}

    print(
        f"Training matches: {len(train_data):,}"
    )

    print(
        f"Test CL matches: {len(test_data):,}"
    )

    for experiment_name, columns in (
        experiments.items()
    ):
        feature_columns = list(
            dict.fromkeys(columns)
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
            "feature_count": int(
                len(feature_columns)
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

    best_accuracy_name = max(
        results,
        key=lambda name: (
            results[name]["accuracy"],
            -results[name]["log_loss"],
        ),
    )

    best_log_loss_name = min(
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
            best_accuracy_name
        ),
        "best_log_loss_experiment": (
            best_log_loss_name
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
    print("HOME / AWAY FORM SUMMARY")
    print("=" * 70)

    print(
        f"Best accuracy experiment: "
        f"{best_accuracy_name}"
    )

    print(
        f"Accuracy: "
        f"{results[best_accuracy_name]['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{results[best_accuracy_name]['log_loss']:.4f}"
    )

    print()
    print(
        f"Best log-loss experiment: "
        f"{best_log_loss_name}"
    )

    print(
        f"Accuracy: "
        f"{results[best_log_loss_name]['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{results[best_log_loss_name]['log_loss']:.4f}"
    )

    print()
    print(
        f"Report saved to:\n"
        f"{REPORT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
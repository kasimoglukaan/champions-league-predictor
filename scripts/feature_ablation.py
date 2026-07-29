import json
import sys
from pathlib import Path
from typing import Dict, List

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
    / "feature_ablation_report.json"
)

TEST_START_DATE = "2024-07-01"

LABEL_ORDER = ["A", "D", "H"]


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
    predictions = model.predict(
        features
    )

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


def add_candidate_features(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataset.copy()

    # 1. Absolute team-strength difference
    if {
        "home_elo",
        "away_elo",
    }.issubset(dataframe.columns):
        dataframe[
            "absolute_elo_difference"
        ] = (
            dataframe["home_elo"]
            - dataframe["away_elo"]
        ).abs()

        dataframe[
            "squared_elo_difference"
        ] = (
            dataframe["elo_difference"]
            ** 2
        )

        dataframe[
            "close_elo_match"
        ] = (
            dataframe[
                "absolute_elo_difference"
            ]
            < 75
        ).astype(int)

    # 2. Attacking versus defensive matchup
    if {
        "home_goals_scored",
        "away_goals_conceded",
    }.issubset(dataframe.columns):
        dataframe[
            "home_attack_matchup"
        ] = (
            dataframe[
                "home_goals_scored"
            ]
            - dataframe[
                "away_goals_conceded"
            ]
        )

    if {
        "away_goals_scored",
        "home_goals_conceded",
    }.issubset(dataframe.columns):
        dataframe[
            "away_attack_matchup"
        ] = (
            dataframe[
                "away_goals_scored"
            ]
            - dataframe[
                "home_goals_conceded"
            ]
        )

    if {
        "home_attack_matchup",
        "away_attack_matchup",
    }.issubset(dataframe.columns):
        dataframe[
            "attack_matchup_difference"
        ] = (
            dataframe[
                "home_attack_matchup"
            ]
            - dataframe[
                "away_attack_matchup"
            ]
        )

    # 3. Similarity features for draw detection
    if {
        "home_form_points",
        "away_form_points",
    }.issubset(dataframe.columns):
        dataframe[
            "form_similarity"
        ] = (
            dataframe[
                "home_form_points"
            ]
            - dataframe[
                "away_form_points"
            ]
        ).abs()

    if {
        "home_goal_difference",
        "away_goal_difference",
    }.issubset(dataframe.columns):
        dataframe[
            "goal_difference_similarity"
        ] = (
            dataframe[
                "home_goal_difference"
            ]
            - dataframe[
                "away_goal_difference"
            ]
        ).abs()

    if {
        "home_win_rate",
        "away_win_rate",
    }.issubset(dataframe.columns):
        dataframe[
            "win_rate_difference"
        ] = (
            dataframe[
                "home_win_rate"
            ]
            - dataframe[
                "away_win_rate"
            ]
        )

        dataframe[
            "win_rate_similarity"
        ] = (
            dataframe[
                "home_win_rate"
            ]
            - dataframe[
                "away_win_rate"
            ]
        ).abs()

    # 4. Rest advantage
    if {
        "home_rest_days",
        "away_rest_days",
    }.issubset(dataframe.columns):
        dataframe[
            "rest_days_difference"
        ] = (
            dataframe[
                "home_rest_days"
            ]
            - dataframe[
                "away_rest_days"
            ]
        )

    return dataframe


def available_columns(
    dataset: pd.DataFrame,
    requested_columns: List[str],
) -> List[str]:
    return [
        column
        for column in requested_columns
        if column in dataset.columns
    ]


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Run python scripts/download_data.py first."
        )

    print("=" * 70)
    print("FEATURE ABLATION EXPERIMENT")
    print("=" * 70)

    matches = pd.read_csv(
        RAW_DATA_PATH
    )

    print(
        f"Loaded {len(matches):,} matches."
    )

    feature_builder = FeatureBuilder(
        initial_elo=1500,
        k_factor=25,
        home_advantage=60,
        form_window=8,
    )

    dataset = feature_builder.build(
        matches
    )

    dataset = add_candidate_features(
        dataset
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
            "Champions League test data is empty."
        )

    baseline_features = list(
        FeatureBuilder.FEATURE_COLUMNS
    )

    elo_features = available_columns(
        dataset,
        [
            "absolute_elo_difference",
            "squared_elo_difference",
            "close_elo_match",
        ],
    )

    matchup_features = available_columns(
        dataset,
        [
            "home_attack_matchup",
            "away_attack_matchup",
            "attack_matchup_difference",
        ],
    )

    similarity_features = available_columns(
        dataset,
        [
            "form_similarity",
            "goal_difference_similarity",
            "win_rate_difference",
            "win_rate_similarity",
        ],
    )

    rest_features = available_columns(
        dataset,
        [
            "rest_days_difference",
        ],
    )

    experiments = {
        "baseline": (
            baseline_features
        ),
        "baseline_plus_elo": (
            baseline_features
            + elo_features
        ),
        "baseline_plus_matchups": (
            baseline_features
            + matchup_features
        ),
        "baseline_plus_similarity": (
            baseline_features
            + similarity_features
        ),
        "baseline_plus_rest": (
            baseline_features
            + rest_features
        ),
        "baseline_plus_elo_matchups": (
            baseline_features
            + elo_features
            + matchup_features
        ),
        "all_candidate_features": (
            baseline_features
            + elo_features
            + matchup_features
            + similarity_features
            + rest_features
        ),
    }

    y_train = train_data[
        "target"
    ].copy()

    y_test = test_data[
        "target"
    ].copy()

    results: Dict[
        str,
        Dict[str, object],
    ] = {}

    print(
        f"Training matches: "
        f"{len(train_data):,}"
    )

    print(
        f"Test CL matches: "
        f"{len(test_data):,}"
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

        try:
            model.fit(
                X_train,
                y_train,
            )

            result = evaluate_model(
                model=model,
                features=X_test,
                targets=y_test,
            )

        except Exception as error:
            print()
            print(
                f"{experiment_name} failed:"
            )
            print(error)

            results[
                experiment_name
            ] = {
                "failed": True,
                "error": str(error),
            }

            continue

        results[
            experiment_name
        ] = {
            "failed": False,
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
        print(
            experiment_name
        )
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

    successful_results = {
        name: result
        for name, result in results.items()
        if not result.get(
            "failed",
            False,
        )
    }

    if not successful_results:
        raise RuntimeError(
            "All feature experiments failed."
        )

    best_accuracy_name = max(
        successful_results,
        key=lambda name: (
            successful_results[
                name
            ]["accuracy"],
            -successful_results[
                name
            ]["log_loss"],
        ),
    )

    best_log_loss_name = min(
        successful_results,
        key=lambda name: (
            successful_results[
                name
            ]["log_loss"],
            -successful_results[
                name
            ]["accuracy"],
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
    print("FEATURE ABLATION SUMMARY")
    print("=" * 70)

    print(
        f"Best accuracy experiment: "
        f"{best_accuracy_name}"
    )

    print(
        f"Accuracy: "
        f"{successful_results[best_accuracy_name]['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{successful_results[best_accuracy_name]['log_loss']:.4f}"
    )

    print()
    print(
        f"Best log-loss experiment: "
        f"{best_log_loss_name}"
    )

    print(
        f"Accuracy: "
        f"{successful_results[best_log_loss_name]['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{successful_results[best_log_loss_name]['log_loss']:.4f}"
    )

    print()
    print(
        f"Report saved to:\n"
        f"{REPORT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
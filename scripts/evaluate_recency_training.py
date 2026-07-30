from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
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


RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_matches.csv"
)

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "match_model.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "recency_training_report.json"
)

BEST_CANDIDATE_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "recency_candidate.joblib"
)


TEST_START_DATE = "2024-07-01"

INITIAL_ELO = 1500
K_FACTOR = 25
HOME_ADVANTAGE = 60
FORM_WINDOW = 8

LABEL_ORDER = [
    "A",
    "D",
    "H",
]


EXPERIMENTS = [
    {
        "name": "all_history",
        "years": None,
        "half_life_days": None,
    },
    {
        "name": "last_1_year",
        "years": 1,
        "half_life_days": None,
    },
    {
        "name": "last_2_years",
        "years": 2,
        "half_life_days": None,
    },
    {
        "name": "last_3_years",
        "years": 3,
        "half_life_days": None,
    },
    {
        "name": "time_decay_180_days",
        "years": None,
        "half_life_days": 180,
    },
    {
        "name": "time_decay_365_days",
        "years": None,
        "half_life_days": 365,
    },
    {
        "name": "time_decay_730_days",
        "years": None,
        "half_life_days": 730,
    },
    {
        "name": "last_3_years_decay_365",
        "years": 3,
        "half_life_days": 365,
    },
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
                    C=0.01,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def normalize_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    totals = clipped.sum(
        axis=1,
        keepdims=True,
    )

    return clipped / totals


def reorder_probabilities(
    model: Pipeline,
    probabilities: np.ndarray,
) -> np.ndarray:
    classes = [
        str(label)
        for label in model.classes_
    ]

    ordered_columns = []

    for label in LABEL_ORDER:
        if label not in classes:
            raise ValueError(
                f"Class {label!r} not found "
                f"in model classes: {classes}"
            )

        class_index = classes.index(
            label
        )

        ordered_columns.append(
            probabilities[
                :,
                class_index,
            ]
        )

    ordered = np.column_stack(
        ordered_columns
    )

    return normalize_probabilities(
        ordered
    )


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
    normalized = normalize_probabilities(
        probabilities
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

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=LABEL_ORDER,
    )

    report_text = classification_report(
        actual_labels,
        predicted_labels,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    actual_array = np.asarray(
        actual_labels
    )

    predicted_array = np.asarray(
        predicted_labels
    )

    draw_mask = (
        actual_array == "D"
    )

    predicted_draw_mask = (
        predicted_array == "D"
    )

    correct_draws = int(
        np.sum(
            draw_mask
            & predicted_draw_mask
        )
    )

    total_draws = int(
        np.sum(
            draw_mask
        )
    )

    predicted_draws = int(
        np.sum(
            predicted_draw_mask
        )
    )

    draw_recall = (
        correct_draws / total_draws
        if total_draws > 0
        else 0.0
    )

    draw_precision = (
        correct_draws / predicted_draws
        if predicted_draws > 0
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "log_loss": float(
            evaluation_log_loss
        ),
        "draw_recall": float(
            draw_recall
        ),
        "draw_precision": float(
            draw_precision
        ),
        "predicted_draws": predicted_draws,
        "correct_draws": correct_draws,
        "confusion_matrix": (
            matrix.tolist()
        ),
        "classification_report": (
            report_text
        ),
    }


def select_training_window(
    development_data: pd.DataFrame,
    years: Optional[int],
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    if years is None:
        return development_data.copy()

    window_start = (
        test_start
        - pd.DateOffset(
            years=years
        )
    )

    selected = development_data[
        development_data["date"]
        >= window_start
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"No training data found "
            f"for the last {years} years."
        )

    return selected


def create_time_weights(
    training_dates: pd.Series,
    reference_date: pd.Timestamp,
    half_life_days: Optional[int],
) -> Optional[np.ndarray]:
    if half_life_days is None:
        return None

    ages = (
        reference_date
        - training_dates
    ).dt.total_seconds() / 86400.0

    ages = np.maximum(
        ages.to_numpy(
            dtype=float
        ),
        0.0,
    )

    weights = np.power(
        0.5,
        ages / float(
            half_life_days
        ),
    )

    minimum_weight = 0.05

    weights = np.clip(
        weights,
        minimum_weight,
        1.0,
    )

    return weights.astype(
        float
    )


def fit_experiment(
    training_data: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_columns: List[str],
    test_start: pd.Timestamp,
    half_life_days: Optional[int],
) -> tuple[
    Pipeline,
    np.ndarray,
    Optional[np.ndarray],
]:
    X_train = training_data[
        feature_columns
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    y_train = (
        training_data[
            "target"
        ]
        .astype(str)
    )

    sample_weights = create_time_weights(
        training_dates=(
            training_data[
                "date"
            ]
        ),
        reference_date=test_start,
        half_life_days=(
            half_life_days
        ),
    )

    model = create_model()

    if sample_weights is None:
        model.fit(
            X_train,
            y_train,
        )

    else:
        model.fit(
            X_train,
            y_train,
            classifier__sample_weight=(
                sample_weights
            ),
        )

    raw_probabilities = (
        model.predict_proba(
            test_features
        )
    )

    ordered_probabilities = (
        reorder_probabilities(
            model=model,
            probabilities=np.asarray(
                raw_probabilities,
                dtype=float,
            ),
        )
    )

    return (
        model,
        ordered_probabilities,
        sample_weights,
    )


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Match data not found: "
            f"{RAW_DATA_PATH}"
        )

    if not PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Production model not found: "
            f"{PRODUCTION_MODEL_PATH}"
        )

    print("=" * 76)
    print("RECENCY AND TIME-DECAY TRAINING EXPERIMENT")
    print("=" * 76)

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

    print(
        f"Loaded matches: "
        f"{len(matches):,}"
    )

    feature_builder = FeatureBuilder(
        initial_elo=INITIAL_ELO,
        k_factor=K_FACTOR,
        home_advantage=(
            HOME_ADVANTAGE
        ),
        form_window=FORM_WINDOW,
    )

    print(
        "Generating pre-match features..."
    )

    dataset = feature_builder.build(
        matches
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

    development_data = dataset[
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

    development_data = (
        development_data.sort_values(
            [
                "date",
                "match_id",
            ]
        ).reset_index(
            drop=True
        )
    )

    test_data = test_data.sort_values(
        [
            "date",
            "match_id",
        ]
    ).reset_index(
        drop=True
    )

    if development_data.empty:
        raise RuntimeError(
            "Development data is empty."
        )

    if test_data.empty:
        raise RuntimeError(
            "Test data is empty."
        )

    feature_columns = list(
        FeatureBuilder.FEATURE_COLUMNS
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

    test_labels = (
        test_data[
            "target"
        ]
        .astype(str)
        .tolist()
    )

    production_model = (
        MachineLearningModel()
    )

    production_model.load(
        str(
            PRODUCTION_MODEL_PATH
        )
    )

    production_probabilities = (
        normalize_probabilities(
            np.asarray(
                production_model.predict_proba(
                    X_test
                ),
                dtype=float,
            )
        )
    )

    production_result = (
        evaluate_probabilities(
            actual_labels=test_labels,
            probabilities=(
                production_probabilities
            ),
        )
    )

    print()
    print("DATA SPLIT")
    print("-" * 76)

    print(
        f"Development matches: "
        f"{len(development_data):,}"
    )

    print(
        f"Unseen CL test matches: "
        f"{len(test_data):,}"
    )

    print(
        "Development period: "
        f"{development_data['date'].min()} "
        "to "
        f"{development_data['date'].max()}"
    )

    print(
        "Test period: "
        f"{test_data['date'].min()} "
        "to "
        f"{test_data['date'].max()}"
    )

    print()
    print("PRODUCTION BASELINE")
    print("-" * 76)

    print(
        f"Accuracy: "
        f"{production_result['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{production_result['log_loss']:.4f}"
    )

    results: Dict[
        str,
        Dict[str, object],
    ] = {}

    candidate_models: Dict[
        str,
        Pipeline,
    ] = {}

    print()
    print("RECENCY EXPERIMENT RESULTS")
    print("-" * 76)

    for experiment in EXPERIMENTS:
        experiment_name = str(
            experiment["name"]
        )

        years_value = (
            experiment["years"]
        )

        half_life_value = (
            experiment[
                "half_life_days"
            ]
        )

        years = (
            int(years_value)
            if years_value is not None
            else None
        )

        half_life_days = (
            int(half_life_value)
            if half_life_value
            is not None
            else None
        )

        selected_training_data = (
            select_training_window(
                development_data=(
                    development_data
                ),
                years=years,
                test_start=test_start,
            )
        )

        (
            model,
            probabilities,
            sample_weights,
        ) = fit_experiment(
            training_data=(
                selected_training_data
            ),
            test_features=X_test,
            feature_columns=(
                feature_columns
            ),
            test_start=test_start,
            half_life_days=(
                half_life_days
            ),
        )

        result = evaluate_probabilities(
            actual_labels=test_labels,
            probabilities=probabilities,
        )

        minimum_weight = None
        average_weight = None

        if sample_weights is not None:
            minimum_weight = float(
                np.min(
                    sample_weights
                )
            )

            average_weight = float(
                np.mean(
                    sample_weights
                )
            )

        results[
            experiment_name
        ] = {
            "name": experiment_name,
            "years": years,
            "half_life_days": (
                half_life_days
            ),
            "training_matches": int(
                len(
                    selected_training_data
                )
            ),
            "training_start": str(
                selected_training_data[
                    "date"
                ].min()
            ),
            "training_end": str(
                selected_training_data[
                    "date"
                ].max()
            ),
            "minimum_weight": (
                minimum_weight
            ),
            "average_weight": (
                average_weight
            ),
            **result,
        }

        candidate_models[
            experiment_name
        ] = model

        print(
            f"{experiment_name:<26} | "
            f"Train "
            f"{len(selected_training_data):>5,} | "
            f"Accuracy "
            f"{result['accuracy']:.2%} | "
            f"Log loss "
            f"{result['log_loss']:.4f} | "
            f"Draw recall "
            f"{result['draw_recall']:.2%}"
        )

    best_accuracy_name = max(
        results,
        key=lambda name: (
            results[name][
                "accuracy"
            ],
            -results[name][
                "log_loss"
            ],
        ),
    )

    best_log_loss_name = min(
        results,
        key=lambda name: (
            results[name][
                "log_loss"
            ],
            -results[name][
                "accuracy"
            ],
        ),
    )

    best_accuracy_result = (
        results[
            best_accuracy_name
        ]
    )

    best_log_loss_result = (
        results[
            best_log_loss_name
        ]
    )

    best_candidate_bundle = {
        "model_type": (
            "recency_weighted_logistic"
        ),
        "experiment_name": (
            best_accuracy_name
        ),
        "model": (
            candidate_models[
                best_accuracy_name
            ]
        ),
        "feature_columns": (
            feature_columns
        ),
        "label_order": (
            LABEL_ORDER
        ),
        "years": (
            best_accuracy_result[
                "years"
            ]
        ),
        "half_life_days": (
            best_accuracy_result[
                "half_life_days"
            ]
        ),
        "test_start_date": (
            TEST_START_DATE
        ),
        "test_accuracy": (
            best_accuracy_result[
                "accuracy"
            ]
        ),
        "test_log_loss": (
            best_accuracy_result[
                "log_loss"
            ]
        ),
    }

    BEST_CANDIDATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_candidate_bundle,
        BEST_CANDIDATE_PATH,
    )

    accuracy_improvement = (
        best_accuracy_result[
            "accuracy"
        ]
        - production_result[
            "accuracy"
        ]
    )

    log_loss_improvement = (
        production_result[
            "log_loss"
        ]
        - best_log_loss_result[
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
        "production_baseline": (
            production_result
        ),
        "best_accuracy_experiment": {
            "name": (
                best_accuracy_name
            ),
            **best_accuracy_result,
        },
        "best_log_loss_experiment": {
            "name": (
                best_log_loss_name
            ),
            **best_log_loss_result,
        },
        "accuracy_improvement": float(
            accuracy_improvement
        ),
        "log_loss_improvement": float(
            log_loss_improvement
        ),
        "all_experiments": results,
        "candidate_model_path": str(
            BEST_CANDIDATE_PATH
        ),
    }

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
    print("=" * 76)
    print("RECENCY EXPERIMENT SUMMARY")
    print("=" * 76)

    print("Production baseline:")
    print(
        f"Accuracy: "
        f"{production_result['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{production_result['log_loss']:.4f}"
    )

    print()
    print(
        "Best accuracy experiment:"
    )

    print(
        f"Name: "
        f"{best_accuracy_name}"
    )

    print(
        "Training matches: "
        f"{best_accuracy_result['training_matches']:,}"
    )

    print(
        f"Accuracy: "
        f"{best_accuracy_result['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{best_accuracy_result['log_loss']:.4f}"
    )

    print(
        f"Draw recall: "
        f"{best_accuracy_result['draw_recall']:.2%}"
    )

    print()
    print(
        "Best log-loss experiment:"
    )

    print(
        f"Name: "
        f"{best_log_loss_name}"
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
        "Accuracy improvement over "
        "production: "
        f"{accuracy_improvement:+.2%}"
    )

    print(
        "Log-loss improvement over "
        "production: "
        f"{log_loss_improvement:+.4f}"
    )

    print()
    print(
        "Best-accuracy classification "
        "report:"
    )

    print(
        best_accuracy_result[
            "classification_report"
        ]
    )

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
        f"Candidate model saved to:\n"
        f"{BEST_CANDIDATE_PATH}"
    )

    print(
        f"Report saved to:\n"
        f"{REPORT_PATH}"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
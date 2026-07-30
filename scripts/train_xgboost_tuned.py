from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


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

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "xgboost_tuned_candidate.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "xgboost_tuning_report.json"
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

LABEL_TO_INTEGER = {
    "A": 0,
    "D": 1,
    "H": 2,
}

INTEGER_TO_LABEL = {
    0: "A",
    1: "D",
    2: "H",
}


PARAMETER_SETS = [
    {
        "name": "shallow_regularized",
        "n_estimators": 250,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "gamma": 0.10,
        "reg_alpha": 0.50,
        "reg_lambda": 4.00,
    },
    {
        "name": "shallow_balanced",
        "n_estimators": 350,
        "max_depth": 2,
        "learning_rate": 0.025,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.00,
        "reg_alpha": 0.25,
        "reg_lambda": 3.00,
    },
    {
        "name": "medium_regularized",
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.025,
        "min_child_weight": 5,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "gamma": 0.15,
        "reg_alpha": 0.50,
        "reg_lambda": 5.00,
    },
    {
        "name": "medium_conservative",
        "n_estimators": 220,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 8,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "gamma": 0.20,
        "reg_alpha": 0.75,
        "reg_lambda": 6.00,
    },
    {
        "name": "deeper_regularized",
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.02,
        "min_child_weight": 8,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "gamma": 0.25,
        "reg_alpha": 1.00,
        "reg_lambda": 8.00,
    },
]


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


def chronological_development_split(
    development_data: pd.DataFrame,
    validation_fraction: float = 0.20,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    ordered = development_data.sort_values(
        [
            "date",
            "match_id",
        ]
    ).reset_index(
        drop=True
    )

    split_index = int(
        len(ordered)
        * (
            1.0
            - validation_fraction
        )
    )

    if split_index <= 0:
        raise RuntimeError(
            "Training split is empty."
        )

    if split_index >= len(ordered):
        raise RuntimeError(
            "Validation split is empty."
        )

    training_data = ordered.iloc[
        :split_index
    ].copy()

    validation_data = ordered.iloc[
        split_index:
    ].copy()

    return (
        training_data,
        validation_data,
    )


def build_pipeline(
    parameters: Dict[str, object],
) -> Pipeline:
    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",

        n_estimators=int(
            parameters["n_estimators"]
        ),
        max_depth=int(
            parameters["max_depth"]
        ),
        learning_rate=float(
            parameters["learning_rate"]
        ),
        min_child_weight=float(
            parameters["min_child_weight"]
        ),
        subsample=float(
            parameters["subsample"]
        ),
        colsample_bytree=float(
            parameters["colsample_bytree"]
        ),
        gamma=float(
            parameters["gamma"]
        ),
        reg_alpha=float(
            parameters["reg_alpha"]
        ),
        reg_lambda=float(
            parameters["reg_lambda"]
        ),
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def labels_to_integers(
    labels: pd.Series,
) -> np.ndarray:
    mapped = labels.map(
        LABEL_TO_INTEGER
    )

    if mapped.isna().any():
        invalid_labels = sorted(
            labels[
                mapped.isna()
            ].astype(str).unique()
        )

        raise ValueError(
            "Unexpected labels: "
            + ", ".join(
                invalid_labels
            )
        )

    return mapped.astype(
        int
    ).to_numpy()


def probabilities_to_labels(
    probabilities: np.ndarray,
) -> List[str]:
    best_indices = np.argmax(
        probabilities,
        axis=1,
    )

    return [
        INTEGER_TO_LABEL[
            int(index)
        ]
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

    draw_actual = (
        np.asarray(
            actual_labels
        )
        == "D"
    )

    draw_predicted = (
        np.asarray(
            predicted_labels
        )
        == "D"
    )

    correct_draws = int(
        np.sum(
            draw_actual
            & draw_predicted
        )
    )

    total_draws = int(
        np.sum(
            draw_actual
        )
    )

    predicted_draw_count = int(
        np.sum(
            draw_predicted
        )
    )

    draw_recall = (
        correct_draws
        / total_draws
        if total_draws > 0
        else 0.0
    )

    draw_precision = (
        correct_draws
        / predicted_draw_count
        if predicted_draw_count > 0
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
        "predicted_draws": (
            predicted_draw_count
        ),
        "correct_draws": (
            correct_draws
        ),
        "classification_report": (
            report_text
        ),
        "confusion_matrix": (
            matrix
        ),
    }


def ordered_production_probabilities(
    model: MachineLearningModel,
    features: pd.DataFrame,
) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(
            features
        ),
        dtype=float,
    )

    return normalize_probabilities(
        probabilities
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

    print("=" * 74)
    print("XGBOOST CHRONOLOGICAL TUNING")
    print("=" * 74)

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

    if development_data.empty:
        raise RuntimeError(
            "Development data is empty."
        )

    if test_data.empty:
        raise RuntimeError(
            "Test data is empty."
        )

    (
        tuning_training_data,
        validation_data,
    ) = chronological_development_split(
        development_data=(
            development_data
        ),
        validation_fraction=0.20,
    )

    feature_columns = list(
        FeatureBuilder.FEATURE_COLUMNS
    )

    X_tuning_train = (
        tuning_training_data[
            feature_columns
        ].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    y_tuning_train = labels_to_integers(
        tuning_training_data[
            "target"
        ].astype(str)
    )

    X_validation = validation_data[
        feature_columns
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    validation_labels = (
        validation_data[
            "target"
        ]
        .astype(str)
        .tolist()
    )

    print()
    print("DATA SPLIT")
    print("-" * 74)

    print(
        "Tuning training matches: "
        f"{len(tuning_training_data):,}"
    )

    print(
        "Validation matches: "
        f"{len(validation_data):,}"
    )

    print(
        "Final development matches: "
        f"{len(development_data):,}"
    )

    print(
        "Unseen CL test matches: "
        f"{len(test_data):,}"
    )

    search_results = []

    print()
    print("VALIDATION RESULTS")
    print("-" * 74)

    for parameters in PARAMETER_SETS:
        model_name = str(
            parameters["name"]
        )

        pipeline = build_pipeline(
            parameters
        )

        pipeline.fit(
            X_tuning_train,
            y_tuning_train,
        )

        validation_probabilities = (
            pipeline.predict_proba(
                X_validation
            )
        )

        validation_result = (
            evaluate_probabilities(
                actual_labels=(
                    validation_labels
                ),
                probabilities=np.asarray(
                    validation_probabilities,
                    dtype=float,
                ),
            )
        )

        search_result = {
            "name": model_name,
            "parameters": parameters,
            "accuracy": (
                validation_result[
                    "accuracy"
                ]
            ),
            "log_loss": (
                validation_result[
                    "log_loss"
                ]
            ),
            "draw_recall": (
                validation_result[
                    "draw_recall"
                ]
            ),
        }

        search_results.append(
            search_result
        )

        print(
            f"{model_name:<24} | "
            f"Accuracy "
            f"{validation_result['accuracy']:.2%} | "
            f"Log loss "
            f"{validation_result['log_loss']:.4f} | "
            f"Draw recall "
            f"{validation_result['draw_recall']:.2%}"
        )

    best_validation_result = max(
        search_results,
        key=lambda result: (
            result["accuracy"],
            -result["log_loss"],
            result["draw_recall"],
        ),
    )

    best_parameters = dict(
        best_validation_result[
            "parameters"
        ]
    )

    print()
    print("BEST VALIDATION MODEL")
    print("-" * 74)

    print(
        f"Name: "
        f"{best_validation_result['name']}"
    )

    print(
        "Validation accuracy: "
        f"{best_validation_result['accuracy']:.2%}"
    )

    print(
        "Validation log loss: "
        f"{best_validation_result['log_loss']:.4f}"
    )

    print(
        "Validation draw recall: "
        f"{best_validation_result['draw_recall']:.2%}"
    )

    X_development = development_data[
        feature_columns
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    y_development = labels_to_integers(
        development_data[
            "target"
        ].astype(str)
    )

    final_model = build_pipeline(
        best_parameters
    )

    final_model.fit(
        X_development,
        y_development,
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

    xgboost_probabilities = (
        final_model.predict_proba(
            X_test
        )
    )

    xgboost_result = (
        evaluate_probabilities(
            actual_labels=test_labels,
            probabilities=np.asarray(
                xgboost_probabilities,
                dtype=float,
            ),
        )
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
        ordered_production_probabilities(
            model=production_model,
            features=X_test,
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

    candidate_bundle = {
        "model_type": (
            "xgboost_multiclass"
        ),
        "model": final_model,
        "feature_columns": (
            feature_columns
        ),
        "label_order": (
            LABEL_ORDER
        ),
        "integer_to_label": (
            INTEGER_TO_LABEL
        ),
        "parameters": (
            best_parameters
        ),
        "test_start_date": (
            TEST_START_DATE
        ),
    }

    CANDIDATE_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        candidate_bundle,
        CANDIDATE_MODEL_PATH,
    )

    accuracy_difference = (
        xgboost_result[
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
        - xgboost_result[
            "log_loss"
        ]
    )

    report = {
        "test_start_date": (
            TEST_START_DATE
        ),
        "development_matches": int(
            len(development_data)
        ),
        "validation_matches": int(
            len(validation_data)
        ),
        "test_matches": int(
            len(test_data)
        ),
        "best_validation_result": (
            best_validation_result
        ),
        "production_model": {
            "accuracy": (
                production_result[
                    "accuracy"
                ]
            ),
            "log_loss": (
                production_result[
                    "log_loss"
                ]
            ),
            "draw_recall": (
                production_result[
                    "draw_recall"
                ]
            ),
            "confusion_matrix": (
                production_result[
                    "confusion_matrix"
                ].tolist()
            ),
        },
        "xgboost_model": {
            "accuracy": (
                xgboost_result[
                    "accuracy"
                ]
            ),
            "log_loss": (
                xgboost_result[
                    "log_loss"
                ]
            ),
            "draw_recall": (
                xgboost_result[
                    "draw_recall"
                ]
            ),
            "draw_precision": (
                xgboost_result[
                    "draw_precision"
                ]
            ),
            "predicted_draws": (
                xgboost_result[
                    "predicted_draws"
                ]
            ),
            "correct_draws": (
                xgboost_result[
                    "correct_draws"
                ]
            ),
            "classification_report": (
                xgboost_result[
                    "classification_report"
                ]
            ),
            "confusion_matrix": (
                xgboost_result[
                    "confusion_matrix"
                ].tolist()
            ),
        },
        "accuracy_difference": float(
            accuracy_difference
        ),
        "log_loss_improvement": float(
            log_loss_improvement
        ),
        "candidate_model_path": str(
            CANDIDATE_MODEL_PATH
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
    print("=" * 74)
    print("FINAL UNSEEN TEST RESULTS")
    print("=" * 74)

    print("Production ML model:")
    print(
        f"Accuracy: "
        f"{production_result['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{production_result['log_loss']:.4f}"
    )

    print(
        f"Draw recall: "
        f"{production_result['draw_recall']:.2%}"
    )

    print()
    print("Tuned XGBoost model:")
    print(
        f"Accuracy: "
        f"{xgboost_result['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{xgboost_result['log_loss']:.4f}"
    )

    print(
        f"Draw recall: "
        f"{xgboost_result['draw_recall']:.2%}"
    )

    print(
        f"Draw precision: "
        f"{xgboost_result['draw_precision']:.2%}"
    )

    print(
        f"Predicted draws: "
        f"{xgboost_result['predicted_draws']}"
    )

    print(
        f"Correct draws: "
        f"{xgboost_result['correct_draws']}"
    )

    print()
    print(
        "Accuracy difference: "
        f"{accuracy_difference:+.2%}"
    )

    print(
        "Log-loss improvement: "
        f"{log_loss_improvement:+.4f}"
    )

    print()
    print("Classification report:")
    print(
        xgboost_result[
            "classification_report"
        ]
    )

    print("Confusion matrix:")
    print(
        xgboost_result[
            "confusion_matrix"
        ]
    )

    print()
    print(
        f"Candidate model saved to:\n"
        f"{CANDIDATE_MODEL_PATH}"
    )

    print(
        f"Report saved to:\n"
        f"{REPORT_PATH}"
    )

    print("=" * 74)


if __name__ == "__main__":
    main()
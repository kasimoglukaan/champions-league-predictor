from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.features.feature_builder import FeatureBuilder
from src.models.machine_learning_model import MachineLearningModel


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
    / "two_stage_candidate.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "two_stage_model_report.json"
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

C_VALUES = [
    0.01,
    0.03,
    0.10,
    0.30,
    1.00,
    3.00,
    10.00,
]

DRAW_LOGIT_BIASES = [
    -1.50,
    -1.25,
    -1.00,
    -0.75,
    -0.50,
    -0.25,
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


def create_binary_model(
    c_value: float,
    class_weight=None,
) -> Pipeline:
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
                    C=c_value,
                    class_weight=class_weight,
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

    row_sums = clipped.sum(
        axis=1,
        keepdims=True,
    )

    return clipped / row_sums


def probability_for_class(
    model: Pipeline,
    features: pd.DataFrame,
    wanted_class,
) -> np.ndarray:
    probabilities = model.predict_proba(
        features
    )

    classes = list(
        model.classes_
    )

    if wanted_class not in classes:
        raise ValueError(
            f"Class {wanted_class!r} was not "
            f"found in model classes: {classes}"
        )

    class_index = classes.index(
        wanted_class
    )

    return probabilities[
        :,
        class_index,
    ]


def apply_logit_bias(
    probabilities: np.ndarray,
    bias: float,
) -> np.ndarray:
    clipped = np.clip(
        probabilities,
        1e-8,
        1.0 - 1e-8,
    )

    logits = np.log(
        clipped
        / (1.0 - clipped)
    )

    adjusted = 1.0 / (
        1.0
        + np.exp(
            -(
                logits
                + bias
            )
        )
    )

    return np.clip(
        adjusted,
        1e-8,
        1.0 - 1e-8,
    )


def create_two_stage_probabilities(
    draw_model: Pipeline,
    winner_model: Pipeline,
    features: pd.DataFrame,
    draw_logit_bias: float,
) -> np.ndarray:
    raw_draw_probability = (
        probability_for_class(
            model=draw_model,
            features=features,
            wanted_class=1,
        )
    )

    draw_probability = apply_logit_bias(
        probabilities=raw_draw_probability,
        bias=draw_logit_bias,
    )

    home_given_not_draw = (
        probability_for_class(
            model=winner_model,
            features=features,
            wanted_class="H",
        )
    )

    away_given_not_draw = (
        probability_for_class(
            model=winner_model,
            features=features,
            wanted_class="A",
        )
    )

    winner_total = (
        home_given_not_draw
        + away_given_not_draw
    )

    winner_total = np.where(
        winner_total <= 0.0,
        1.0,
        winner_total,
    )

    home_given_not_draw = (
        home_given_not_draw
        / winner_total
    )

    away_given_not_draw = (
        away_given_not_draw
        / winner_total
    )

    not_draw_probability = (
        1.0
        - draw_probability
    )

    away_probability = (
        not_draw_probability
        * away_given_not_draw
    )

    home_probability = (
        not_draw_probability
        * home_given_not_draw
    )

    probabilities = np.column_stack(
        [
            away_probability,
            draw_probability,
            home_probability,
        ]
    )

    return normalize_probabilities(
        probabilities
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
    actual_labels: Sequence[str],
    probabilities: np.ndarray,
) -> Dict[str, object]:
    normalized = normalize_probabilities(
        probabilities
    )

    predictions = probabilities_to_labels(
        normalized
    )

    accuracy = accuracy_score(
        actual_labels,
        predictions,
    )

    result_log_loss = log_loss(
        actual_labels,
        normalized,
        labels=LABEL_ORDER,
    )

    report_text = classification_report(
        actual_labels,
        predictions,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual_labels,
        predictions,
        labels=LABEL_ORDER,
    )

    draw_actual = np.asarray(
        actual_labels
    ) == "D"

    draw_predicted = np.asarray(
        predictions
    ) == "D"

    true_draws = int(
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

    predicted_draws = int(
        np.sum(
            draw_predicted
        )
    )

    draw_recall = (
        true_draws / total_draws
        if total_draws > 0
        else 0.0
    )

    draw_precision = (
        true_draws / predicted_draws
        if predicted_draws > 0
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "log_loss": float(
            result_log_loss
        ),
        "draw_recall": float(
            draw_recall
        ),
        "draw_precision": float(
            draw_precision
        ),
        "predicted_draws": (
            predicted_draws
        ),
        "correct_draws": (
            true_draws
        ),
        "classification_report": (
            report_text
        ),
        "confusion_matrix": (
            matrix
        ),
        "predictions": (
            predictions
        ),
    }


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


def ordered_production_probabilities(
    model: MachineLearningModel,
    features: pd.DataFrame,
) -> np.ndarray:
    raw_probabilities = np.asarray(
        model.predict_proba(
            features
        ),
        dtype=float,
    )

    # Existing production model uses A, D, H order.
    return normalize_probabilities(
        raw_probabilities
    )


def fit_two_stage_models(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    draw_c: float,
    winner_c: float,
) -> Tuple[
    Pipeline,
    Pipeline,
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

    draw_targets = (
        training_data[
            "target"
        ]
        .eq("D")
        .astype(int)
    )

    draw_model = create_binary_model(
        c_value=draw_c,
        class_weight=None,
    )

    draw_model.fit(
        X_train,
        draw_targets,
    )

    winner_mask = (
        training_data[
            "target"
        ]
        .isin(
            [
                "A",
                "H",
            ]
        )
    )

    winner_training = training_data[
        winner_mask
    ].copy()

    X_winner = winner_training[
        feature_columns
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    winner_targets = (
        winner_training[
            "target"
        ]
        .astype(str)
    )

    winner_model = create_binary_model(
        c_value=winner_c,
        class_weight=None,
    )

    winner_model.fit(
        X_winner,
        winner_targets,
    )

    return (
        draw_model,
        winner_model,
    )


def tune_two_stage_model(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[str, object]:
    X_validation = validation_data[
        feature_columns
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    validation_targets = (
        validation_data[
            "target"
        ]
        .astype(str)
        .tolist()
    )

    best_result = None

    print()
    print("=" * 74)
    print("TWO-STAGE VALIDATION SEARCH")
    print("=" * 74)

    for draw_c in C_VALUES:
        for winner_c in C_VALUES:
            (
                draw_model,
                winner_model,
            ) = fit_two_stage_models(
                training_data=training_data,
                feature_columns=(
                    feature_columns
                ),
                draw_c=draw_c,
                winner_c=winner_c,
            )

            for draw_bias in DRAW_LOGIT_BIASES:
                probabilities = (
                    create_two_stage_probabilities(
                        draw_model=draw_model,
                        winner_model=winner_model,
                        features=X_validation,
                        draw_logit_bias=(
                            draw_bias
                        ),
                    )
                )

                result = evaluate_probabilities(
                    actual_labels=(
                        validation_targets
                    ),
                    probabilities=(
                        probabilities
                    ),
                )

                candidate = {
                    "draw_c": float(
                        draw_c
                    ),
                    "winner_c": float(
                        winner_c
                    ),
                    "draw_logit_bias": float(
                        draw_bias
                    ),
                    "accuracy": (
                        result[
                            "accuracy"
                        ]
                    ),
                    "log_loss": (
                        result[
                            "log_loss"
                        ]
                    ),
                    "draw_recall": (
                        result[
                            "draw_recall"
                        ]
                    ),
                    "draw_precision": (
                        result[
                            "draw_precision"
                        ]
                    ),
                    "predicted_draws": (
                        result[
                            "predicted_draws"
                        ]
                    ),
                }

                if best_result is None:
                    best_result = candidate
                    continue

                candidate_score = (
                    candidate[
                        "accuracy"
                    ],
                    -candidate[
                        "log_loss"
                    ],
                    candidate[
                        "draw_recall"
                    ],
                )

                best_score = (
                    best_result[
                        "accuracy"
                    ],
                    -best_result[
                        "log_loss"
                    ],
                    best_result[
                        "draw_recall"
                    ],
                )

                if candidate_score > best_score:
                    best_result = candidate

    if best_result is None:
        raise RuntimeError(
            "No valid parameter combination "
            "was evaluated."
        )

    print(
        f"Best draw C: "
        f"{best_result['draw_c']}"
    )

    print(
        f"Best winner C: "
        f"{best_result['winner_c']}"
    )

    print(
        "Best draw logit bias: "
        f"{best_result['draw_logit_bias']:+.2f}"
    )

    print(
        "Validation accuracy: "
        f"{best_result['accuracy']:.2%}"
    )

    print(
        "Validation log loss: "
        f"{best_result['log_loss']:.4f}"
    )

    print(
        "Validation draw recall: "
        f"{best_result['draw_recall']:.2%}"
    )

    print(
        "Validation draw precision: "
        f"{best_result['draw_precision']:.2%}"
    )

    return best_result


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
    print("TWO-STAGE DRAW MODEL EXPERIMENT")
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
        tuning_train_data,
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

    print()
    print("DATA SPLIT")
    print("-" * 74)

    print(
        "Tuning training matches: "
        f"{len(tuning_train_data):,}"
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

    print(
        "Tuning period: "
        f"{tuning_train_data['date'].min()} "
        "to "
        f"{tuning_train_data['date'].max()}"
    )

    print(
        "Validation period: "
        f"{validation_data['date'].min()} "
        "to "
        f"{validation_data['date'].max()}"
    )

    print(
        "Test period: "
        f"{test_data['date'].min()} "
        "to "
        f"{test_data['date'].max()}"
    )

    best_parameters = tune_two_stage_model(
        training_data=(
            tuning_train_data
        ),
        validation_data=(
            validation_data
        ),
        feature_columns=(
            feature_columns
        ),
    )

    (
        final_draw_model,
        final_winner_model,
    ) = fit_two_stage_models(
        training_data=(
            development_data
        ),
        feature_columns=(
            feature_columns
        ),
        draw_c=float(
            best_parameters[
                "draw_c"
            ]
        ),
        winner_c=float(
            best_parameters[
                "winner_c"
            ]
        ),
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

    test_targets = (
        test_data[
            "target"
        ]
        .astype(str)
        .tolist()
    )

    two_stage_probabilities = (
        create_two_stage_probabilities(
            draw_model=(
                final_draw_model
            ),
            winner_model=(
                final_winner_model
            ),
            features=X_test,
            draw_logit_bias=float(
                best_parameters[
                    "draw_logit_bias"
                ]
            ),
        )
    )

    two_stage_result = (
        evaluate_probabilities(
            actual_labels=test_targets,
            probabilities=(
                two_stage_probabilities
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
            actual_labels=test_targets,
            probabilities=(
                production_probabilities
            ),
        )
    )

    candidate_bundle = {
        "model_type": (
            "two_stage_draw_model"
        ),
        "feature_columns": (
            feature_columns
        ),
        "draw_model": (
            final_draw_model
        ),
        "winner_model": (
            final_winner_model
        ),
        "draw_logit_bias": float(
            best_parameters[
                "draw_logit_bias"
            ]
        ),
        "draw_c": float(
            best_parameters[
                "draw_c"
            ]
        ),
        "winner_c": float(
            best_parameters[
                "winner_c"
            ]
        ),
        "label_order": (
            LABEL_ORDER
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
        two_stage_result[
            "accuracy"
        ]
        - production_result[
            "accuracy"
        ]
    )

    log_loss_difference = (
        production_result[
            "log_loss"
        ]
        - two_stage_result[
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
        "best_validation_parameters": (
            best_parameters
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
            "draw_precision": (
                production_result[
                    "draw_precision"
                ]
            ),
            "predicted_draws": (
                production_result[
                    "predicted_draws"
                ]
            ),
            "correct_draws": (
                production_result[
                    "correct_draws"
                ]
            ),
            "confusion_matrix": (
                production_result[
                    "confusion_matrix"
                ].tolist()
            ),
        },
        "two_stage_model": {
            "accuracy": (
                two_stage_result[
                    "accuracy"
                ]
            ),
            "log_loss": (
                two_stage_result[
                    "log_loss"
                ]
            ),
            "draw_recall": (
                two_stage_result[
                    "draw_recall"
                ]
            ),
            "draw_precision": (
                two_stage_result[
                    "draw_precision"
                ]
            ),
            "predicted_draws": (
                two_stage_result[
                    "predicted_draws"
                ]
            ),
            "correct_draws": (
                two_stage_result[
                    "correct_draws"
                ]
            ),
            "classification_report": (
                two_stage_result[
                    "classification_report"
                ]
            ),
            "confusion_matrix": (
                two_stage_result[
                    "confusion_matrix"
                ].tolist()
            ),
        },
        "accuracy_difference": float(
            accuracy_difference
        ),
        "log_loss_improvement": float(
            log_loss_difference
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

    print(
        f"Draw precision: "
        f"{production_result['draw_precision']:.2%}"
    )

    print(
        f"Predicted draws: "
        f"{production_result['predicted_draws']}"
    )

    print()
    print("Two-stage model:")
    print(
        f"Accuracy: "
        f"{two_stage_result['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{two_stage_result['log_loss']:.4f}"
    )

    print(
        f"Draw recall: "
        f"{two_stage_result['draw_recall']:.2%}"
    )

    print(
        f"Draw precision: "
        f"{two_stage_result['draw_precision']:.2%}"
    )

    print(
        f"Predicted draws: "
        f"{two_stage_result['predicted_draws']}"
    )

    print(
        f"Correct draws: "
        f"{two_stage_result['correct_draws']}"
    )

    print()
    print(
        "Accuracy difference: "
        f"{accuracy_difference:+.2%}"
    )

    print(
        "Log-loss improvement: "
        f"{log_loss_difference:+.4f}"
    )

    print()
    print("Classification report:")
    print(
        two_stage_result[
            "classification_report"
        ]
    )

    print("Confusion matrix:")
    print(
        two_stage_result[
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
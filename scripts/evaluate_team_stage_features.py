from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
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
    / "team_stage_candidate.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "team_stage_feature_report.json"
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
        "name": "numeric_baseline",
        "categorical_columns": [],
        "extra_numeric_columns": [],
    },
    {
        "name": "stage_only",
        "categorical_columns": [
            "stage",
        ],
        "extra_numeric_columns": [],
    },
    {
        "name": "stage_matchday",
        "categorical_columns": [
            "stage",
        ],
        "extra_numeric_columns": [
            "matchday",
        ],
    },
    {
        "name": "team_identity",
        "categorical_columns": [
            "home_team",
            "away_team",
        ],
        "extra_numeric_columns": [],
    },
    {
        "name": "team_stage",
        "categorical_columns": [
            "home_team",
            "away_team",
            "stage",
        ],
        "extra_numeric_columns": [],
    },
    {
        "name": "team_stage_matchday",
        "categorical_columns": [
            "home_team",
            "away_team",
            "stage",
        ],
        "extra_numeric_columns": [
            "matchday",
        ],
    },
]


C_VALUES = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.10,
    0.30,
    1.00,
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

        index = classes.index(
            label
        )

        ordered_columns.append(
            probabilities[
                :,
                index,
            ]
        )

    return normalize_probabilities(
        np.column_stack(
            ordered_columns
        )
    )


def probabilities_to_labels(
    probabilities: np.ndarray,
) -> List[str]:
    indices = np.argmax(
        probabilities,
        axis=1,
    )

    return [
        LABEL_ORDER[index]
        for index in indices
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

    actual_draws = (
        actual_array == "D"
    )

    predicted_draws_mask = (
        predicted_array == "D"
    )

    correct_draws = int(
        np.sum(
            actual_draws
            & predicted_draws_mask
        )
    )

    total_draws = int(
        np.sum(
            actual_draws
        )
    )

    predicted_draws = int(
        np.sum(
            predicted_draws_mask
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
        "predicted_draws": (
            predicted_draws
        ),
        "correct_draws": (
            correct_draws
        ),
        "classification_report": (
            report_text
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
    }


def chronological_split(
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

    return (
        ordered.iloc[
            :split_index
        ].copy(),
        ordered.iloc[
            split_index:
        ].copy(),
    )


def ensure_metadata_columns(
    dataset: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        "match_id",
        "home_team",
        "away_team",
        "stage",
        "matchday",
        "season",
    ]

    missing_metadata = [
        column
        for column in metadata_columns
        if column not in dataset.columns
    ]

    if not missing_metadata:
        return dataset.copy()

    metadata = matches[
        metadata_columns
    ].drop_duplicates(
        subset=[
            "match_id",
        ]
    )

    columns_to_merge = [
        "match_id",
        *missing_metadata,
    ]

    return dataset.merge(
        metadata[
            columns_to_merge
        ],
        on="match_id",
        how="left",
        validate="one_to_one",
    )


def clean_metadata(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    cleaned = dataset.copy()

    cleaned["home_team"] = (
        cleaned["home_team"]
        .fillna("UNKNOWN_HOME")
        .astype(str)
    )

    cleaned["away_team"] = (
        cleaned["away_team"]
        .fillna("UNKNOWN_AWAY")
        .astype(str)
    )

    cleaned["stage"] = (
        cleaned["stage"]
        .fillna("UNKNOWN_STAGE")
        .astype(str)
    )

    cleaned["matchday"] = pd.to_numeric(
        cleaned["matchday"],
        errors="coerce",
    )

    return cleaned


def create_model(
    numeric_columns: List[str],
    categorical_columns: List[str],
    c_value: float,
) -> Pipeline:
    numeric_pipeline = Pipeline(
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
        ]
    )

    transformers = [
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
    ]

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    classifier = LogisticRegression(
        C=c_value,
        solver="saga",
        penalty="l2",
        max_iter=10000,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def fit_and_evaluate(
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
    c_value: float,
) -> Tuple[
    Pipeline,
    Dict[str, object],
    np.ndarray,
]:
    selected_columns = [
        *numeric_columns,
        *categorical_columns,
    ]

    X_train = training_data[
        selected_columns
    ].copy()

    y_train = (
        training_data[
            "target"
        ]
        .astype(str)
    )

    X_evaluation = evaluation_data[
        selected_columns
    ].copy()

    evaluation_labels = (
        evaluation_data[
            "target"
        ]
        .astype(str)
        .tolist()
    )

    model = create_model(
        numeric_columns=numeric_columns,
        categorical_columns=(
            categorical_columns
        ),
        c_value=c_value,
    )

    model.fit(
        X_train,
        y_train,
    )

    raw_probabilities = (
        model.predict_proba(
            X_evaluation
        )
    )

    probabilities = reorder_probabilities(
        model=model,
        probabilities=np.asarray(
            raw_probabilities,
            dtype=float,
        ),
    )

    result = evaluate_probabilities(
        actual_labels=(
            evaluation_labels
        ),
        probabilities=probabilities,
    )

    return (
        model,
        result,
        probabilities,
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

    print("=" * 78)
    print(
        "TEAM IDENTITY, STAGE AND MATCHDAY "
        "FEATURE EXPERIMENT"
    )
    print("=" * 78)

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

    dataset = ensure_metadata_columns(
        dataset=dataset,
        matches=matches,
    )

    dataset = clean_metadata(
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

    test_data = (
        test_data.sort_values(
            [
                "date",
                "match_id",
            ]
        ).reset_index(
            drop=True
        )
    )

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
    ) = chronological_split(
        development_data=(
            development_data
        ),
        validation_fraction=0.20,
    )

    base_numeric_columns = list(
        FeatureBuilder.FEATURE_COLUMNS
    )

    print()
    print("DATA SPLIT")
    print("-" * 78)

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

    validation_results: Dict[
        str,
        Dict[str, object],
    ] = {}

    print()
    print("VALIDATION SEARCH")
    print("-" * 78)

    for experiment in EXPERIMENTS:
        experiment_name = str(
            experiment["name"]
        )

        categorical_columns = list(
            experiment[
                "categorical_columns"
            ]
        )

        extra_numeric_columns = list(
            experiment[
                "extra_numeric_columns"
            ]
        )

        numeric_columns = [
            *base_numeric_columns,
            *extra_numeric_columns,
        ]

        best_validation = None

        for c_value in C_VALUES:
            (
                _,
                result,
                _,
            ) = fit_and_evaluate(
                training_data=(
                    tuning_training_data
                ),
                evaluation_data=(
                    validation_data
                ),
                numeric_columns=(
                    numeric_columns
                ),
                categorical_columns=(
                    categorical_columns
                ),
                c_value=c_value,
            )

            candidate = {
                "experiment_name": (
                    experiment_name
                ),
                "c_value": float(
                    c_value
                ),
                "numeric_columns": (
                    numeric_columns
                ),
                "categorical_columns": (
                    categorical_columns
                ),
                **result,
            }

            if best_validation is None:
                best_validation = candidate

            else:
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
                    best_validation[
                        "accuracy"
                    ],
                    -best_validation[
                        "log_loss"
                    ],
                    best_validation[
                        "draw_recall"
                    ],
                )

                if candidate_score > best_score:
                    best_validation = candidate

        if best_validation is None:
            raise RuntimeError(
                f"No model evaluated for "
                f"{experiment_name}."
            )

        validation_results[
            experiment_name
        ] = best_validation

        print(
            f"{experiment_name:<25} | "
            f"C {best_validation['c_value']:<5} | "
            f"Accuracy "
            f"{best_validation['accuracy']:.2%} | "
            f"Log loss "
            f"{best_validation['log_loss']:.4f} | "
            f"Draw recall "
            f"{best_validation['draw_recall']:.2%}"
        )

    best_experiment_name = max(
        validation_results,
        key=lambda name: (
            validation_results[
                name
            ]["accuracy"],
            -validation_results[
                name
            ]["log_loss"],
            validation_results[
                name
            ]["draw_recall"],
        ),
    )

    best_validation_result = (
        validation_results[
            best_experiment_name
        ]
    )

    print()
    print("SELECTED VALIDATION MODEL")
    print("-" * 78)

    print(
        f"Experiment: "
        f"{best_experiment_name}"
    )

    print(
        f"C value: "
        f"{best_validation_result['c_value']}"
    )

    print(
        "Validation accuracy: "
        f"{best_validation_result['accuracy']:.2%}"
    )

    print(
        "Validation log loss: "
        f"{best_validation_result['log_loss']:.4f}"
    )

    (
        final_candidate_model,
        candidate_test_result,
        candidate_probabilities,
    ) = fit_and_evaluate(
        training_data=(
            development_data
        ),
        evaluation_data=(
            test_data
        ),
        numeric_columns=list(
            best_validation_result[
                "numeric_columns"
            ]
        ),
        categorical_columns=list(
            best_validation_result[
                "categorical_columns"
            ]
        ),
        c_value=float(
            best_validation_result[
                "c_value"
            ]
        ),
    )

    base_test_features = test_data[
        base_numeric_columns
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
                    base_test_features
                ),
                dtype=float,
            )
        )
    )

    production_result = (
        evaluate_probabilities(
            actual_labels=(
                test_labels
            ),
            probabilities=(
                production_probabilities
            ),
        )
    )

    hybrid_probabilities = (
        0.50
        * production_probabilities
        + 0.50
        * candidate_probabilities
    )

    hybrid_result = evaluate_probabilities(
        actual_labels=test_labels,
        probabilities=(
            hybrid_probabilities
        ),
    )

    candidate_bundle = {
        "model_type": (
            "team_stage_logistic"
        ),
        "experiment_name": (
            best_experiment_name
        ),
        "model": (
            final_candidate_model
        ),
        "numeric_columns": list(
            best_validation_result[
                "numeric_columns"
            ]
        ),
        "categorical_columns": list(
            best_validation_result[
                "categorical_columns"
            ]
        ),
        "label_order": (
            LABEL_ORDER
        ),
        "c_value": float(
            best_validation_result[
                "c_value"
            ]
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

    candidate_accuracy_difference = (
        candidate_test_result[
            "accuracy"
        ]
        - production_result[
            "accuracy"
        ]
    )

    hybrid_accuracy_difference = (
        hybrid_result[
            "accuracy"
        ]
        - production_result[
            "accuracy"
        ]
    )

    report = {
        "test_start_date": (
            TEST_START_DATE
        ),
        "test_matches": int(
            len(test_data)
        ),
        "validation_results": (
            validation_results
        ),
        "selected_experiment": (
            best_experiment_name
        ),
        "production_model": (
            production_result
        ),
        "candidate_model": (
            candidate_test_result
        ),
        "production_candidate_50_50": (
            hybrid_result
        ),
        "candidate_accuracy_difference": float(
            candidate_accuracy_difference
        ),
        "hybrid_accuracy_difference": float(
            hybrid_accuracy_difference
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
    print("=" * 78)
    print("FINAL UNSEEN TEST RESULTS")
    print("=" * 78)

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
    print("Selected candidate:")
    print(
        f"Experiment: "
        f"{best_experiment_name}"
    )

    print(
        f"Accuracy: "
        f"{candidate_test_result['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{candidate_test_result['log_loss']:.4f}"
    )

    print(
        f"Draw recall: "
        f"{candidate_test_result['draw_recall']:.2%}"
    )

    print(
        "Accuracy difference: "
        f"{candidate_accuracy_difference:+.2%}"
    )

    print()
    print(
        "Production + candidate "
        "50/50 ensemble:"
    )

    print(
        f"Accuracy: "
        f"{hybrid_result['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{hybrid_result['log_loss']:.4f}"
    )

    print(
        f"Draw recall: "
        f"{hybrid_result['draw_recall']:.2%}"
    )

    print(
        "Accuracy difference: "
        f"{hybrid_accuracy_difference:+.2%}"
    )

    print()
    print(
        "Candidate classification report:"
    )

    print(
        candidate_test_result[
            "classification_report"
        ]
    )

    print(
        "Candidate confusion matrix:"
    )

    print(
        np.asarray(
            candidate_test_result[
                "confusion_matrix"
            ]
        )
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

    print("=" * 78)


if __name__ == "__main__":
    main()
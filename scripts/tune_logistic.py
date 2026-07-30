import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
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

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_tuning_data.csv"
)

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "match_model.joblib"
)

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "tuned_logistic_candidate.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "logistic_tuning_report.json"
)


VALIDATION_START_DATE = "2024-02-01"
TEST_START_DATE = "2024-07-01"

LABEL_ORDER = ["A", "D", "H"]

BASELINE_ACCURACY = 0.5767
BASELINE_LOG_LOSS = 1.0054

MIN_ACCURACY_GAIN = 0.001
MIN_LOG_LOSS_GAIN = 0.001


C_VALUES = [
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
]

CLASS_WEIGHTS: List[
    Optional[Union[str, Dict[str, float]]]
] = [
    None,
    "balanced",
    {
        "A": 1.0,
        "D": 1.10,
        "H": 1.0,
    },
    {
        "A": 1.0,
        "D": 1.25,
        "H": 1.0,
    },
]


def create_model(
    c_value: float,
    class_weight,
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
) -> Dict[str, float]:
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
    }


def class_weight_name(
    class_weight,
) -> str:
    if class_weight is None:
        return "none"

    if class_weight == "balanced":
        return "balanced"

    return json.dumps(
        class_weight,
        sort_keys=True,
    )


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Match data was not found. "
            "Run python scripts/download_data.py first."
        )

    print("=" * 65)
    print("LOGISTIC REGRESSION TUNING")
    print("=" * 65)

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

    dataset = feature_builder.build(matches)

    dataset["date"] = pd.to_datetime(
        dataset["date"],
        utc=True,
        errors="raise",
    )

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    validation_start = pd.Timestamp(
        VALIDATION_START_DATE,
        tz="UTC",
    )

    test_start = pd.Timestamp(
        TEST_START_DATE,
        tz="UTC",
    )

    train_data = dataset[
        dataset["date"] < validation_start
    ].copy()

    validation_data = dataset[
        (
            dataset["date"]
            >= validation_start
        )
        & (
            dataset["date"]
            < test_start
        )
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

    if validation_data.empty:
        raise RuntimeError(
            "Validation data is empty."
        )

    if test_data.empty:
        raise RuntimeError(
            "Champions League test data is empty."
        )

    feature_columns = (
        FeatureBuilder.FEATURE_COLUMNS
    )

    X_train = train_data[
        feature_columns
    ].copy()

    y_train = train_data[
        "target"
    ].copy()

    X_validation = validation_data[
        feature_columns
    ].copy()

    y_validation = validation_data[
        "target"
    ].copy()

    X_test = test_data[
        feature_columns
    ].copy()

    y_test = test_data[
        "target"
    ].copy()

    print()
    print("Chronological split")
    print("-" * 65)

    print(
        f"Training matches: "
        f"{len(train_data):,}"
    )

    print(
        f"Validation matches: "
        f"{len(validation_data):,}"
    )

    print(
        f"Test CL matches: "
        f"{len(test_data):,}"
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    tuning_results = []

    best_model = None
    best_parameters = None
    best_validation_result = None

    total_combinations = (
        len(C_VALUES)
        * len(CLASS_WEIGHTS)
    )

    combination_number = 0

    print()
    print(
        f"Testing {total_combinations} "
        f"parameter combinations..."
    )

    for c_value in C_VALUES:
        for class_weight in CLASS_WEIGHTS:
            combination_number += 1

            model = create_model(
                c_value=c_value,
                class_weight=class_weight,
            )

            try:
                model.fit(
                    X_train,
                    y_train,
                )

                validation_result = (
                    evaluate_model(
                        model=model,
                        features=X_validation,
                        targets=y_validation,
                    )
                )

            except Exception as error:
                print(
                    f"[{combination_number}/"
                    f"{total_combinations}] "
                    f"FAILED: C={c_value}, "
                    f"weight="
                    f"{class_weight_name(class_weight)}"
                )

                print(error)
                continue

            result_row = {
                "C": c_value,
                "class_weight": (
                    class_weight_name(
                        class_weight
                    )
                ),
                "validation_accuracy": (
                    validation_result[
                        "accuracy"
                    ]
                ),
                "validation_log_loss": (
                    validation_result[
                        "log_loss"
                    ]
                ),
            }

            tuning_results.append(
                result_row
            )

            print(
                f"[{combination_number}/"
                f"{total_combinations}] "
                f"C={c_value:<5} "
                f"weight="
                f"{class_weight_name(class_weight):<35} "
                f"accuracy="
                f"{validation_result['accuracy']:.2%} "
                f"log_loss="
                f"{validation_result['log_loss']:.4f}"
            )

            candidate_key = (
                validation_result[
                    "log_loss"
                ],
                -validation_result[
                    "accuracy"
                ],
            )

            if best_validation_result is None:
                should_select = True

            else:
                current_best_key = (
                    best_validation_result[
                        "log_loss"
                    ],
                    -best_validation_result[
                        "accuracy"
                    ],
                )

                should_select = (
                    candidate_key
                    < current_best_key
                )

            if should_select:
                best_model = model
                best_parameters = {
                    "C": c_value,
                    "class_weight": (
                        class_weight
                    ),
                }

                best_validation_result = (
                    validation_result
                )

    if best_model is None:
        raise RuntimeError(
            "All parameter combinations failed."
        )

    print()
    print("=" * 65)
    print("BEST VALIDATION PARAMETERS")
    print("=" * 65)

    print(
        f"C: {best_parameters['C']}"
    )

    print(
        "Class weight: "
        f"{class_weight_name(best_parameters['class_weight'])}"
    )

    print(
        "Validation accuracy: "
        f"{best_validation_result['accuracy']:.2%}"
    )

    print(
        "Validation log loss: "
        f"{best_validation_result['log_loss']:.4f}"
    )

    # Train again using both training and validation data.
    development_data = pd.concat(
        [
            train_data,
            validation_data,
        ],
        ignore_index=True,
    )

    X_development = development_data[
        feature_columns
    ].copy()

    y_development = development_data[
        "target"
    ].copy()

    final_model = create_model(
        c_value=best_parameters["C"],
        class_weight=best_parameters[
            "class_weight"
        ],
    )

    final_model.fit(
        X_development,
        y_development,
    )

    test_result = evaluate_model(
        model=final_model,
        features=X_test,
        targets=y_test,
    )

    CANDIDATE_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model_name": (
                "tuned_logistic_regression"
            ),
            "model": final_model,
            "label_order": LABEL_ORDER,
            "parameters": {
                "C": best_parameters["C"],
                "class_weight": (
                    class_weight_name(
                        best_parameters[
                            "class_weight"
                        ]
                    )
                ),
            },
        },
        CANDIDATE_MODEL_PATH,
    )

    accuracy_improved = (
        test_result["accuracy"]
        >= BASELINE_ACCURACY
        + MIN_ACCURACY_GAIN
    )

    log_loss_improved = (
        test_result["log_loss"]
        <= BASELINE_LOG_LOSS
        - MIN_LOG_LOSS_GAIN
    )

    promoted = (
        accuracy_improved
        and log_loss_improved
    )

    if promoted:
        shutil.copy2(
            CANDIDATE_MODEL_PATH,
            PRODUCTION_MODEL_PATH,
        )

    report = {
        "split": {
            "validation_start": (
                VALIDATION_START_DATE
            ),
            "test_start": (
                TEST_START_DATE
            ),
            "training_matches": int(
                len(train_data)
            ),
            "validation_matches": int(
                len(validation_data)
            ),
            "test_matches": int(
                len(test_data)
            ),
        },
        "baseline": {
            "accuracy": (
                BASELINE_ACCURACY
            ),
            "log_loss": (
                BASELINE_LOG_LOSS
            ),
        },
        "best_parameters": {
            "C": best_parameters["C"],
            "class_weight": (
                class_weight_name(
                    best_parameters[
                        "class_weight"
                    ]
                )
            ),
        },
        "validation": (
            best_validation_result
        ),
        "test": test_result,
        "promoted_to_production": promoted,
        "all_results": tuning_results,
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
    print("=" * 65)
    print("FINAL UNSEEN TEST RESULT")
    print("=" * 65)

    print(
        f"Test accuracy: "
        f"{test_result['accuracy']:.2%}"
    )

    print(
        f"Test log loss: "
        f"{test_result['log_loss']:.4f}"
    )

    print(
        f"Baseline accuracy: "
        f"{BASELINE_ACCURACY:.2%}"
    )

    print(
        f"Baseline log loss: "
        f"{BASELINE_LOG_LOSS:.4f}"
    )

    if promoted:
        print()
        print(
            "The tuned model beat the baseline "
            "and was promoted to production."
        )

    else:
        print()
        print(
            "The tuned model did not beat both "
            "baseline requirements."
        )

        print(
            "The production model was not changed."
        )

    print()
    print(
        f"Candidate model:\n"
        f"{CANDIDATE_MODEL_PATH}"
    )

    print(
        f"Tuning report:\n"
        f"{REPORT_PATH}"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


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

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "match_model.joblib"
)

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "xgboost_candidate.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "xgboost_report.json"
)

TEST_START_DATE = "2024-07-01"

LABEL_ORDER = ["A", "D", "H"]

LABEL_TO_NUMBER = {
    "A": 0,
    "D": 1,
    "H": 2,
}

NUMBER_TO_LABEL = {
    0: "A",
    1: "D",
    2: "H",
}

BASELINE_ACCURACY = 0.5767
BASELINE_LOG_LOSS = 1.0054

MIN_ACCURACY_GAIN = 0.001
MIN_LOG_LOSS_GAIN = 0.001


PARAMETER_SETS = [
    {
        "name": "xgb_small",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    {
        "name": "xgb_medium",
        "n_estimators": 500,
        "learning_rate": 0.025,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 3.0,
    },
    {
        "name": "xgb_regularized",
        "n_estimators": 400,
        "learning_rate": 0.02,
        "max_depth": 3,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
]


def create_model(parameters: Dict) -> Pipeline:
    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=parameters[
            "n_estimators"
        ],
        learning_rate=parameters[
            "learning_rate"
        ],
        max_depth=parameters[
            "max_depth"
        ],
        min_child_weight=parameters[
            "min_child_weight"
        ],
        subsample=parameters[
            "subsample"
        ],
        colsample_bytree=parameters[
            "colsample_bytree"
        ],
        reg_alpha=parameters[
            "reg_alpha"
        ],
        reg_lambda=parameters[
            "reg_lambda"
        ],
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
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


def decode_predictions(
    predictions: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            NUMBER_TO_LABEL[int(value)]
            for value in predictions
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    numeric_predictions = model.predict(
        X_test
    )

    predictions = decode_predictions(
        numeric_predictions
    )

    probabilities = model.predict_proba(
        X_test
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "log_loss": float(
            log_loss(
                y_test,
                probabilities,
                labels=LABEL_ORDER,
            )
        ),
        "classification_report": (
            classification_report(
                y_test,
                predictions,
                labels=LABEL_ORDER,
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_test,
                predictions,
                labels=LABEL_ORDER,
            )
        ),
    }


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Run python scripts/download_data.py first."
        )

    print("=" * 65)
    print("XGBOOST MODEL COMPARISON")
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

    dataset = feature_builder.build(
        matches
    )

    dataset["date"] = pd.to_datetime(
        dataset["date"],
        utc=True,
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

    feature_columns = (
        FeatureBuilder.FEATURE_COLUMNS
    )

    X_train = train_data[
        feature_columns
    ].copy()

    y_train = train_data[
        "target"
    ].map(
        LABEL_TO_NUMBER
    )

    X_test = test_data[
        feature_columns
    ].copy()

    y_test = test_data[
        "target"
    ].copy()

    if y_train.isna().any():
        raise ValueError(
            "Unknown class found in training target."
        )

    print(
        f"Training matches: {len(train_data):,}"
    )

    print(
        f"Test CL matches: {len(test_data):,}"
    )

    results = []
    best_model = None
    best_parameters = None
    best_result = None

    for parameters in PARAMETER_SETS:
        model_name = parameters["name"]

        print()
        print(
            f"Training {model_name}..."
        )

        model = create_model(
            parameters
        )

        model.fit(
            X_train,
            y_train,
        )

        result = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

        print(
            f"Accuracy: "
            f"{result['accuracy']:.2%}"
        )

        print(
            f"Log loss: "
            f"{result['log_loss']:.4f}"
        )

        results.append(
            {
                "name": model_name,
                "accuracy": result[
                    "accuracy"
                ],
                "log_loss": result[
                    "log_loss"
                ],
                "parameters": parameters,
            }
        )

        candidate_key = (
            result["log_loss"],
            -result["accuracy"],
        )

        if best_result is None:
            should_select = True
        else:
            best_key = (
                best_result["log_loss"],
                -best_result["accuracy"],
            )

            should_select = (
                candidate_key < best_key
            )

        if should_select:
            best_model = model
            best_parameters = parameters
            best_result = result

    if best_model is None:
        raise RuntimeError(
            "No XGBoost model was trained."
        )

    CANDIDATE_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model_name": "xgboost",
            "model": best_model,
            "label_order": LABEL_ORDER,
            "numeric_classes": True,
            "parameters": best_parameters,
        },
        CANDIDATE_MODEL_PATH,
    )

    accuracy_improved = (
        best_result["accuracy"]
        >= BASELINE_ACCURACY
        + MIN_ACCURACY_GAIN
    )

    log_loss_improved = (
        best_result["log_loss"]
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
        "baseline": {
            "accuracy": (
                BASELINE_ACCURACY
            ),
            "log_loss": (
                BASELINE_LOG_LOSS
            ),
        },
        "best_model": {
            "name": best_parameters[
                "name"
            ],
            "accuracy": best_result[
                "accuracy"
            ],
            "log_loss": best_result[
                "log_loss"
            ],
            "parameters": best_parameters,
        },
        "promoted_to_production": (
            promoted
        ),
        "all_results": results,
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
    print("BEST XGBOOST RESULT")
    print("=" * 65)

    print(
        f"Model: "
        f"{best_parameters['name']}"
    )

    print(
        f"Accuracy: "
        f"{best_result['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{best_result['log_loss']:.4f}"
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
        print(
            "XGBoost beat the baseline "
            "and was promoted."
        )
    else:
        print(
            "XGBoost did not beat both "
            "baseline requirements."
        )

        print(
            "Production model was not changed."
        )

    print()
    print("Classification report:")

    print(
        best_result[
            "classification_report"
        ]
    )

    print("Confusion matrix:")

    print(
        best_result[
            "confusion_matrix"
        ]
    )

    print()
    print(
        f"Candidate model:\n"
        f"{CANDIDATE_MODEL_PATH}"
    )

    print(
        f"Report:\n"
        f"{REPORT_PATH}"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()
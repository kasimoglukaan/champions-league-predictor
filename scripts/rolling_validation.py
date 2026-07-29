import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "rolling_validation_candidate.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "rolling_validation_report.json"
)


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


# Validation always uses Champions League matches.
# Training uses all available competitions before validation.
VALIDATION_FOLDS: List[
    Tuple[str, str]
] = [
    (
        "2024-01-01",
        "2024-07-01",
    ),
    (
        "2024-07-01",
        "2025-01-01",
    ),
    (
        "2025-01-01",
        "2025-07-01",
    ),
    (
        "2025-07-01",
        "2026-01-01",
    ),
]


# Completely untouched final test period.
FINAL_TEST_START_DATE = "2026-01-01"


def create_logistic_model() -> Pipeline:
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


def create_xgboost_model() -> Pipeline:
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
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    n_estimators=400,
                    learning_rate=0.02,
                    max_depth=3,
                    min_child_weight=8,
                    subsample=0.80,
                    colsample_bytree=0.80,
                    reg_alpha=0.50,
                    reg_lambda=5.0,
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def create_models() -> Dict[str, Pipeline]:
    return {
        "logistic_regression": (
            create_logistic_model()
        ),
        "xgboost_regularized": (
            create_xgboost_model()
        ),
    }


def ordered_probabilities(
    model: Pipeline,
    model_name: str,
    features: pd.DataFrame,
) -> np.ndarray:
    probabilities = model.predict_proba(
        features
    )

    if model_name == "xgboost_regularized":
        return probabilities

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
        ] = probabilities[
            :,
            class_index,
        ]

    return ordered


def decode_predictions(
    predictions: np.ndarray,
    model_name: str,
) -> np.ndarray:
    if model_name != "xgboost_regularized":
        return predictions

    return np.array(
        [
            NUMBER_TO_LABEL[int(value)]
            for value in predictions
        ]
    )


def encode_targets(
    targets: pd.Series,
    model_name: str,
) -> pd.Series:
    if model_name != "xgboost_regularized":
        return targets

    encoded = targets.map(
        LABEL_TO_NUMBER
    )

    if encoded.isna().any():
        raise ValueError(
            "Unknown target class encountered."
        )

    return encoded


def evaluate_model(
    model: Pipeline,
    model_name: str,
    features: pd.DataFrame,
    targets: pd.Series,
) -> Dict[str, float]:
    raw_predictions = model.predict(
        features
    )

    predictions = decode_predictions(
        predictions=raw_predictions,
        model_name=model_name,
    )

    probabilities = ordered_probabilities(
        model=model,
        model_name=model_name,
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


def calculate_average(
    values: List[float],
) -> float:
    if not values:
        raise ValueError(
            "Cannot average an empty list."
        )

    return float(
        sum(values) / len(values)
    )


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Run this command first:\n"
            "python scripts/download_data.py"
        )

    print("=" * 70)
    print("ROLLING TIME-SERIES VALIDATION")
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

    dataset = feature_builder.build(
        matches
    )

    dataset["date"] = pd.to_datetime(
        dataset["date"],
        utc=True,
        errors="raise",
    )

    dataset = dataset.sort_values(
        "date"
    ).reset_index(drop=True)

    feature_columns = (
        FeatureBuilder.FEATURE_COLUMNS
    )

    print(
        f"Generated feature rows: "
        f"{len(dataset):,}"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    models = create_models()

    validation_results: Dict[
        str,
        List[Dict[str, float]]
    ] = {
        model_name: []
        for model_name in models
    }

    fold_information = []

    for fold_number, (
        validation_start_text,
        validation_end_text,
    ) in enumerate(
        VALIDATION_FOLDS,
        start=1,
    ):
        validation_start = pd.Timestamp(
            validation_start_text,
            tz="UTC",
        )

        validation_end = pd.Timestamp(
            validation_end_text,
            tz="UTC",
        )

        train_data = dataset[
            dataset["date"]
            < validation_start
        ].copy()

        validation_data = dataset[
            (
                dataset["date"]
                >= validation_start
            )
            & (
                dataset["date"]
                < validation_end
            )
            & (
                dataset["competition"]
                == "CL"
            )
        ].copy()

        if train_data.empty:
            print()
            print(
                f"Fold {fold_number} skipped: "
                "training data is empty."
            )
            continue

        if validation_data.empty:
            print()
            print(
                f"Fold {fold_number} skipped: "
                "validation data is empty."
            )
            continue

        X_train = train_data[
            feature_columns
        ].copy()

        X_validation = validation_data[
            feature_columns
        ].copy()

        y_train_original = train_data[
            "target"
        ].copy()

        y_validation = validation_data[
            "target"
        ].copy()

        print()
        print("=" * 70)

        print(
            f"FOLD {fold_number}: "
            f"{validation_start_text} "
            f"to {validation_end_text}"
        )

        print("=" * 70)

        print(
            f"Training rows: "
            f"{len(train_data):,}"
        )

        print(
            f"Validation CL rows: "
            f"{len(validation_data):,}"
        )

        fold_information.append(
            {
                "fold": fold_number,
                "validation_start": (
                    validation_start_text
                ),
                "validation_end": (
                    validation_end_text
                ),
                "training_rows": int(
                    len(train_data)
                ),
                "validation_rows": int(
                    len(validation_data)
                ),
            }
        )

        for model_name in models:
            model = create_models()[
                model_name
            ]

            y_train = encode_targets(
                targets=y_train_original,
                model_name=model_name,
            )

            model.fit(
                X_train,
                y_train,
            )

            result = evaluate_model(
                model=model,
                model_name=model_name,
                features=X_validation,
                targets=y_validation,
            )

            validation_results[
                model_name
            ].append(result)

            print()
            print(model_name)

            print(
                f"Accuracy: "
                f"{result['accuracy']:.2%}"
            )

            print(
                f"Log loss: "
                f"{result['log_loss']:.4f}"
            )

    summary_results = {}

    print()
    print("=" * 70)
    print("ROLLING VALIDATION SUMMARY")
    print("=" * 70)

    for model_name, fold_results in (
        validation_results.items()
    ):
        if not fold_results:
            continue

        accuracy_values = [
            result["accuracy"]
            for result in fold_results
        ]

        log_loss_values = [
            result["log_loss"]
            for result in fold_results
        ]

        average_accuracy = (
            calculate_average(
                accuracy_values
            )
        )

        average_log_loss = (
            calculate_average(
                log_loss_values
            )
        )

        accuracy_std = float(
            np.std(accuracy_values)
        )

        log_loss_std = float(
            np.std(log_loss_values)
        )

        summary_results[
            model_name
        ] = {
            "average_accuracy": (
                average_accuracy
            ),
            "average_log_loss": (
                average_log_loss
            ),
            "accuracy_std": accuracy_std,
            "log_loss_std": log_loss_std,
            "folds": fold_results,
        }

        print()
        print(model_name)

        print(
            f"Average accuracy: "
            f"{average_accuracy:.2%}"
        )

        print(
            f"Accuracy std: "
            f"{accuracy_std:.2%}"
        )

        print(
            f"Average log loss: "
            f"{average_log_loss:.4f}"
        )

        print(
            f"Log loss std: "
            f"{log_loss_std:.4f}"
        )

    if not summary_results:
        raise RuntimeError(
            "No validation results were generated."
        )

    best_model_name = min(
        summary_results,
        key=lambda name: (
            summary_results[name][
                "average_log_loss"
            ],
            -summary_results[name][
                "average_accuracy"
            ],
            summary_results[name][
                "accuracy_std"
            ],
        ),
    )

    print()
    print(
        f"Selected model from rolling "
        f"validation: {best_model_name}"
    )

    final_test_start = pd.Timestamp(
        FINAL_TEST_START_DATE,
        tz="UTC",
    )

    development_data = dataset[
        dataset["date"]
        < final_test_start
    ].copy()

    final_test_data = dataset[
        (
            dataset["date"]
            >= final_test_start
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

    if final_test_data.empty:
        raise RuntimeError(
            "Final Champions League "
            "test data is empty."
        )

    X_development = development_data[
        feature_columns
    ].copy()

    y_development_original = (
        development_data[
            "target"
        ].copy()
    )

    X_final_test = final_test_data[
        feature_columns
    ].copy()

    y_final_test = final_test_data[
        "target"
    ].copy()

    final_model = create_models()[
        best_model_name
    ]

    y_development = encode_targets(
        targets=y_development_original,
        model_name=best_model_name,
    )

    final_model.fit(
        X_development,
        y_development,
    )

    final_result = evaluate_model(
        model=final_model,
        model_name=best_model_name,
        features=X_final_test,
        targets=y_final_test,
    )

    CANDIDATE_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model_name": best_model_name,
            "model": final_model,
            "label_order": LABEL_ORDER,
            "numeric_classes": (
                best_model_name
                == "xgboost_regularized"
            ),
            "feature_columns": (
                feature_columns
            ),
            "final_test_start": (
                FINAL_TEST_START_DATE
            ),
        },
        CANDIDATE_MODEL_PATH,
    )

    report = {
        "dataset_rows": int(
            len(dataset)
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "folds": fold_information,
        "rolling_validation": (
            summary_results
        ),
        "selected_model": (
            best_model_name
        ),
        "final_test": {
            "start_date": (
                FINAL_TEST_START_DATE
            ),
            "development_rows": int(
                len(development_data)
            ),
            "test_rows": int(
                len(final_test_data)
            ),
            "accuracy": (
                final_result["accuracy"]
            ),
            "log_loss": (
                final_result["log_loss"]
            ),
        },
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
    print("=" * 70)
    print("FINAL UNSEEN CHAMPIONS LEAGUE TEST")
    print("=" * 70)

    print(
        f"Selected model: "
        f"{best_model_name}"
    )

    print(
        f"Development matches used: "
        f"{len(development_data):,}"
    )

    print(
        f"Final CL test matches: "
        f"{len(final_test_data):,}"
    )

    print(
        f"Final test accuracy: "
        f"{final_result['accuracy']:.2%}"
    )

    print(
        f"Final test log loss: "
        f"{final_result['log_loss']:.4f}"
    )

    print()
    print(
        "Do not compare this accuracy "
        "directly with the old 57.67% "
        "result because the final test "
        "period is different."
    )

    print()
    print(
        f"Candidate model:\n"
        f"{CANDIDATE_MODEL_PATH}"
    )

    print(
        f"Validation report:\n"
        f"{REPORT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
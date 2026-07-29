import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    / "ensemble_candidate.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "ensemble_report.json"
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


FINAL_TEST_START_DATE = "2026-01-01"


XGBOOST_WEIGHTS = [
    0.50,
    0.60,
    0.70,
    0.80,
]


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


def get_logistic_probabilities(
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


def get_xgboost_probabilities(
    model: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    return model.predict_proba(features)


def combine_probabilities(
    logistic_probabilities: np.ndarray,
    xgboost_probabilities: np.ndarray,
    xgboost_weight: float,
) -> np.ndarray:
    logistic_weight = (
        1.0 - xgboost_weight
    )

    combined = (
        logistic_probabilities
        * logistic_weight
        + xgboost_probabilities
        * xgboost_weight
    )

    row_totals = combined.sum(
        axis=1,
        keepdims=True,
    )

    return combined / row_totals


def probabilities_to_labels(
    probabilities: np.ndarray,
) -> np.ndarray:
    winning_indexes = np.argmax(
        probabilities,
        axis=1,
    )

    return np.array(
        [
            LABEL_ORDER[index]
            for index in winning_indexes
        ]
    )


def evaluate_probabilities(
    targets: pd.Series,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    predictions = probabilities_to_labels(
        probabilities
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


def calculate_mean(
    values: List[float],
) -> float:
    return float(
        sum(values) / len(values)
    )


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Run python scripts/download_data.py first."
        )

    print("=" * 70)
    print("LOGISTIC + XGBOOST ENSEMBLE")
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

    validation_results: Dict[
        str,
        List[Dict[str, float]],
    ] = {
        str(weight): []
        for weight in XGBOOST_WEIGHTS
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

        if (
            train_data.empty
            or validation_data.empty
        ):
            print(
                f"Fold {fold_number} skipped."
            )
            continue

        X_train = train_data[
            feature_columns
        ].copy()

        y_train_labels = train_data[
            "target"
        ].copy()

        y_train_numbers = (
            y_train_labels.map(
                LABEL_TO_NUMBER
            )
        )

        X_validation = validation_data[
            feature_columns
        ].copy()

        y_validation = validation_data[
            "target"
        ].copy()

        logistic_model = (
            create_logistic_model()
        )

        xgboost_model = (
            create_xgboost_model()
        )

        logistic_model.fit(
            X_train,
            y_train_labels,
        )

        xgboost_model.fit(
            X_train,
            y_train_numbers,
        )

        logistic_probabilities = (
            get_logistic_probabilities(
                model=logistic_model,
                features=X_validation,
            )
        )

        xgboost_probabilities = (
            get_xgboost_probabilities(
                model=xgboost_model,
                features=X_validation,
            )
        )

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

        for xgboost_weight in (
            XGBOOST_WEIGHTS
        ):
            combined_probabilities = (
                combine_probabilities(
                    logistic_probabilities=(
                        logistic_probabilities
                    ),
                    xgboost_probabilities=(
                        xgboost_probabilities
                    ),
                    xgboost_weight=(
                        xgboost_weight
                    ),
                )
            )

            result = (
                evaluate_probabilities(
                    targets=y_validation,
                    probabilities=(
                        combined_probabilities
                    ),
                )
            )

            validation_results[
                str(xgboost_weight)
            ].append(result)

            print()
            print(
                f"XGBoost weight: "
                f"{xgboost_weight:.0%}"
            )

            print(
                f"Logistic weight: "
                f"{1 - xgboost_weight:.0%}"
            )

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
    print("ENSEMBLE VALIDATION SUMMARY")
    print("=" * 70)

    for weight_text, results in (
        validation_results.items()
    ):
        if not results:
            continue

        accuracy_values = [
            result["accuracy"]
            for result in results
        ]

        log_loss_values = [
            result["log_loss"]
            for result in results
        ]

        average_accuracy = calculate_mean(
            accuracy_values
        )

        average_log_loss = calculate_mean(
            log_loss_values
        )

        accuracy_std = float(
            np.std(accuracy_values)
        )

        log_loss_std = float(
            np.std(log_loss_values)
        )

        summary_results[
            weight_text
        ] = {
            "xgboost_weight": float(
                weight_text
            ),
            "logistic_weight": (
                1.0
                - float(weight_text)
            ),
            "average_accuracy": (
                average_accuracy
            ),
            "average_log_loss": (
                average_log_loss
            ),
            "accuracy_std": (
                accuracy_std
            ),
            "log_loss_std": (
                log_loss_std
            ),
            "folds": results,
        }

        print()
        print(
            f"XGBoost weight: "
            f"{float(weight_text):.0%}"
        )

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

    if not summary_results:
        raise RuntimeError(
            "No ensemble results were generated."
        )

    best_weight_text = min(
        summary_results,
        key=lambda weight: (
            summary_results[weight][
                "average_log_loss"
            ],
            -summary_results[weight][
                "average_accuracy"
            ],
            summary_results[weight][
                "accuracy_std"
            ],
        ),
    )

    best_xgboost_weight = float(
        best_weight_text
    )

    print()
    print(
        f"Selected XGBoost weight: "
        f"{best_xgboost_weight:.0%}"
    )

    print(
        f"Selected Logistic weight: "
        f"{1 - best_xgboost_weight:.0%}"
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

    if final_test_data.empty:
        raise RuntimeError(
            "Final CL test data is empty."
        )

    X_development = development_data[
        feature_columns
    ].copy()

    y_development_labels = (
        development_data[
            "target"
        ].copy()
    )

    y_development_numbers = (
        y_development_labels.map(
            LABEL_TO_NUMBER
        )
    )

    X_final_test = final_test_data[
        feature_columns
    ].copy()

    y_final_test = final_test_data[
        "target"
    ].copy()

    final_logistic_model = (
        create_logistic_model()
    )

    final_xgboost_model = (
        create_xgboost_model()
    )

    final_logistic_model.fit(
        X_development,
        y_development_labels,
    )

    final_xgboost_model.fit(
        X_development,
        y_development_numbers,
    )

    final_logistic_probabilities = (
        get_logistic_probabilities(
            model=final_logistic_model,
            features=X_final_test,
        )
    )

    final_xgboost_probabilities = (
        get_xgboost_probabilities(
            model=final_xgboost_model,
            features=X_final_test,
        )
    )

    final_probabilities = (
        combine_probabilities(
            logistic_probabilities=(
                final_logistic_probabilities
            ),
            xgboost_probabilities=(
                final_xgboost_probabilities
            ),
            xgboost_weight=(
                best_xgboost_weight
            ),
        )
    )

    final_result = evaluate_probabilities(
        targets=y_final_test,
        probabilities=final_probabilities,
    )

    final_predictions = (
        probabilities_to_labels(
            final_probabilities
        )
    )

    CANDIDATE_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model_name": (
                "logistic_xgboost_ensemble"
            ),
            "logistic_model": (
                final_logistic_model
            ),
            "xgboost_model": (
                final_xgboost_model
            ),
            "xgboost_weight": (
                best_xgboost_weight
            ),
            "logistic_weight": (
                1.0
                - best_xgboost_weight
            ),
            "label_order": LABEL_ORDER,
            "feature_columns": (
                feature_columns
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
        "validation": summary_results,
        "selected_weights": {
            "xgboost": (
                best_xgboost_weight
            ),
            "logistic": (
                1.0
                - best_xgboost_weight
            ),
        },
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
    print("FINAL ENSEMBLE TEST")
    print("=" * 70)

    print(
        f"Development matches: "
        f"{len(development_data):,}"
    )

    print(
        f"Final CL test matches: "
        f"{len(final_test_data):,}"
    )

    print(
        f"XGBoost weight: "
        f"{best_xgboost_weight:.0%}"
    )

    print(
        f"Logistic weight: "
        f"{1 - best_xgboost_weight:.0%}"
    )

    print(
        f"Final accuracy: "
        f"{final_result['accuracy']:.2%}"
    )

    print(
        f"Final log loss: "
        f"{final_result['log_loss']:.4f}"
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            y_final_test,
            final_predictions,
            labels=LABEL_ORDER,
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_final_test,
            final_predictions,
            labels=LABEL_ORDER,
        )
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

    print("=" * 70)


if __name__ == "__main__":
    main()
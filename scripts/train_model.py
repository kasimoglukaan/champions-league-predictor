import json
import shutil
import sys
from pathlib import Path
from typing import Dict

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.evaluation.model_evaluator import (
    ModelEvaluator,
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

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_data.csv"
)

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "match_model.joblib"
)

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "candidate_model.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "model_comparison.json"
)

TEST_START_DATE = "2024-07-01"

BASELINE_ACCURACY = 0.5767
BASELINE_LOG_LOSS = 1.0054


def print_result(
    model_name: str,
    result: Dict[str, float],
) -> None:
    print()
    print(model_name)
    print("-" * 50)

    print(
        f"Accuracy: "
        f"{result['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{result['log_loss']:.4f}"
    )

    print(
        f"Brier score: "
        f"{result['multiclass_brier']:.4f}"
    )


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Run this command first:\n"
            "python scripts/download_data.py"
        )

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

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

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
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
    ].copy()

    X_test = test_data[
        feature_columns
    ].copy()

    y_test = test_data[
        "target"
    ].copy()

    print(
        f"Training matches: "
        f"{len(train_data):,}"
    )

    print(
        f"Test matches: "
        f"{len(test_data):,}"
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    model_manager = (
        MachineLearningModel()
    )

    training_status = (
        model_manager.fit_all(
            X_train=X_train,
            y_train=y_train,
        )
    )

    evaluator = ModelEvaluator()

    comparison_results: Dict[
        str,
        Dict[str, float],
    ] = {}

    for model_name, status in (
        training_status.items()
    ):
        if status != "success":
            print(
                f"Skipping {model_name}: "
                f"{status}"
            )
            continue

        try:
            result = (
                evaluator
                .evaluate_named_model(
                    model_manager=(
                        model_manager
                    ),
                    model_name=model_name,
                    X_test=X_test,
                    y_test=y_test,
                )
            )

        except Exception as error:
            print(
                f"Evaluation failed for "
                f"{model_name}: {error}"
            )
            continue

        comparison_results[
            model_name
        ] = result

        print_result(
            model_name=model_name,
            result=result,
        )

    if not comparison_results:
        raise RuntimeError(
            "All models failed."
        )

    best_model_name = (
        model_manager.select_best_model(
            comparison_results
        )
    )

    final_result = evaluator.evaluate(
        model=model_manager,
        X_test=X_test,
        y_test=y_test,
    )

    model_manager.save(
        str(CANDIDATE_MODEL_PATH)
    )

    candidate_accuracy = float(
        final_result["accuracy"]
    )

    candidate_log_loss = float(
        final_result["log_loss"]
    )

    accuracy_improved = (
        candidate_accuracy
        > BASELINE_ACCURACY
    )

    log_loss_improved = (
        candidate_log_loss
        < BASELINE_LOG_LOSS
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
        "best_candidate": {
            "model": best_model_name,
            "accuracy": (
                candidate_accuracy
            ),
            "log_loss": (
                candidate_log_loss
            ),
            "brier_score": float(
                final_result[
                    "multiclass_brier"
                ]
            ),
        },
        "promoted_to_production": (
            promoted
        ),
        "all_models": (
            comparison_results
        ),
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
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        f"Best candidate: "
        f"{best_model_name}"
    )

    print(
        f"Candidate accuracy: "
        f"{candidate_accuracy:.2%}"
    )

    print(
        f"Candidate log loss: "
        f"{candidate_log_loss:.4f}"
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
            "Candidate promoted to "
            "production model."
        )
    else:
        print(
            "Candidate did not beat "
            "the baseline."
        )

        print(
            "Production model was "
            "not changed."
        )

    print()
    print("Classification report:")

    print(
        final_result[
            "classification_report"
        ]
    )

    print("Confusion matrix:")

    print(
        final_result[
            "confusion_matrix"
        ]
    )

    print()
    print(
        f"Candidate model:\n"
        f"{CANDIDATE_MODEL_PATH}"
    )

    print(
        f"Comparison report:\n"
        f"{REPORT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
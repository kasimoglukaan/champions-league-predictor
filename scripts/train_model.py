import sys
from pathlib import Path
from typing import Dict

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.evaluation.model_evaluator import ModelEvaluator
from src.features.feature_builder import FeatureBuilder
from src.models.machine_learning_model import MachineLearningModel


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

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "match_model.joblib"
)


# Matches before this date are used for training.
# Champions League matches on/after this date are used for testing.
TEST_START_DATE = "2024-07-01"


def print_model_result(
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
    # 1. Check raw data
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Historical match data was not found.\n"
            "Run this command first:\n"
            "python scripts/download_data.py"
        )

    print("=" * 60)
    print("CHAMPIONS LEAGUE MODEL TRAINING")
    print("=" * 60)

    # 2. Load downloaded matches
    print()
    print("Loading historical match data...")

    matches = pd.read_csv(
        RAW_DATA_PATH
    )

    if matches.empty:
        raise RuntimeError(
            "The raw match dataset is empty."
        )

    print(
        f"Loaded {len(matches):,} matches."
    )

    # 3. Build chronological pre-match features
    print()
    print("Generating pre-match features...")

    feature_builder = FeatureBuilder(
        initial_elo=1500.0,
        k_factor=25.0,
        home_advantage=60.0,
    )

    dataset = feature_builder.build(
        matches
    )

    if dataset.empty:
        raise RuntimeError(
            "Feature generation returned "
            "an empty dataset."
        )

    # 4. Save generated training dataset
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print(
        f"Generated {len(dataset):,} "
        f"feature rows."
    )

    print(
        f"Processed dataset saved to:\n"
        f"{PROCESSED_DATA_PATH}"
    )

    # 5. Convert dates
    dataset["date"] = pd.to_datetime(
        dataset["date"],
        utc=True,
        errors="raise",
    )

    test_start_timestamp = pd.Timestamp(
        TEST_START_DATE,
        tz="UTC",
    )

    # 6. Chronological train/test split
    train_data = dataset[
        dataset["date"]
        < test_start_timestamp
    ].copy()

    test_data = dataset[
        (
            dataset["date"]
            >= test_start_timestamp
        )
        & (
            dataset["competition"]
            == "CL"
        )
    ].copy()

    if train_data.empty:
        raise RuntimeError(
            "Training dataset is empty. "
            "Check TEST_START_DATE."
        )

    if test_data.empty:
        raise RuntimeError(
            "Champions League test dataset "
            "is empty. Check TEST_START_DATE "
            "and competition codes."
        )

    # 7. Select model features
    feature_columns = (
        FeatureBuilder.FEATURE_COLUMNS
    )

    missing_features = (
        set(feature_columns)
        - set(dataset.columns)
    )

    if missing_features:
        raise ValueError(
            "Generated dataset is missing "
            "these feature columns: "
            + ", ".join(
                sorted(missing_features)
            )
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

    print()
    print("=" * 60)
    print("DATA SPLIT")
    print("=" * 60)

    print(
        f"Training matches: "
        f"{len(train_data):,}"
    )

    print(
        f"Test matches: "
        f"{len(test_data):,}"
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Training period: "
        f"{train_data['date'].min()} "
        f"to "
        f"{train_data['date'].max()}"
    )

    print(
        f"Test period: "
        f"{test_data['date'].min()} "
        f"to "
        f"{test_data['date'].max()}"
    )

    print()
    print("Training target distribution:")

    print(
        y_train.value_counts(
            normalize=True
        ).sort_index()
    )

    print()
    print("Test target distribution:")

    print(
        y_test.value_counts(
            normalize=True
        ).sort_index()
    )

    # 8. Train all models
    print()
    print("=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    model_manager = (
        MachineLearningModel()
    )

    model_manager.fit_all(
        X_train=X_train,
        y_train=y_train,
    )

    evaluator = ModelEvaluator()

    comparison_results: Dict[
        str,
        Dict[str, float],
    ] = {}

    # 9. Evaluate each model
    for model_name in (
        model_manager.models.keys()
    ):
        result = (
            evaluator.evaluate_named_model(
                model_manager=(
                    model_manager
                ),
                model_name=model_name,
                X_test=X_test,
                y_test=y_test,
            )
        )

        comparison_results[
            model_name
        ] = result

        print_model_result(
            model_name=model_name,
            result=result,
        )

    # 10. Select best model
    best_model_name = (
        model_manager.select_best_model(
            comparison_results
        )
    )

    # 11. Full evaluation of selected model
    final_result = evaluator.evaluate(
        model=model_manager,
        X_test=X_test,
        y_test=y_test,
    )

    # 12. Save selected model
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_manager.save(
        str(MODEL_PATH)
    )

    # 13. Print final report
    print()
    print("=" * 60)
    print("FINAL MODEL RESULTS")
    print("=" * 60)

    print(
        f"Best model: "
        f"{best_model_name}"
    )

    print(
        f"Test matches: "
        f"{len(test_data):,}"
    )

    print(
        f"Accuracy: "
        f"{final_result['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{final_result['log_loss']:.4f}"
    )

    print(
        f"Brier score: "
        f"{final_result['multiclass_brier']:.4f}"
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
        f"Model saved to:\n"
        f"{MODEL_PATH}"
    )

    print()
    print(
        f"Processed features saved to:\n"
        f"{PROCESSED_DATA_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
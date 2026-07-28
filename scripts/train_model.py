import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "match_model.joblib"
)

TEST_START_DATE = "2024-07-01"


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "First run: "
            "python scripts/download_data.py"
        )

    matches = pd.read_csv(
        RAW_DATA_PATH
    )

    feature_builder = FeatureBuilder(
        initial_elo=1500,
        k_factor=25,
        home_advantage=60,
        form_window=8,
    )

    dataset = feature_builder.build(matches)

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    feature_columns = (
        FeatureBuilder.FEATURE_COLUMNS
    )

    dataset["date"] = pd.to_datetime(
        dataset["date"],
        utc=True,
    )

    train_data = dataset[
        dataset["date"]
        < pd.Timestamp(
            TEST_START_DATE,
            tz="UTC",
        )
    ].copy()

    test_data = dataset[
        (
            dataset["date"]
            >= pd.Timestamp(
                TEST_START_DATE,
                tz="UTC",
            )
        )
        & (
            dataset["competition"]
            == "CL"
        )
    ].copy()

    if train_data.empty:
        raise RuntimeError(
            "Training dataset is empty."
        )

    if test_data.empty:
        raise RuntimeError(
            "Champions League test "
            "dataset is empty."
        )

    X_train = train_data[
        feature_columns
    ]

    y_train = train_data["target"]

    X_test = test_data[
        feature_columns
    ]

    y_test = test_data["target"]

    model_manager = MachineLearningModel()
    model_manager.fit_all(
        X_train,
        y_train,
    )

    evaluator = ModelEvaluator()

    comparison_results = {}

    for model_name, model in (
        model_manager.models.items()
    ):
        result = (
            evaluator.evaluate_single_model(
                sklearn_model=model,
                X_test=X_test,
                y_test=y_test,
            )
        )

        comparison_results[
            model_name
        ] = result

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
            f"Brier: "
            f"{result['multiclass_brier']:.4f}"
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
        str(MODEL_PATH)
    )

    print()
    print("=" * 60)
    print(
        f"Best model: {best_model_name}"
    )
    print(
        f"Test matches: {len(test_data)}"
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
    print(f"Model saved to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.features.live_feature_builder import (
    LiveFeatureBuilder,
)


EXPANDED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_matches_expanded_clean.csv"
)

ORIGINAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_matches.csv"
)

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "expanded_match_model_candidate.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "expanded_model_report.json"
)


LABEL_ORDER = [
    "A",
    "D",
    "H",
]

TEST_MATCH_COUNT = 378

INITIAL_ELO = 1500.0
K_FACTOR = 25.0
HOME_ADVANTAGE = 60.0
FORM_WINDOW = 8


def determine_winner(
    home_goals: int,
    away_goals: int,
) -> str:
    if home_goals > away_goals:
        return "H"

    if away_goals > home_goals:
        return "A"

    return "D"


def normalize_competition(
    competition: object,
) -> str:
    return (
        str(competition)
        .strip()
        .upper()
    )


def is_champions_league(
    competition: object,
) -> bool:
    normalized = normalize_competition(
        competition
    )

    return normalized in {
        "CL",
        "CHAMPIONS_LEAGUE",
        "UEFA_CHAMPIONS_LEAGUE",
        "UEFA CHAMPIONS LEAGUE",
    }


def load_matches(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    matches = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    }

    missing_columns = (
        required_columns
        - set(matches.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    matches = matches.copy()

    matches["date"] = pd.to_datetime(
        matches["date"],
        errors="coerce",
        utc=True,
    )

    matches["home_team"] = (
        matches["home_team"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    matches["away_team"] = (
        matches["away_team"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    matches["home_goals"] = pd.to_numeric(
        matches["home_goals"],
        errors="coerce",
    )

    matches["away_goals"] = pd.to_numeric(
        matches["away_goals"],
        errors="coerce",
    )

    if "competition" not in matches.columns:
        matches["competition"] = "UNKNOWN"

    matches["competition"] = (
        matches["competition"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )

    matches = matches.dropna(
        subset=[
            "date",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    matches = matches[
        (matches["home_team"] != "")
        & (matches["away_team"] != "")
        & (
            matches["home_team"]
            != matches["away_team"]
        )
    ].copy()

    matches["home_goals"] = (
        matches["home_goals"]
        .astype(int)
    )

    matches["away_goals"] = (
        matches["away_goals"]
        .astype(int)
    )

    if "winner" not in matches.columns:
        matches["winner"] = [
            determine_winner(
                int(home_goals),
                int(away_goals),
            )
            for home_goals, away_goals
            in zip(
                matches["home_goals"],
                matches["away_goals"],
            )
        ]

    else:
        matches["winner"] = (
            matches["winner"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        invalid_winner_mask = (
            ~matches["winner"]
            .isin(LABEL_ORDER)
        )

        if invalid_winner_mask.any():
            matches.loc[
                invalid_winner_mask,
                "winner",
            ] = [
                determine_winner(
                    int(home_goals),
                    int(away_goals),
                )
                for home_goals, away_goals
                in zip(
                    matches.loc[
                        invalid_winner_mask,
                        "home_goals",
                    ],
                    matches.loc[
                        invalid_winner_mask,
                        "away_goals",
                    ],
                )
            ]

    return (
        matches.sort_values(
            by=[
                "date",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(drop=True)
    )


def select_test_matches(
    expanded_matches: pd.DataFrame,
) -> pd.DataFrame:
    champions_league_matches = (
        expanded_matches[
            expanded_matches[
                "competition"
            ].map(
                is_champions_league
            )
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if (
        len(champions_league_matches)
        < TEST_MATCH_COUNT
    ):
        raise ValueError(
            "There are not enough Champions "
            "League matches for the test set. "
            f"Available: "
            f"{len(champions_league_matches):,}; "
            f"required: {TEST_MATCH_COUNT:,}."
        )

    return (
        champions_league_matches
        .tail(TEST_MATCH_COUNT)
        .copy()
        .reset_index(drop=True)
    )


def create_training_matches(
    matches: pd.DataFrame,
    test_start_date: pd.Timestamp,
) -> pd.DataFrame:
    training_matches = matches[
        matches["date"]
        < test_start_date
    ].copy()

    if training_matches.empty:
        raise ValueError(
            "No training matches remain "
            "before the test period."
        )

    return (
        training_matches
        .sort_values(
            by=[
                "date",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(drop=True)
    )


def create_empty_feature_builder(
) -> LiveFeatureBuilder:
    builder = LiveFeatureBuilder(
        initial_elo=INITIAL_ELO,
        k_factor=K_FACTOR,
        home_advantage=HOME_ADVANTAGE,
        form_window=FORM_WINDOW,
    )

    empty_matches = pd.DataFrame(
        {
            "date": pd.Series(
                dtype="datetime64[ns, UTC]"
            ),
            "home_team": pd.Series(
                dtype=str
            ),
            "away_team": pd.Series(
                dtype=str
            ),
            "home_goals": pd.Series(
                dtype=int
            ),
            "away_goals": pd.Series(
                dtype=int
            ),
        }
    )

    builder.fit(
        empty_matches
    )

    return builder


def actual_rest_days(
    builder: LiveFeatureBuilder,
    team: str,
    match_date: pd.Timestamp,
) -> float:
    previous_date = (
        builder.last_match_dates.get(
            team
        )
    )

    if previous_date is None:
        return 7.0

    difference = (
        match_date
        - previous_date
    ).days

    return float(
        max(
            1,
            min(
                difference,
                30,
            ),
        )
    )


def build_feature_before_match(
    builder: LiveFeatureBuilder,
    match: Any,
) -> pd.DataFrame:
    home_team = str(
        match.home_team
    )

    away_team = str(
        match.away_team
    )

    match_date = pd.Timestamp(
        match.date
    )

    feature_row = (
        builder.build_match_features(
            home_team=home_team,
            away_team=away_team,
            competition=str(
                match.competition
            ),
        )
    )

    # LiveFeatureBuilder uses the current date for
    # rest-day calculations. During historical
    # training we must replace those values with
    # the actual pre-match rest periods.
    feature_row.loc[
        0,
        "home_rest_days",
    ] = actual_rest_days(
        builder=builder,
        team=home_team,
        match_date=match_date,
    )

    feature_row.loc[
        0,
        "away_rest_days",
    ] = actual_rest_days(
        builder=builder,
        team=away_team,
        match_date=match_date,
    )

    feature_row.loc[
        0,
        "is_champions_league",
    ] = int(
        is_champions_league(
            match.competition
        )
    )

    return feature_row


def update_builder_after_match(
    builder: LiveFeatureBuilder,
    match: Any,
) -> None:
    home_team = str(
        match.home_team
    )

    away_team = str(
        match.away_team
    )

    home_goals = int(
        match.home_goals
    )

    away_goals = int(
        match.away_goals
    )

    match_date = pd.Timestamp(
        match.date
    )

    builder._update_elo(
        home_team=home_team,
        away_team=away_team,
        home_goals=home_goals,
        away_goals=away_goals,
    )

    builder._append_history(
        team=home_team,
        goals_scored=home_goals,
        goals_conceded=away_goals,
    )

    builder._append_history(
        team=away_team,
        goals_scored=away_goals,
        goals_conceded=home_goals,
    )

    builder.last_match_dates[
        home_team
    ] = match_date

    builder.last_match_dates[
        away_team
    ] = match_date

    builder.latest_data_date = max(
        builder.latest_data_date,
        match_date,
    )


def generate_training_features(
    training_matches: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    LiveFeatureBuilder,
]:
    builder = (
        create_empty_feature_builder()
    )

    feature_rows = []
    labels = []

    total_matches = len(
        training_matches
    )

    for index, match in enumerate(
        training_matches.itertuples(
            index=False
        ),
        start=1,
    ):
        feature_row = (
            build_feature_before_match(
                builder=builder,
                match=match,
            )
        )

        feature_rows.append(
            feature_row
        )

        labels.append(
            str(match.winner)
        )

        update_builder_after_match(
            builder=builder,
            match=match,
        )

        if (
            index % 5000 == 0
            or index == total_matches
        ):
            print(
                "  Training features: "
                f"{index:,}/"
                f"{total_matches:,}"
            )

    features = pd.concat(
        feature_rows,
        ignore_index=True,
    )

    labels_series = pd.Series(
        labels,
        name="winner",
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    features = features.fillna(
        0.0
    )

    return (
        features,
        labels_series,
        builder,
    )


def generate_test_features(
    builder: LiveFeatureBuilder,
    test_matches: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    feature_rows = []
    labels = []

    for match in test_matches.itertuples(
        index=False
    ):
        feature_row = (
            build_feature_before_match(
                builder=builder,
                match=match,
            )
        )

        feature_rows.append(
            feature_row
        )

        labels.append(
            str(match.winner)
        )

        # The next test match is allowed to use
        # results from earlier test matches.
        update_builder_after_match(
            builder=builder,
            match=match,
        )

    features = pd.concat(
        feature_rows,
        ignore_index=True,
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    features = features.fillna(
        0.0
    )

    labels_series = pd.Series(
        labels,
        name="winner",
    )

    return (
        features,
        labels_series,
    )


def create_pipeline(
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    C=0.75,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )


def reorder_probabilities(
    probabilities: np.ndarray,
    classes: list[str],
) -> np.ndarray:
    class_indexes = {
        str(label): index
        for index, label
        in enumerate(classes)
    }

    missing_labels = [
        label
        for label in LABEL_ORDER
        if label not in class_indexes
    ]

    if missing_labels:
        raise ValueError(
            "Classifier output is missing labels: "
            + ", ".join(missing_labels)
        )

    return np.column_stack(
        [
            probabilities[
                :,
                class_indexes[label],
            ]
            for label in LABEL_ORDER
        ]
    )


def calculate_metrics(
    actual_labels: pd.Series,
    probabilities: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    ordered_probabilities = (
        reorder_probabilities(
            probabilities=probabilities,
            classes=classes,
        )
    )

    predicted_indexes = np.argmax(
        ordered_probabilities,
        axis=1,
    )

    predicted_labels = np.array(
        LABEL_ORDER
    )[predicted_indexes]

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=LABEL_ORDER,
    )

    return {
        "accuracy": float(
            accuracy_score(
                actual_labels,
                predicted_labels,
            )
        ),
        "log_loss": float(
            log_loss(
                actual_labels,
                ordered_probabilities,
                labels=LABEL_ORDER,
            )
        ),
        "away_precision": float(
            report["A"]["precision"]
        ),
        "away_recall": float(
            report["A"]["recall"]
        ),
        "draw_precision": float(
            report["D"]["precision"]
        ),
        "draw_recall": float(
            report["D"]["recall"]
        ),
        "draw_f1": float(
            report["D"]["f1-score"]
        ),
        "home_precision": float(
            report["H"]["precision"]
        ),
        "home_recall": float(
            report["H"]["recall"]
        ),
        "predicted_draws": int(
            (
                predicted_labels
                == "D"
            ).sum()
        ),
        "correct_draws": int(
            (
                (
                    predicted_labels
                    == "D"
                )
                & (
                    actual_labels.to_numpy()
                    == "D"
                )
            ).sum()
        ),
        "classification_report": report,
        "confusion_matrix": (
            matrix.tolist()
        ),
    }


def train_and_evaluate(
    training_matches: pd.DataFrame,
    test_matches: pd.DataFrame,
    title: str,
) -> tuple[
    Pipeline,
    list[str],
    dict[str, Any],
]:
    print()
    print(title)
    print("-" * 82)

    print(
        f"Training matches: "
        f"{len(training_matches):,}"
    )

    (
        training_features,
        training_labels,
        trained_builder,
    ) = generate_training_features(
        training_matches
    )

    (
        test_features,
        test_labels,
    ) = generate_test_features(
        builder=trained_builder,
        test_matches=test_matches,
    )

    if list(
        training_features.columns
    ) != list(
        test_features.columns
    ):
        raise ValueError(
            "Training and test feature "
            "columns do not match."
        )

    pipeline = create_pipeline()

    pipeline.fit(
        training_features,
        training_labels,
    )

    probabilities = (
        pipeline.predict_proba(
            test_features
        )
    )

    classifier = (
        pipeline.named_steps[
            "classifier"
        ]
    )

    classes = [
        str(label)
        for label in classifier.classes_
    ]

    metrics = calculate_metrics(
        actual_labels=test_labels,
        probabilities=probabilities,
        classes=classes,
    )

    return (
        pipeline,
        list(
            training_features.columns
        ),
        metrics,
    )


def save_candidate_model(
    pipeline: Pipeline,
    feature_columns: list[str],
    training_match_count: int,
) -> None:
    model_bundle = {
        "model": pipeline,
        "feature_columns": (
            feature_columns
        ),
        "label_order": (
            LABEL_ORDER
        ),
        "model_type": (
            "expanded_multileague_logistic"
        ),
        "training_matches": (
            training_match_count
        ),
        "data_path": str(
            EXPANDED_DATA_PATH
        ),
        "initial_elo": INITIAL_ELO,
        "k_factor": K_FACTOR,
        "home_advantage": (
            HOME_ADVANTAGE
        ),
        "form_window": FORM_WINDOW,
    }

    CANDIDATE_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_bundle,
        CANDIDATE_MODEL_PATH,
    )


def print_metrics(
    title: str,
    metrics: dict[str, Any],
) -> None:
    print()
    print(title)
    print("-" * 82)

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Log loss: "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Away recall: "
        f"{metrics['away_recall']:.2%}"
    )

    print(
        f"Draw precision: "
        f"{metrics['draw_precision']:.2%}"
    )

    print(
        f"Draw recall: "
        f"{metrics['draw_recall']:.2%}"
    )

    print(
        f"Draw F1: "
        f"{metrics['draw_f1']:.2%}"
    )

    print(
        f"Home recall: "
        f"{metrics['home_recall']:.2%}"
    )

    print(
        f"Predicted draws: "
        f"{metrics['predicted_draws']}"
    )

    print(
        f"Correct draws: "
        f"{metrics['correct_draws']}"
    )

    print()
    print("Confusion matrix [A, D, H]:")

    print(
        np.asarray(
            metrics[
                "confusion_matrix"
            ]
        )
    )


def main() -> None:
    print()
    print("=" * 82)
    print(
        "EXPANDED MULTI-LEAGUE "
        "MODEL EXPERIMENT"
    )
    print("=" * 82)

    expanded_matches = load_matches(
        EXPANDED_DATA_PATH
    )

    original_matches = load_matches(
        ORIGINAL_DATA_PATH
    )

    test_matches = select_test_matches(
        expanded_matches
    )

    test_start_date = (
        test_matches["date"].min()
    )

    baseline_training_matches = (
        create_training_matches(
            matches=original_matches,
            test_start_date=(
                test_start_date
            ),
        )
    )

    expanded_training_matches = (
        create_training_matches(
            matches=expanded_matches,
            test_start_date=(
                test_start_date
            ),
        )
    )

    print(
        f"Original dataset rows: "
        f"{len(original_matches):,}"
    )

    print(
        f"Expanded dataset rows: "
        f"{len(expanded_matches):,}"
    )

    print(
        f"Baseline training rows: "
        f"{len(baseline_training_matches):,}"
    )

    print(
        f"Expanded training rows: "
        f"{len(expanded_training_matches):,}"
    )

    print(
        f"Unseen CL test matches: "
        f"{len(test_matches):,}"
    )

    print(
        "Test date range: "
        f"{test_matches['date'].min()} "
        "to "
        f"{test_matches['date'].max()}"
    )

    (
        baseline_pipeline,
        baseline_feature_columns,
        baseline_metrics,
    ) = train_and_evaluate(
        training_matches=(
            baseline_training_matches
        ),
        test_matches=test_matches,
        title=(
            "BUILDING ORIGINAL-DATA BASELINE"
        ),
    )

    (
        candidate_pipeline,
        candidate_feature_columns,
        candidate_metrics,
    ) = train_and_evaluate(
        training_matches=(
            expanded_training_matches
        ),
        test_matches=test_matches,
        title=(
            "BUILDING EXPANDED-DATA CANDIDATE"
        ),
    )

    if (
        baseline_feature_columns
        != candidate_feature_columns
    ):
        raise ValueError(
            "Baseline and candidate feature "
            "columns are different."
        )

    print_metrics(
        title="ORIGINAL-DATA BASELINE",
        metrics=baseline_metrics,
    )

    print_metrics(
        title="EXPANDED-DATA CANDIDATE",
        metrics=candidate_metrics,
    )

    accuracy_difference = (
        candidate_metrics["accuracy"]
        - baseline_metrics["accuracy"]
    )

    log_loss_difference = (
        candidate_metrics["log_loss"]
        - baseline_metrics["log_loss"]
    )

    draw_recall_difference = (
        candidate_metrics["draw_recall"]
        - baseline_metrics["draw_recall"]
    )

    print()
    print("=" * 82)
    print("COMPARISON")
    print("=" * 82)

    print(
        "Accuracy difference: "
        f"{accuracy_difference:+.2%}"
    )

    print(
        "Log-loss difference: "
        f"{log_loss_difference:+.4f}"
    )

    print(
        "Draw-recall difference: "
        f"{draw_recall_difference:+.2%}"
    )

    if (
        accuracy_difference > 0
        and log_loss_difference < 0
    ):
        recommendation = (
            "PROMOTE_CANDIDATE"
        )

        print(
            "Recommendation: "
            "candidate improves both "
            "accuracy and log loss."
        )

    elif accuracy_difference > 0:
        recommendation = (
            "REVIEW_CANDIDATE"
        )

        print(
            "Recommendation: candidate "
            "improves accuracy, but probability "
            "calibration needs review."
        )

    else:
        recommendation = (
            "KEEP_BASELINE"
        )

        print(
            "Recommendation: do not replace "
            "the current model yet."
        )

    save_candidate_model(
        pipeline=candidate_pipeline,
        feature_columns=(
            candidate_feature_columns
        ),
        training_match_count=len(
            expanded_training_matches
        ),
    )

    report = {
        "expanded_dataset_path": str(
            EXPANDED_DATA_PATH
        ),
        "original_dataset_path": str(
            ORIGINAL_DATA_PATH
        ),
        "test_start_date": (
            test_start_date.isoformat()
        ),
        "test_matches": int(
            len(test_matches)
        ),
        "baseline_training_matches": int(
            len(
                baseline_training_matches
            )
        ),
        "candidate_training_matches": int(
            len(
                expanded_training_matches
            )
        ),
        "feature_count": int(
            len(
                candidate_feature_columns
            )
        ),
        "baseline": (
            baseline_metrics
        ),
        "candidate": (
            candidate_metrics
        ),
        "comparison": {
            "accuracy_difference": float(
                accuracy_difference
            ),
            "log_loss_difference": float(
                log_loss_difference
            ),
            "draw_recall_difference": float(
                draw_recall_difference
            ),
            "recommendation": (
                recommendation
            ),
        },
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Candidate model saved to:")
    print(CANDIDATE_MODEL_PATH)

    print()
    print("Report saved to:")
    print(REPORT_PATH)

    print("=" * 82)


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
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
from src.services.poisson_simulation_service import (
    PoissonSimulationService,
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_matches.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "match_model.joblib"
)

CALIBRATOR_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "hybrid_probability_calibrator.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "probability_calibration_report.json"
)

RESULTS_CSV_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "probability_calibration_results.csv"
)


TEST_START_DATE = "2024-07-01"

CALIBRATION_MATCH_COUNT = 250

ML_WEIGHT = 0.50
POISSON_WEIGHT = 0.50

FORM_WINDOW = 8

INITIAL_ELO = 1500.0
K_FACTOR = 25.0
HOME_ADVANTAGE = 60.0

POISSON_SIMULATIONS = 20_000
RANDOM_SEED = 42

LABEL_ORDER = [
    "A",
    "D",
    "H",
]

EPSILON = 1e-9
ECE_BINS = 10


HistoryItem = Tuple[
    int,
    int,
    int,
]


class MulticlassPlattCalibrator:
    """
    Multiclass Platt-style calibration.

    A multinomial logistic regression model learns a mapping from
    uncalibrated class log-probabilities to calibrated probabilities.
    """

    def __init__(
        self,
    ) -> None:
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            multi_class="auto",
            random_state=42,
        )

        self.is_fitted = False

    def fit(
        self,
        probabilities: np.ndarray,
        labels: Iterable[str],
    ) -> None:
        transformed = self._transform(
            probabilities
        )

        self.model.fit(
            transformed,
            np.asarray(
                list(labels),
                dtype=str,
            ),
        )

        self.is_fitted = True

    def predict_proba(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "Platt calibrator must be fitted first."
            )

        transformed = self._transform(
            probabilities
        )

        raw_probabilities = (
            self.model.predict_proba(
                transformed
            )
        )

        classes = [
            str(label)
            for label in self.model.classes_
        ]

        ordered = np.zeros(
            (
                len(probabilities),
                len(LABEL_ORDER),
            ),
            dtype=float,
        )

        for output_index, label in enumerate(
            LABEL_ORDER
        ):
            class_index = classes.index(
                label
            )

            ordered[
                :,
                output_index,
            ] = raw_probabilities[
                :,
                class_index,
            ]

        return normalize_probabilities(
            ordered
        )

    @staticmethod
    def _transform(
        probabilities: np.ndarray,
    ) -> np.ndarray:
        normalized = normalize_probabilities(
            probabilities
        )

        clipped = np.clip(
            normalized,
            EPSILON,
            1.0,
        )

        return np.log(
            clipped
        )


class MulticlassIsotonicCalibrator:
    """
    One-vs-rest isotonic calibration.

    One isotonic model is fitted for each result class. Class outputs are
    normalized afterwards so that each row sums to one.
    """

    def __init__(
        self,
    ) -> None:
        self.models: Dict[
            str,
            IsotonicRegression,
        ] = {}

        self.is_fitted = False

    def fit(
        self,
        probabilities: np.ndarray,
        labels: Iterable[str],
    ) -> None:
        normalized = normalize_probabilities(
            probabilities
        )

        label_array = np.asarray(
            list(labels),
            dtype=str,
        )

        self.models = {}

        for class_index, label in enumerate(
            LABEL_ORDER
        ):
            binary_targets = (
                label_array
                == label
            ).astype(
                float
            )

            calibrator = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )

            calibrator.fit(
                normalized[
                    :,
                    class_index,
                ],
                binary_targets,
            )

            self.models[
                label
            ] = calibrator

        self.is_fitted = True

    def predict_proba(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "Isotonic calibrator must be fitted first."
            )

        normalized = normalize_probabilities(
            probabilities
        )

        calibrated = np.zeros_like(
            normalized,
            dtype=float,
        )

        for class_index, label in enumerate(
            LABEL_ORDER
        ):
            calibrated[
                :,
                class_index,
            ] = self.models[
                label
            ].predict(
                normalized[
                    :,
                    class_index,
                ]
            )

        return normalize_probabilities(
            calibrated
        )


def points_for_result(
    goals_scored: int,
    goals_conceded: int,
) -> int:
    if goals_scored > goals_conceded:
        return 3

    if goals_scored == goals_conceded:
        return 1

    return 0


def determine_actual_label(
    home_goals: int,
    away_goals: int,
) -> str:
    if home_goals > away_goals:
        return "H"

    if away_goals > home_goals:
        return "A"

    return "D"


def history_summary(
    history: Deque[HistoryItem],
) -> Dict[str, float]:
    if not history:
        return {
            "form_points": 1.0,
            "goals_scored": 1.2,
            "goals_conceded": 1.2,
            "win_rate": 0.33,
        }

    rows = list(
        history
    )

    points = np.mean(
        [
            row[0]
            for row in rows
        ]
    )

    goals_scored = np.mean(
        [
            row[1]
            for row in rows
        ]
    )

    goals_conceded = np.mean(
        [
            row[2]
            for row in rows
        ]
    )

    wins = sum(
        1
        for row in rows
        if row[0] == 3
    )

    return {
        "form_points": float(
            points
        ),
        "goals_scored": float(
            goals_scored
        ),
        "goals_conceded": float(
            goals_conceded
        ),
        "win_rate": float(
            wins / len(rows)
        ),
    }


def expected_home_score(
    home_elo: float,
    away_elo: float,
) -> float:
    return 1.0 / (
        1.0
        + 10.0
        ** (
            (
                away_elo
                - home_elo
                - HOME_ADVANTAGE
            )
            / 400.0
        )
    )


def update_elo(
    ratings: Dict[str, float],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> None:
    home_elo = float(
        ratings[
            home_team
        ]
    )

    away_elo = float(
        ratings[
            away_team
        ]
    )

    expected_home = (
        expected_home_score(
            home_elo=home_elo,
            away_elo=away_elo,
        )
    )

    if home_goals > away_goals:
        actual_home = 1.0

    elif home_goals == away_goals:
        actual_home = 0.5

    else:
        actual_home = 0.0

    goal_margin = abs(
        home_goals
        - away_goals
    )

    margin_multiplier = (
        1.0
        + np.log1p(
            goal_margin
        )
        if goal_margin > 0
        else 1.0
    )

    rating_change = (
        K_FACTOR
        * margin_multiplier
        * (
            actual_home
            - expected_home
        )
    )

    ratings[
        home_team
    ] = (
        home_elo
        + rating_change
    )

    ratings[
        away_team
    ] = (
        away_elo
        - rating_change
    )


def load_matches(
) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Production dataset was not found: "
            f"{DATA_PATH}"
        )

    matches = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    required_columns = {
        "match_id",
        "date",
        "competition",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    }

    missing_columns = (
        required_columns
        - set(
            matches.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    matches = matches.copy()

    matches["date"] = pd.to_datetime(
        matches["date"],
        utc=True,
        errors="raise",
    )

    matches["home_goals"] = pd.to_numeric(
        matches["home_goals"],
        errors="raise",
    ).astype(int)

    matches["away_goals"] = pd.to_numeric(
        matches["away_goals"],
        errors="raise",
    ).astype(int)

    matches["competition"] = (
        matches["competition"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    matches["winner"] = [
        determine_actual_label(
            home_goals=int(
                home_goals
            ),
            away_goals=int(
                away_goals
            ),
        )
        for home_goals, away_goals
        in zip(
            matches["home_goals"],
            matches["away_goals"],
        )
    ]

    return (
        matches.sort_values(
            by=[
                "date",
                "match_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def normalize_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        probabilities,
        dtype=float,
    )

    values = np.clip(
        values,
        EPSILON,
        1.0,
    )

    row_sums = values.sum(
        axis=1,
        keepdims=True,
    )

    zero_rows = (
        row_sums[:, 0]
        <= 0
    )

    if np.any(
        zero_rows
    ):
        values[
            zero_rows
        ] = (
            1.0
            / len(
                LABEL_ORDER
            )
        )

        row_sums = values.sum(
            axis=1,
            keepdims=True,
        )

    return (
        values
        / row_sums
    )


def probabilities_to_labels(
    probabilities: np.ndarray,
) -> np.ndarray:
    best_indexes = np.argmax(
        probabilities,
        axis=1,
    )

    return np.asarray(
        LABEL_ORDER
    )[best_indexes]


def multiclass_brier_score(
    actual_labels: Iterable[str],
    probabilities: np.ndarray,
) -> float:
    labels = list(
        actual_labels
    )

    one_hot = np.zeros(
        (
            len(labels),
            len(LABEL_ORDER),
        ),
        dtype=float,
    )

    label_indexes = {
        label: index
        for index, label in enumerate(
            LABEL_ORDER
        )
    }

    for row_index, label in enumerate(
        labels
    ):
        one_hot[
            row_index,
            label_indexes[
                label
            ],
        ] = 1.0

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - one_hot
                )
                ** 2,
                axis=1,
            )
        )
    )


def expected_calibration_error(
    actual_labels: Iterable[str],
    probabilities: np.ndarray,
    bins: int = ECE_BINS,
) -> float:
    labels = np.asarray(
        list(
            actual_labels
        ),
        dtype=str,
    )

    normalized = normalize_probabilities(
        probabilities
    )

    predicted_labels = (
        probabilities_to_labels(
            normalized
        )
    )

    confidences = np.max(
        normalized,
        axis=1,
    )

    correctness = (
        predicted_labels
        == labels
    ).astype(
        float
    )

    boundaries = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    total_rows = len(
        labels
    )

    error = 0.0

    for bin_index in range(
        bins
    ):
        lower = boundaries[
            bin_index
        ]

        upper = boundaries[
            bin_index + 1
        ]

        if (
            bin_index
            == bins - 1
        ):
            mask = (
                confidences >= lower
            ) & (
                confidences <= upper
            )

        else:
            mask = (
                confidences >= lower
            ) & (
                confidences < upper
            )

        count = int(
            np.sum(
                mask
            )
        )

        if count == 0:
            continue

        mean_confidence = float(
            np.mean(
                confidences[
                    mask
                ]
            )
        )

        mean_accuracy = float(
            np.mean(
                correctness[
                    mask
                ]
            )
        )

        error += (
            count
            / total_rows
        ) * abs(
            mean_accuracy
            - mean_confidence
        )

    return float(
        error
    )


def classwise_calibration_error(
    actual_labels: Iterable[str],
    probabilities: np.ndarray,
    bins: int = ECE_BINS,
) -> Dict[str, float]:
    labels = np.asarray(
        list(
            actual_labels
        ),
        dtype=str,
    )

    normalized = normalize_probabilities(
        probabilities
    )

    boundaries = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    errors: Dict[
        str,
        float,
    ] = {}

    for class_index, label in enumerate(
        LABEL_ORDER
    ):
        class_probabilities = normalized[
            :,
            class_index,
        ]

        class_targets = (
            labels == label
        ).astype(
            float
        )

        class_error = 0.0

        for bin_index in range(
            bins
        ):
            lower = boundaries[
                bin_index
            ]

            upper = boundaries[
                bin_index + 1
            ]

            if (
                bin_index
                == bins - 1
            ):
                mask = (
                    class_probabilities
                    >= lower
                ) & (
                    class_probabilities
                    <= upper
                )

            else:
                mask = (
                    class_probabilities
                    >= lower
                ) & (
                    class_probabilities
                    < upper
                )

            count = int(
                np.sum(
                    mask
                )
            )

            if count == 0:
                continue

            mean_probability = float(
                np.mean(
                    class_probabilities[
                        mask
                    ]
                )
            )

            observed_frequency = float(
                np.mean(
                    class_targets[
                        mask
                    ]
                )
            )

            class_error += (
                count
                / len(labels)
            ) * abs(
                observed_frequency
                - mean_probability
            )

        errors[
            label
        ] = float(
            class_error
        )

    return errors


def evaluate_probabilities(
    actual_labels: Iterable[str],
    probabilities: np.ndarray,
) -> Dict[str, object]:
    labels = np.asarray(
        list(
            actual_labels
        ),
        dtype=str,
    )

    normalized = normalize_probabilities(
        probabilities
    )

    predicted_labels = (
        probabilities_to_labels(
            normalized
        )
    )

    report = classification_report(
        labels,
        predicted_labels,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predicted_labels,
        labels=LABEL_ORDER,
    )

    classwise_error = (
        classwise_calibration_error(
            actual_labels=labels,
            probabilities=normalized,
        )
    )

    return {
        "match_count": int(
            len(labels)
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predicted_labels,
            )
        ),
        "log_loss": float(
            log_loss(
                labels,
                normalized,
                labels=LABEL_ORDER,
            )
        ),
        "brier_score": (
            multiclass_brier_score(
                actual_labels=labels,
                probabilities=normalized,
            )
        ),
        "ece": (
            expected_calibration_error(
                actual_labels=labels,
                probabilities=normalized,
            )
        ),
        "classwise_ece": (
            classwise_error
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predicted_labels,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predicted_labels,
                labels=LABEL_ORDER,
                average="weighted",
                zero_division=0,
            )
        ),
        "draw_recall": float(
            recall_score(
                labels,
                predicted_labels,
                labels=[
                    "D",
                ],
                average=None,
                zero_division=0,
            )[0]
        ),
        "away_recall": float(
            report["A"]["recall"]
        ),
        "home_recall": float(
            report["H"]["recall"]
        ),
        "predicted_away_wins": int(
            np.sum(
                predicted_labels
                == "A"
            )
        ),
        "predicted_draws": int(
            np.sum(
                predicted_labels
                == "D"
            )
        ),
        "predicted_home_wins": int(
            np.sum(
                predicted_labels
                == "H"
            )
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "classification_report": (
            report
        ),
    }


def build_ml_probabilities(
    matches: pd.DataFrame,
    test_data: pd.DataFrame,
) -> np.ndarray:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Production model was not found: "
            f"{MODEL_PATH}"
        )

    builder = FeatureBuilder(
        initial_elo=INITIAL_ELO,
        k_factor=K_FACTOR,
        home_advantage=HOME_ADVANTAGE,
        form_window=FORM_WINDOW,
    )

    print(
        "Generating production ML features..."
    )

    dataset = builder.build(
        matches
    )

    dataset = dataset.set_index(
        "match_id",
        drop=False,
    )

    test_ids = (
        test_data[
            "match_id"
        ].tolist()
    )

    missing_ids = [
        match_id
        for match_id in test_ids
        if match_id
        not in dataset.index
    ]

    if missing_ids:
        raise RuntimeError(
            "ML features are missing for "
            f"{len(missing_ids)} matches."
        )

    features = (
        dataset.loc[
            test_ids,
            FeatureBuilder.FEATURE_COLUMNS,
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    model = MachineLearningModel()

    model.load(
        str(
            MODEL_PATH
        )
    )

    probabilities = (
        model.predict_proba(
            features
        )
    )

    return normalize_probabilities(
        probabilities
    )


def build_poisson_probabilities(
    matches: pd.DataFrame,
    test_match_ids: set,
) -> Dict[object, List[float]]:
    ratings: Dict[
        str,
        float,
    ] = defaultdict(
        lambda: INITIAL_ELO
    )

    histories: Dict[
        str,
        Deque[HistoryItem],
    ] = defaultdict(
        lambda: deque(
            maxlen=FORM_WINDOW
        )
    )

    service = PoissonSimulationService(
        simulations=POISSON_SIMULATIONS,
        random_seed=RANDOM_SEED,
    )

    probabilities_by_id: Dict[
        object,
        List[float],
    ] = {}

    for row in matches.itertuples(
        index=False
    ):
        match_id = row.match_id

        home_team = str(
            row.home_team
        )

        away_team = str(
            row.away_team
        )

        home_goals = int(
            row.home_goals
        )

        away_goals = int(
            row.away_goals
        )

        home_summary = history_summary(
            histories[
                home_team
            ]
        )

        away_summary = history_summary(
            histories[
                away_team
            ]
        )

        home_elo = float(
            ratings[
                home_team
            ]
        )

        away_elo = float(
            ratings[
                away_team
            ]
        )

        if match_id in test_match_ids:
            simulation = service.simulate(
                home_summary=home_summary,
                away_summary=away_summary,
                home_elo=home_elo,
                away_elo=away_elo,
            )

            probabilities_by_id[
                match_id
            ] = [
                simulation
                .away_win_probability,

                simulation
                .draw_probability,

                simulation
                .home_win_probability,
            ]

        histories[
            home_team
        ].append(
            (
                points_for_result(
                    goals_scored=home_goals,
                    goals_conceded=away_goals,
                ),
                home_goals,
                away_goals,
            )
        )

        histories[
            away_team
        ].append(
            (
                points_for_result(
                    goals_scored=away_goals,
                    goals_conceded=home_goals,
                ),
                away_goals,
                home_goals,
            )
        )

        update_elo(
            ratings=ratings,
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
        )

    return probabilities_by_id


def print_results(
    results: Dict[
        str,
        Dict[str, object],
    ],
) -> None:
    print()
    print("=" * 112)
    print(
        "FINAL CALIBRATION TEST RESULTS"
    )
    print("=" * 112)

    print(
        f"{'Method':<22}"
        f"{'Accuracy':>11}"
        f"{'Log loss':>12}"
        f"{'Brier':>11}"
        f"{'ECE':>10}"
        f"{'Macro F1':>12}"
        f"{'Draw recall':>14}"
        f"{'Matches':>10}"
    )

    print(
        "-" * 112
    )

    sorted_results = sorted(
        results.items(),
        key=lambda item: (
            item[1][
                "log_loss"
            ],
            item[1][
                "brier_score"
            ],
            item[1][
                "ece"
            ],
        ),
    )

    for method, metrics in sorted_results:
        print(
            f"{method:<22}"
            f"{metrics['accuracy']:>10.2%}"
            f"{metrics['log_loss']:>12.4f}"
            f"{metrics['brier_score']:>11.4f}"
            f"{metrics['ece']:>10.4f}"
            f"{metrics['macro_f1']:>12.2%}"
            f"{metrics['draw_recall']:>14.2%}"
            f"{metrics['match_count']:>10,}"
        )


def create_results_dataframe(
    results: Dict[
        str,
        Dict[str, object],
    ],
) -> pd.DataFrame:
    rows = []

    for method, metrics in results.items():
        rows.append(
            {
                "method": method,
                "match_count": metrics[
                    "match_count"
                ],
                "accuracy": metrics[
                    "accuracy"
                ],
                "log_loss": metrics[
                    "log_loss"
                ],
                "brier_score": metrics[
                    "brier_score"
                ],
                "ece": metrics[
                    "ece"
                ],
                "macro_f1": metrics[
                    "macro_f1"
                ],
                "weighted_f1": metrics[
                    "weighted_f1"
                ],
                "draw_recall": metrics[
                    "draw_recall"
                ],
                "away_recall": metrics[
                    "away_recall"
                ],
                "home_recall": metrics[
                    "home_recall"
                ],
                "away_ece": metrics[
                    "classwise_ece"
                ]["A"],
                "draw_ece": metrics[
                    "classwise_ece"
                ]["D"],
                "home_ece": metrics[
                    "classwise_ece"
                ]["H"],
            }
        )

    return pd.DataFrame(
        rows
    )


def select_recommended_method(
    results: Dict[
        str,
        Dict[str, object],
    ],
) -> str:
    baseline = results[
        "Uncalibrated hybrid"
    ]

    candidate_names = [
        "Platt calibrated",
        "Isotonic calibrated",
    ]

    qualifying_candidates = []

    for candidate_name in candidate_names:
        candidate = results[
            candidate_name
        ]

        improves_log_loss = (
            candidate[
                "log_loss"
            ]
            < baseline[
                "log_loss"
            ]
        )

        improves_brier = (
            candidate[
                "brier_score"
            ]
            < baseline[
                "brier_score"
            ]
        )

        improves_ece = (
            candidate[
                "ece"
            ]
            < baseline[
                "ece"
            ]
        )

        accuracy_not_materially_worse = (
            candidate[
                "accuracy"
            ]
            >= baseline[
                "accuracy"
            ]
            - 0.01
        )

        if (
            improves_log_loss
            and improves_brier
            and improves_ece
            and accuracy_not_materially_worse
        ):
            qualifying_candidates.append(
                candidate_name
            )

    if not qualifying_candidates:
        return "KEEP_UNCALIBRATED"

    return min(
        qualifying_candidates,
        key=lambda name: (
            results[
                name
            ]["log_loss"],
            results[
                name
            ]["brier_score"],
            results[
                name
            ]["ece"],
        ),
    )


def main(
) -> None:
    print()
    print("=" * 112)
    print(
        "50/50 HYBRID PROBABILITY "
        "CALIBRATION EXPERIMENT"
    )
    print("=" * 112)

    matches = load_matches()

    test_start = pd.Timestamp(
        TEST_START_DATE,
        tz="UTC",
    )

    cl_test_data = matches[
        (
            matches[
                "date"
            ]
            >= test_start
        )
        & (
            matches[
                "competition"
            ]
            == "CL"
        )
    ].copy()

    cl_test_data = (
        cl_test_data
        .sort_values(
            by=[
                "date",
                "match_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            cl_test_data
        )
        <= CALIBRATION_MATCH_COUNT
    ):
        raise ValueError(
            "Not enough test matches for the "
            "requested calibration split. "
            f"Available: {len(cl_test_data)}, "
            f"calibration requested: "
            f"{CALIBRATION_MATCH_COUNT}."
        )

    print(
        f"Production matches: "
        f"{len(matches):,}"
    )

    print(
        f"CL evaluation matches: "
        f"{len(cl_test_data):,}"
    )

    print(
        f"Calibration matches: "
        f"{CALIBRATION_MATCH_COUNT:,}"
    )

    print(
        "Final untouched test matches: "
        f"{len(cl_test_data) - CALIBRATION_MATCH_COUNT:,}"
    )

    ml_probabilities = build_ml_probabilities(
        matches=matches,
        test_data=cl_test_data,
    )

    print(
        "Generating Poisson probabilities..."
    )

    test_match_ids = set(
        cl_test_data[
            "match_id"
        ].tolist()
    )

    poisson_by_id = (
        build_poisson_probabilities(
            matches=matches,
            test_match_ids=(
                test_match_ids
            ),
        )
    )

    missing_ids = [
        match_id
        for match_id in cl_test_data[
            "match_id"
        ].tolist()
        if match_id
        not in poisson_by_id
    ]

    if missing_ids:
        raise RuntimeError(
            "Poisson probabilities are missing "
            f"for {len(missing_ids)} matches."
        )

    poisson_probabilities = np.asarray(
        [
            poisson_by_id[
                match_id
            ]
            for match_id
            in cl_test_data[
                "match_id"
            ].tolist()
        ],
        dtype=float,
    )

    hybrid_probabilities = (
        ML_WEIGHT
        * ml_probabilities
        + POISSON_WEIGHT
        * poisson_probabilities
    )

    hybrid_probabilities = normalize_probabilities(
        hybrid_probabilities
    )

    labels = (
        cl_test_data[
            "winner"
        ]
        .astype(str)
        .to_numpy()
    )

    calibration_probabilities = (
        hybrid_probabilities[
            :CALIBRATION_MATCH_COUNT
        ]
    )

    calibration_labels = labels[
        :CALIBRATION_MATCH_COUNT
    ]

    final_probabilities = (
        hybrid_probabilities[
            CALIBRATION_MATCH_COUNT:
        ]
    )

    final_labels = labels[
        CALIBRATION_MATCH_COUNT:
    ]

    final_match_data = (
        cl_test_data.iloc[
            CALIBRATION_MATCH_COUNT:
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    platt_calibrator = (
        MulticlassPlattCalibrator()
    )

    platt_calibrator.fit(
        probabilities=(
            calibration_probabilities
        ),
        labels=(
            calibration_labels
        ),
    )

    isotonic_calibrator = (
        MulticlassIsotonicCalibrator()
    )

    isotonic_calibrator.fit(
        probabilities=(
            calibration_probabilities
        ),
        labels=(
            calibration_labels
        ),
    )

    platt_probabilities = (
        platt_calibrator.predict_proba(
            final_probabilities
        )
    )

    isotonic_probabilities = (
        isotonic_calibrator.predict_proba(
            final_probabilities
        )
    )

    results = {
        "Uncalibrated hybrid": (
            evaluate_probabilities(
                actual_labels=final_labels,
                probabilities=(
                    final_probabilities
                ),
            )
        ),
        "Platt calibrated": (
            evaluate_probabilities(
                actual_labels=final_labels,
                probabilities=(
                    platt_probabilities
                ),
            )
        ),
        "Isotonic calibrated": (
            evaluate_probabilities(
                actual_labels=final_labels,
                probabilities=(
                    isotonic_probabilities
                ),
            )
        ),
    }

    print_results(
        results
    )

    recommendation = (
        select_recommended_method(
            results
        )
    )

    print()
    print("=" * 112)
    print(
        "CALIBRATION RECOMMENDATION"
    )
    print("=" * 112)

    print(
        f"Recommendation: "
        f"{recommendation}"
    )

    baseline = results[
        "Uncalibrated hybrid"
    ]

    if recommendation == "KEEP_UNCALIBRATED":
        print(
            "Neither calibration method improved "
            "log loss, Brier score and ECE together "
            "without an unacceptable accuracy loss."
        )

    else:
        selected = results[
            recommendation
        ]

        print(
            "Log-loss improvement: "
            f"{baseline['log_loss'] - selected['log_loss']:+.4f}"
        )

        print(
            "Brier improvement: "
            f"{baseline['brier_score'] - selected['brier_score']:+.4f}"
        )

        print(
            "ECE improvement: "
            f"{baseline['ece'] - selected['ece']:+.4f}"
        )

        print(
            "Accuracy difference: "
            f"{selected['accuracy'] - baseline['accuracy']:+.2%}"
        )

    calibrator_bundle = {
        "recommended_method": (
            recommendation
        ),
        "platt_calibrator": (
            platt_calibrator
        ),
        "isotonic_calibrator": (
            isotonic_calibrator
        ),
        "label_order": (
            LABEL_ORDER
        ),
        "ml_weight": (
            ML_WEIGHT
        ),
        "poisson_weight": (
            POISSON_WEIGHT
        ),
        "calibration_match_count": (
            CALIBRATION_MATCH_COUNT
        ),
        "calibration_start": (
            cl_test_data.iloc[
                0
            ]["date"].isoformat()
        ),
        "calibration_end": (
            cl_test_data.iloc[
                CALIBRATION_MATCH_COUNT
                - 1
            ]["date"].isoformat()
        ),
        "final_test_start": (
            final_match_data.iloc[
                0
            ]["date"].isoformat()
        ),
        "final_test_end": (
            final_match_data.iloc[
                -1
            ]["date"].isoformat()
        ),
    }

    CALIBRATOR_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        calibrator_bundle,
        CALIBRATOR_PATH,
    )

    report = {
        "experiment_name": (
            "hybrid_probability_calibration"
        ),
        "data_path": str(
            DATA_PATH
        ),
        "model_path": str(
            MODEL_PATH
        ),
        "calibrator_path": str(
            CALIBRATOR_PATH
        ),
        "ml_weight": (
            ML_WEIGHT
        ),
        "poisson_weight": (
            POISSON_WEIGHT
        ),
        "total_cl_matches": int(
            len(
                cl_test_data
            )
        ),
        "calibration_match_count": int(
            CALIBRATION_MATCH_COUNT
        ),
        "final_test_match_count": int(
            len(
                final_labels
            )
        ),
        "calibration_start": (
            cl_test_data.iloc[
                0
            ]["date"].isoformat()
        ),
        "calibration_end": (
            cl_test_data.iloc[
                CALIBRATION_MATCH_COUNT
                - 1
            ]["date"].isoformat()
        ),
        "final_test_start": (
            final_match_data.iloc[
                0
            ]["date"].isoformat()
        ),
        "final_test_end": (
            final_match_data.iloc[
                -1
            ]["date"].isoformat()
        ),
        "recommendation": (
            recommendation
        ),
        "results": (
            results
        ),
    }

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

    results_dataframe = (
        create_results_dataframe(
            results
        )
    )

    results_dataframe.to_csv(
        RESULTS_CSV_PATH,
        index=False,
    )

    print()
    print(
        "Candidate calibrator saved to:"
    )

    print(
        CALIBRATOR_PATH
    )

    print()
    print(
        "JSON report saved to:"
    )

    print(
        REPORT_PATH
    )

    print()
    print(
        "CSV results saved to:"
    )

    print(
        RESULTS_CSV_PATH
    )

    print("=" * 112)


if __name__ == "__main__":
    main()
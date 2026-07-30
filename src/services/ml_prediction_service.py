from pathlib import Path

import pandas as pd

from src.features.live_feature_builder import (
    LiveFeatureBuilder,
)
from src.models.machine_learning_model import (
    MachineLearningModel,
)
from src.models.ml_prediction import (
    MLPrediction,
)
from src.services.poisson_simulation_service import (
    PoissonSimulationService,
)
from src.services.prediction_explanation_service import (
    PredictionExplanationService,
)


class MLPredictionService:
    LABEL_ORDER = ["A", "D", "H"]

    def __init__(
        self,
        data_path: str,
        model_path: str,
    ) -> None:
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)

        self.feature_builder = LiveFeatureBuilder(
            initial_elo=1500,
            k_factor=25,
            home_advantage=60,
            form_window=8,
        )

        self.model = MachineLearningModel()

        self.explanation_service = (
            PredictionExplanationService()
        )

        self.poisson_service = (
            PoissonSimulationService(
                simulations=20_000,
                random_seed=42,
            )
        )

        self.is_loaded = False

    def load(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(
                "Data file not found: "
                f"{self.data_path}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Model file not found: "
                f"{self.model_path}"
            )

        matches = pd.read_csv(
            self.data_path
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
                "Missing data columns: "
                + ", ".join(
                    sorted(
                        missing_columns
                    )
                )
            )

        self.feature_builder.fit(
            matches
        )

        self.model.load(
            str(self.model_path)
        )

        self.is_loaded = True

    def predict(
        self,
        home_team: str,
        away_team: str,
    ) -> MLPrediction:
        self._ensure_loaded()

        features = (
            self.feature_builder
            .build_match_features(
                home_team=home_team,
                away_team=away_team,
                competition="CL",
            )
        )

        probabilities = (
            self.model.predict_proba(
                features
            )[0]
        )

        away_probability = float(
            probabilities[0]
        )

        draw_probability = float(
            probabilities[1]
        )

        home_probability = float(
            probabilities[2]
        )

        probability_map = {
            "A": away_probability,
            "D": draw_probability,
            "H": home_probability,
        }

        predicted_result = max(
            probability_map,
            key=probability_map.get,
        )

        confidence = float(
            probability_map[
                predicted_result
            ]
        )

        home_summary = (
            self.feature_builder
            .get_team_summary(
                home_team
            )
        )

        away_summary = (
            self.feature_builder
            .get_team_summary(
                away_team
            )
        )

        home_elo = float(
            home_summary["elo"]
        )

        away_elo = float(
            away_summary["elo"]
        )

        explanation = (
            self.explanation_service
            .explain(
                feature_row=features,
                predicted_result=(
                    predicted_result
                ),
                home_team=home_team,
                away_team=away_team,
                top_n=5,
            )
        )

        simulation = (
            self.poisson_service.simulate(
                home_summary=home_summary,
                away_summary=away_summary,
                home_elo=home_elo,
                away_elo=away_elo,
            )
        )

        return MLPrediction(
            home_team=home_team,
            away_team=away_team,

            home_win_probability=(
                home_probability
            ),
            draw_probability=(
                draw_probability
            ),
            away_win_probability=(
                away_probability
            ),

            predicted_result=(
                predicted_result
            ),
            confidence=confidence,
            confidence_label=(
                self._confidence_label(
                    confidence
                )
            ),

            home_elo=home_elo,
            away_elo=away_elo,

            poisson_home_probability=(
                simulation
                .home_win_probability
            ),
            poisson_draw_probability=(
                simulation
                .draw_probability
            ),
            poisson_away_probability=(
                simulation
                .away_win_probability
            ),

            btts_probability=(
                simulation
                .btts_probability
            ),

            over_1_5_probability=(
                simulation
                .over_1_5_probability
            ),
            over_2_5_probability=(
                simulation
                .over_2_5_probability
            ),
            under_2_5_probability=(
                simulation
                .under_2_5_probability
            ),
            under_3_5_probability=(
                simulation
                .under_3_5_probability
            ),

            expected_home_goals=(
                simulation
                .expected_home_goals
            ),
            expected_away_goals=(
                simulation
                .expected_away_goals
            ),

            most_likely_home_goals=(
                simulation
                .most_likely_home_goals
            ),
            most_likely_away_goals=(
                simulation
                .most_likely_away_goals
            ),

            positive_factors=(
                explanation[
                    "positive_factors"
                ]
            ),
            negative_factors=(
                explanation[
                    "negative_factors"
                ]
            ),
        )

    def get_teams(self) -> list:
        self._ensure_loaded()

        return (
            self.feature_builder
            .get_teams()
        )

    def get_team_summary(
        self,
        team_name: str,
    ) -> dict:
        self._ensure_loaded()

        return (
            self.feature_builder
            .get_team_summary(
                team_name
            )
        )

    @staticmethod
    def _confidence_label(
        confidence: float,
    ) -> str:
        if confidence >= 0.70:
            return "HIGH"

        if confidence >= 0.55:
            return "MEDIUM"

        return "LOW"

    def _ensure_loaded(
        self,
    ) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                "ML prediction service "
                "must be loaded first."
            )
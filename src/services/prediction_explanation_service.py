from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.ml_prediction import (
    PredictionFactor,
)


class PredictionExplanationService:
    """
    Creates human-readable explanations from
    the same live features supplied to the model.

    These explanations describe important match
    factors. They are not SHAP values and should
    not be interpreted as exact causal effects.
    """

    def explain(
        self,
        feature_row: pd.DataFrame,
        predicted_result: str,
        home_team: str,
        away_team: str,
        top_n: int = 5,
    ) -> Dict[
        str,
        List[PredictionFactor],
    ]:
        if feature_row.empty:
            raise ValueError(
                "feature_row cannot be empty."
            )

        features = feature_row.iloc[0]

        candidates: List[
            PredictionFactor
        ] = []

        candidates.extend(
            self._elo_factors(
                features=features,
                predicted_result=(
                    predicted_result
                ),
                home_team=home_team,
                away_team=away_team,
            )
        )

        candidates.extend(
            self._form_factors(
                features=features,
                predicted_result=(
                    predicted_result
                ),
                home_team=home_team,
                away_team=away_team,
            )
        )

        candidates.extend(
            self._attack_factors(
                features=features,
                predicted_result=(
                    predicted_result
                ),
                home_team=home_team,
                away_team=away_team,
            )
        )

        candidates.extend(
            self._goal_difference_factors(
                features=features,
                predicted_result=(
                    predicted_result
                ),
                home_team=home_team,
                away_team=away_team,
            )
        )

        candidates.extend(
            self._win_rate_factors(
                features=features,
                predicted_result=(
                    predicted_result
                ),
                home_team=home_team,
                away_team=away_team,
            )
        )

        candidates.extend(
            self._rest_factors(
                features=features,
                predicted_result=(
                    predicted_result
                ),
                home_team=home_team,
                away_team=away_team,
            )
        )

        candidates.extend(
            self._draw_factors(
                features=features,
                predicted_result=(
                    predicted_result
                ),
            )
        )

        positive_factors = sorted(
            [
                factor
                for factor in candidates
                if factor.supports_prediction
            ],
            key=lambda factor: (
                factor.strength
            ),
            reverse=True,
        )[:top_n]

        negative_factors = sorted(
            [
                factor
                for factor in candidates
                if not factor.supports_prediction
            ],
            key=lambda factor: (
                factor.strength
            ),
            reverse=True,
        )[:top_n]

        return {
            "positive_factors": (
                positive_factors
            ),
            "negative_factors": (
                negative_factors
            ),
        }

    def _elo_factors(
        self,
        features: pd.Series,
        predicted_result: str,
        home_team: str,
        away_team: str,
    ) -> List[PredictionFactor]:
        elo_difference = self._value(
            features,
            "elo_difference",
        )

        absolute_difference = abs(
            elo_difference
        )

        if elo_difference > 0:
            favoured_team = home_team

        elif elo_difference < 0:
            favoured_team = away_team

        else:
            favoured_team = "Neither team"

        supports_prediction = (
            (
                predicted_result == "H"
                and elo_difference > 0
            )
            or (
                predicted_result == "A"
                and elo_difference < 0
            )
            or (
                predicted_result == "D"
                and absolute_difference < 75
            )
        )

        if predicted_result == "D":
            explanation = (
                "The Elo ratings are close, "
                "which makes a draw more plausible."
                if absolute_difference < 75
                else
                "The Elo ratings show a meaningful "
                "strength difference, which works "
                "against the draw prediction."
            )

        else:
            explanation = (
                f"{favoured_team} has an estimated "
                f"Elo advantage of approximately "
                f"{absolute_difference:.0f} points."
            )

        return [
            PredictionFactor(
                title="Elo strength",
                explanation=explanation,
                strength=min(
                    absolute_difference / 200.0,
                    2.0,
                ),
                supports_prediction=(
                    supports_prediction
                ),
            )
        ]

    def _form_factors(
        self,
        features: pd.Series,
        predicted_result: str,
        home_team: str,
        away_team: str,
    ) -> List[PredictionFactor]:
        difference = self._value(
            features,
            "form_difference",
        )

        if difference > 0:
            stronger_team = home_team

        elif difference < 0:
            stronger_team = away_team

        else:
            stronger_team = None

        if predicted_result == "D":
            supports_prediction = (
                abs(difference) < 0.30
            )

            explanation = (
                "The teams have similar recent "
                "points-per-match form."
                if supports_prediction
                else
                "The teams have noticeably "
                "different recent form."
            )

        else:
            supports_prediction = (
                (
                    predicted_result == "H"
                    and difference > 0
                )
                or (
                    predicted_result == "A"
                    and difference < 0
                )
            )

            if stronger_team is None:
                explanation = (
                    "The teams have almost identical "
                    "recent form."
                )

            else:
                explanation = (
                    f"{stronger_team} has the better "
                    "recent points-per-match record."
                )

        return [
            PredictionFactor(
                title="Recent form",
                explanation=explanation,
                strength=min(
                    abs(difference),
                    2.0,
                ),
                supports_prediction=(
                    supports_prediction
                ),
            )
        ]

    def _attack_factors(
        self,
        features: pd.Series,
        predicted_result: str,
        home_team: str,
        away_team: str,
    ) -> List[PredictionFactor]:
        difference = self._value(
            features,
            "attack_matchup_difference",
        )

        if difference > 0:
            stronger_team = home_team

        elif difference < 0:
            stronger_team = away_team

        else:
            stronger_team = None

        if predicted_result == "D":
            supports_prediction = (
                abs(difference) < 0.25
            )

            explanation = (
                "The attacking matchup is balanced."
                if supports_prediction
                else
                "One team has a clearer attacking "
                "matchup advantage."
            )

        else:
            supports_prediction = (
                (
                    predicted_result == "H"
                    and difference > 0
                )
                or (
                    predicted_result == "A"
                    and difference < 0
                )
            )

            if stronger_team is None:
                explanation = (
                    "The attacking matchup is "
                    "approximately balanced."
                )

            else:
                explanation = (
                    f"{stronger_team} has the stronger "
                    "recent attack-versus-defence "
                    "matchup."
                )

        return [
            PredictionFactor(
                title="Attacking matchup",
                explanation=explanation,
                strength=min(
                    abs(difference),
                    2.0,
                ),
                supports_prediction=(
                    supports_prediction
                ),
            )
        ]

    def _goal_difference_factors(
        self,
        features: pd.Series,
        predicted_result: str,
        home_team: str,
        away_team: str,
    ) -> List[PredictionFactor]:
        home_goal_difference = self._value(
            features,
            "home_goal_difference",
        )

        away_goal_difference = self._value(
            features,
            "away_goal_difference",
        )

        difference = (
            home_goal_difference
            - away_goal_difference
        )

        if difference > 0:
            stronger_team = home_team

        elif difference < 0:
            stronger_team = away_team

        else:
            stronger_team = None

        if predicted_result == "D":
            supports_prediction = (
                abs(difference) < 0.25
            )

            explanation = (
                "The teams have similar recent "
                "goal-difference records."
                if supports_prediction
                else
                "Recent goal-difference records "
                "favour one side."
            )

        else:
            supports_prediction = (
                (
                    predicted_result == "H"
                    and difference > 0
                )
                or (
                    predicted_result == "A"
                    and difference < 0
                )
            )

            if stronger_team is None:
                explanation = (
                    "The teams have equal recent "
                    "goal-difference records."
                )

            else:
                explanation = (
                    f"{stronger_team} has the better "
                    "recent goal difference."
                )

        return [
            PredictionFactor(
                title="Recent goal difference",
                explanation=explanation,
                strength=min(
                    abs(difference),
                    2.0,
                ),
                supports_prediction=(
                    supports_prediction
                ),
            )
        ]

    def _win_rate_factors(
        self,
        features: pd.Series,
        predicted_result: str,
        home_team: str,
        away_team: str,
    ) -> List[PredictionFactor]:
        home_win_rate = self._value(
            features,
            "home_win_rate",
        )

        away_win_rate = self._value(
            features,
            "away_win_rate",
        )

        difference = (
            home_win_rate
            - away_win_rate
        )

        if difference > 0:
            stronger_team = home_team

        elif difference < 0:
            stronger_team = away_team

        else:
            stronger_team = None

        if predicted_result == "D":
            supports_prediction = (
                abs(difference) < 0.10
            )

            explanation = (
                "Recent win rates are similar."
                if supports_prediction
                else
                "Recent win rates favour one team."
            )

        else:
            supports_prediction = (
                (
                    predicted_result == "H"
                    and difference > 0
                )
                or (
                    predicted_result == "A"
                    and difference < 0
                )
            )

            if stronger_team is None:
                explanation = (
                    "The teams have equal recent "
                    "win rates."
                )

            else:
                explanation = (
                    f"{stronger_team} has the higher "
                    "recent win rate."
                )

        return [
            PredictionFactor(
                title="Recent win rate",
                explanation=explanation,
                strength=min(
                    abs(difference) * 3.0,
                    2.0,
                ),
                supports_prediction=(
                    supports_prediction
                ),
            )
        ]

    def _rest_factors(
        self,
        features: pd.Series,
        predicted_result: str,
        home_team: str,
        away_team: str,
    ) -> List[PredictionFactor]:
        home_rest_days = self._value(
            features,
            "home_rest_days",
        )

        away_rest_days = self._value(
            features,
            "away_rest_days",
        )

        difference = (
            home_rest_days
            - away_rest_days
        )

        if abs(difference) < 2:
            return []

        if difference > 0:
            better_rested_team = home_team

        else:
            better_rested_team = away_team

        supports_prediction = (
            (
                predicted_result == "H"
                and difference > 0
            )
            or (
                predicted_result == "A"
                and difference < 0
            )
        )

        if predicted_result == "D":
            supports_prediction = False

        return [
            PredictionFactor(
                title="Rest advantage",
                explanation=(
                    f"{better_rested_team} has "
                    f"approximately "
                    f"{abs(difference):.0f} more "
                    "rest days."
                ),
                strength=min(
                    abs(difference) / 5.0,
                    2.0,
                ),
                supports_prediction=(
                    supports_prediction
                ),
            )
        ]

    def _draw_factors(
        self,
        features: pd.Series,
        predicted_result: str,
    ) -> List[PredictionFactor]:
        close_elo_match = self._value(
            features,
            "close_elo_match",
        )

        if close_elo_match < 0.5:
            return []

        return [
            PredictionFactor(
                title="Evenly matched teams",
                explanation=(
                    "The Elo ratings classify this "
                    "as a relatively close matchup."
                ),
                strength=1.0,
                supports_prediction=(
                    predicted_result == "D"
                ),
            )
        ]

    @staticmethod
    def _value(
        features: pd.Series,
        feature_name: str,
    ) -> float:
        if feature_name not in features:
            return 0.0

        try:
            value = float(
                features[feature_name]
            )

        except (TypeError, ValueError):
            return 0.0

        if not np.isfinite(value):
            return 0.0

        return value
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.models.prediction_history import (
    PredictionHistoryRecord,
)
from src.repositories.prediction_history_repository import (
    PredictionHistoryRepository,
)


class PredictionHistoryService:
    def __init__(
        self,
        repository: (
            PredictionHistoryRepository
        ),
    ) -> None:
        self.repository = repository

    def save_analysis(
        self,
        event,
        competition: str,
        prediction,
        value_report,
        model_home_team: str,
        model_away_team: str,
    ) -> int:
        hybrid_probabilities = (
            self._hybrid_probabilities(
                prediction
            )
        )

        (
            hybrid_home_probability,
            hybrid_draw_probability,
            hybrid_away_probability,
        ) = hybrid_probabilities

        probability_map = {
            "H": (
                hybrid_home_probability
            ),
            "D": (
                hybrid_draw_probability
            ),
            "A": (
                hybrid_away_probability
            ),
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

        best_bet = (
            value_report.best_opportunity
            if (
                value_report is not None
                and value_report
                .best_opportunity
                is not None
            )
            else None
        )

        record = PredictionHistoryRecord(
            prediction_id=None,

            event_id=str(
                event.event_id
            ),
            created_at=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            competition=str(
                competition
            ),
            kickoff_time=str(
                event.commence_time
            ),

            api_home_team=str(
                event.home_team
            ),
            api_away_team=str(
                event.away_team
            ),

            model_home_team=str(
                model_home_team
            ),
            model_away_team=str(
                model_away_team
            ),

            ml_home_probability=float(
                prediction
                .home_win_probability
            ),
            ml_draw_probability=float(
                prediction
                .draw_probability
            ),
            ml_away_probability=float(
                prediction
                .away_win_probability
            ),

            poisson_home_probability=float(
                prediction
                .poisson_home_probability
            ),
            poisson_draw_probability=float(
                prediction
                .poisson_draw_probability
            ),
            poisson_away_probability=float(
                prediction
                .poisson_away_probability
            ),

            hybrid_home_probability=float(
                hybrid_home_probability
            ),
            hybrid_draw_probability=float(
                hybrid_draw_probability
            ),
            hybrid_away_probability=float(
                hybrid_away_probability
            ),

            predicted_result=(
                predicted_result
            ),
            predicted_result_text=(
                self._result_text(
                    predicted_result=(
                        predicted_result
                    ),
                    home_team=(
                        event.home_team
                    ),
                    away_team=(
                        event.away_team
                    ),
                )
            ),
            confidence=confidence,
            confidence_label=(
                self._confidence_label(
                    confidence
                )
            ),

            expected_home_goals=float(
                prediction
                .expected_home_goals
            ),
            expected_away_goals=float(
                prediction
                .expected_away_goals
            ),

            most_likely_home_goals=int(
                prediction
                .most_likely_home_goals
            ),
            most_likely_away_goals=int(
                prediction
                .most_likely_away_goals
            ),

            recommended_market=(
                None
                if best_bet is None
                else str(
                    best_bet.market_name
                )
            ),
            recommended_selection=(
                None
                if best_bet is None
                else str(
                    best_bet.selection
                )
            ),
            recommended_odds=(
                None
                if best_bet is None
                else float(
                    best_bet
                    .decimal_odds
                )
            ),
            bookmaker=(
                None
                if best_bet is None
                else str(
                    best_bet.bookmaker
                )
            ),

            value_model_probability=(
                None
                if best_bet is None
                else float(
                    best_bet
                    .model_probability
                )
            ),
            market_probability=(
                None
                if best_bet is None
                else float(
                    best_bet
                    .fair_market_probability
                )
            ),
            edge=(
                None
                if best_bet is None
                else float(
                    best_bet.edge
                )
            ),
            expected_value=(
                None
                if best_bet is None
                else float(
                    best_bet
                    .expected_value
                )
            ),

            actual_home_goals=None,
            actual_away_goals=None,
            actual_result=None,

            prediction_correct=None,
            bet_won=None,
            profit_loss=None,

            status="PENDING",
        )

        return self.repository.insert(
            record
        )

    def settle_prediction(
        self,
        prediction_id: int,
        actual_home_goals: int,
        actual_away_goals: int,
    ) -> None:
        dataframe = (
            self.repository
            .list_predictions(
                limit=10_000
            )
        )

        selected_rows = dataframe[
            dataframe[
                "prediction_id"
            ]
            == int(
                prediction_id
            )
        ]

        if selected_rows.empty:
            raise ValueError(
                "Prediction history record "
                "was not found."
            )

        row = selected_rows.iloc[
            0
        ]

        actual_result = (
            self._actual_result(
                home_goals=(
                    actual_home_goals
                ),
                away_goals=(
                    actual_away_goals
                ),
            )
        )

        prediction_correct = (
            str(
                row[
                    "predicted_result"
                ]
            )
            == actual_result
        )

        recommended_selection = row[
            "recommended_selection"
        ]

        recommended_market = row[
            "recommended_market"
        ]

        recommended_odds = row[
            "recommended_odds"
        ]

        bet_won = self._evaluate_bet(
            market_name=(
                recommended_market
            ),
            selection=(
                recommended_selection
            ),
            api_home_team=str(
                row[
                    "api_home_team"
                ]
            ),
            api_away_team=str(
                row[
                    "api_away_team"
                ]
            ),
            actual_home_goals=(
                actual_home_goals
            ),
            actual_away_goals=(
                actual_away_goals
            ),
        )

        profit_loss: Optional[
            float
        ]

        if (
            bet_won is None
            or pd.isna(
                recommended_odds
            )
        ):
            profit_loss = None

        elif bet_won:
            profit_loss = (
                float(
                    recommended_odds
                )
                - 1.0
            )

        else:
            profit_loss = -1.0

        self.repository.update_result(
            prediction_id=(
                prediction_id
            ),
            actual_home_goals=(
                actual_home_goals
            ),
            actual_away_goals=(
                actual_away_goals
            ),
            actual_result=(
                actual_result
            ),
            prediction_correct=(
                prediction_correct
            ),
            bet_won=bet_won,
            profit_loss=(
                profit_loss
            ),
        )

    def list_predictions(
        self,
        status: Optional[str] = None,
        competition: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        return (
            self.repository
            .list_predictions(
                status=status,
                competition=(
                    competition
                ),
                limit=limit,
            )
        )

    def summary(
        self,
    ) -> dict:
        return (
            self.repository.summary()
        )

    def list_competitions(
        self,
    ) -> list[str]:
        return (
            self.repository
            .list_competitions()
        )

    def delete_prediction(
        self,
        prediction_id: int,
    ) -> None:
        self.repository.delete(
            prediction_id
        )

    @staticmethod
    def _hybrid_probabilities(
        prediction,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        home_probability = (
            float(
                prediction
                .home_win_probability
            )
            + float(
                prediction
                .poisson_home_probability
            )
        ) / 2.0

        draw_probability = (
            float(
                prediction
                .draw_probability
            )
            + float(
                prediction
                .poisson_draw_probability
            )
        ) / 2.0

        away_probability = (
            float(
                prediction
                .away_win_probability
            )
            + float(
                prediction
                .poisson_away_probability
            )
        ) / 2.0

        total = (
            home_probability
            + draw_probability
            + away_probability
        )

        if total <= 0:
            return (
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
            )

        return (
            home_probability / total,
            draw_probability / total,
            away_probability / total,
        )

    @staticmethod
    def _result_text(
        predicted_result: str,
        home_team: str,
        away_team: str,
    ) -> str:
        if predicted_result == "H":
            return (
                f"{home_team} wins"
            )

        if predicted_result == "A":
            return (
                f"{away_team} wins"
            )

        return "Draw"

    @staticmethod
    def _confidence_label(
        confidence: float,
    ) -> str:
        if confidence >= 0.70:
            return "HIGH"

        if confidence >= 0.55:
            return "MEDIUM"

        return "LOW"

    @staticmethod
    def _actual_result(
        home_goals: int,
        away_goals: int,
    ) -> str:
        if home_goals > away_goals:
            return "H"

        if away_goals > home_goals:
            return "A"

        return "D"

    @staticmethod
    def _evaluate_bet(
        market_name,
        selection,
        api_home_team: str,
        api_away_team: str,
        actual_home_goals: int,
        actual_away_goals: int,
    ) -> Optional[bool]:
        if (
            market_name is None
            or selection is None
            or pd.isna(
                market_name
            )
            or pd.isna(
                selection
            )
        ):
            return None

        normalized_market = str(
            market_name
        ).strip().casefold()

        normalized_selection = str(
            selection
        ).strip().casefold()

        total_goals = (
            int(
                actual_home_goals
            )
            + int(
                actual_away_goals
            )
        )

        if (
            "match result"
            in normalized_market
            or normalized_market
            == "h2h"
        ):
            if (
                normalized_selection
                == "draw"
            ):
                return (
                    actual_home_goals
                    == actual_away_goals
                )

            if normalized_selection in {
                api_home_team
                .strip()
                .casefold(),
                "home",
                "home win",
            }:
                return (
                    actual_home_goals
                    > actual_away_goals
                )

            if normalized_selection in {
                api_away_team
                .strip()
                .casefold(),
                "away",
                "away win",
            }:
                return (
                    actual_away_goals
                    > actual_home_goals
                )

            return None

        if (
            "total"
            in normalized_market
            or "goal"
            in normalized_market
        ):
            if (
                "over 1.5"
                in normalized_selection
            ):
                return total_goals >= 2

            if (
                "over 2.5"
                in normalized_selection
            ):
                return total_goals >= 3

            if (
                "under 2.5"
                in normalized_selection
            ):
                return total_goals <= 2

            if (
                "under 3.5"
                in normalized_selection
            ):
                return total_goals <= 3

        return None
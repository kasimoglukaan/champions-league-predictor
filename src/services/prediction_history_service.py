from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Optional
import unicodedata

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
            sport_key=str(
                event.sport_key
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

        prediction_id = (
            self.repository.insert(
                record
            )
        )

        self.repository.replace_market_probabilities(
            prediction_id=prediction_id,
            probabilities=(
                self._all_model_probabilities(
                    prediction=prediction,
                    home_team=str(
                        event.home_team
                    ),
                    away_team=str(
                        event.away_team
                    ),
                    hybrid_home_probability=(
                        hybrid_home_probability
                    ),
                    hybrid_draw_probability=(
                        hybrid_draw_probability
                    ),
                    hybrid_away_probability=(
                        hybrid_away_probability
                    ),
                )
            ),
        )

        return prediction_id

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

        probability_frame = (
            self.repository
            .list_market_probabilities(
                prediction_id=prediction_id
            )
        )

        probability_outcomes = []

        for probability_row in (
            probability_frame.itertuples(
                index=False
            )
        ):
            probability_outcomes.append(
                {
                    "market_key": (
                        probability_row.market_key
                    ),
                    "selection": (
                        probability_row.selection
                    ),
                    "outcome_correct": (
                        self._evaluate_model_probability(
                            market_key=str(
                                probability_row.market_key
                            ),
                            selection=str(
                                probability_row.selection
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
                    ),
                }
            )

        if probability_outcomes:
            self.repository.update_market_probability_outcomes(
                prediction_id=prediction_id,
                outcomes=probability_outcomes,
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


    def list_market_probabilities(
        self,
        prediction_id: Optional[int] = None,
    ) -> pd.DataFrame:
        return (
            self.repository
            .list_market_probabilities(
                prediction_id=prediction_id
            )
        )


    def probability_performance_records(
        self,
    ) -> pd.DataFrame:
        return (
            self.repository
            .probability_performance_records()
        )

    def market_probability_summary(
        self,
    ) -> pd.DataFrame:
        return (
            self.repository
            .market_probability_summary()
        )

    def summary(
        self,
    ) -> dict:
        summary = (
            self.repository.summary()
        )

        performance_frame = (
            self.repository
            .probability_performance_records()
        )

        secondary_total = 0
        secondary_correct = 0

        if not performance_frame.empty:
            for row in (
                performance_frame
                .itertuples(
                    index=False
                )
            ):
                probability = float(
                    row.probability
                )

                if probability < 0.50:
                    continue

                if self._is_primary_probability(
                    market_name=str(
                        row.market_name
                    ),
                    selection=str(
                        row.selection
                    ),
                    recommended_market=(
                        row.recommended_market
                    ),
                    recommended_selection=(
                        row.recommended_selection
                    ),
                ):
                    continue

                secondary_total += 1
                secondary_correct += int(
                    bool(
                        row.outcome_correct
                    )
                )

        secondary_hit_rate = (
            secondary_correct
            / secondary_total
            if secondary_total > 0
            else 0.0
        )

        primary_total = int(
            summary["won_bets"]
            + summary["lost_bets"]
        )

        primary_correct = int(
            summary["won_bets"]
        )

        overall_total = (
            primary_total
            + secondary_total
        )

        overall_correct = (
            primary_correct
            + secondary_correct
        )

        overall_forecast_hit_rate = (
            overall_correct
            / overall_total
            if overall_total > 0
            else 0.0
        )

        return {
            **summary,
            "primary_recommendation_total": (
                primary_total
            ),
            "primary_recommendation_correct": (
                primary_correct
            ),
            "primary_recommendation_hit_rate": (
                summary["bet_hit_rate"]
            ),
            "secondary_prediction_total": (
                secondary_total
            ),
            "secondary_prediction_correct": (
                secondary_correct
            ),
            "secondary_prediction_hit_rate": (
                secondary_hit_rate
            ),
            "overall_forecast_total": (
                overall_total
            ),
            "overall_forecast_correct": (
                overall_correct
            ),
            "overall_forecast_hit_rate": (
                overall_forecast_hit_rate
            ),
        }

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
    def _all_model_probabilities(
        prediction,
        home_team: str,
        away_team: str,
        hybrid_home_probability: float,
        hybrid_draw_probability: float,
        hybrid_away_probability: float,
    ) -> list[dict]:
        btts_yes = float(
            prediction.btts_probability
        )

        probabilities = [
            {
                "market_key": "h2h",
                "market_name": "Match result",
                "selection": home_team,
                "probability": float(
                    hybrid_home_probability
                ),
            },
            {
                "market_key": "h2h",
                "market_name": "Match result",
                "selection": "Draw",
                "probability": float(
                    hybrid_draw_probability
                ),
            },
            {
                "market_key": "h2h",
                "market_name": "Match result",
                "selection": away_team,
                "probability": float(
                    hybrid_away_probability
                ),
            },
            {
                "market_key": "btts",
                "market_name": (
                    "Both teams to score"
                ),
                "selection": "Yes",
                "probability": btts_yes,
            },
            {
                "market_key": "btts",
                "market_name": (
                    "Both teams to score"
                ),
                "selection": "No",
                "probability": (
                    1.0
                    - btts_yes
                ),
            },
            {
                "market_key": "totals",
                "market_name": "Total goals",
                "selection": "Over 1.5",
                "probability": float(
                    prediction
                    .over_1_5_probability
                ),
            },
            {
                "market_key": "totals",
                "market_name": "Total goals",
                "selection": "Over 2.5",
                "probability": float(
                    prediction
                    .over_2_5_probability
                ),
            },
            {
                "market_key": "totals",
                "market_name": "Total goals",
                "selection": "Under 2.5",
                "probability": float(
                    prediction
                    .under_2_5_probability
                ),
            },
            {
                "market_key": "totals",
                "market_name": "Total goals",
                "selection": "Under 3.5",
                "probability": float(
                    prediction
                    .under_3_5_probability
                ),
            },
        ]

        return [
            {
                **item,
                "probability": min(
                    max(
                        float(
                            item[
                                "probability"
                            ]
                        ),
                        0.0,
                    ),
                    1.0,
                ),
            }
            for item in probabilities
        ]

    @staticmethod
    def _evaluate_model_probability(
        market_key: str,
        selection: str,
        api_home_team: str,
        api_away_team: str,
        actual_home_goals: int,
        actual_away_goals: int,
    ) -> bool:
        normalized_market = (
            str(market_key)
            .strip()
            .casefold()
        )

        normalized_selection = (
            str(selection)
            .strip()
            .casefold()
        )

        total_goals = (
            int(actual_home_goals)
            + int(actual_away_goals)
        )

        if normalized_market == "h2h":
            if normalized_selection == "draw":
                return (
                    actual_home_goals
                    == actual_away_goals
                )

            if (
                PredictionHistoryService
                ._team_names_match(
                    selection,
                    api_home_team,
                )
            ):
                return (
                    actual_home_goals
                    > actual_away_goals
                )

            if (
                PredictionHistoryService
                ._team_names_match(
                    selection,
                    api_away_team,
                )
            ):
                return (
                    actual_away_goals
                    > actual_home_goals
                )

            return False

        if normalized_market == "btts":
            both_scored = (
                actual_home_goals > 0
                and actual_away_goals > 0
            )

            if normalized_selection == "yes":
                return both_scored

            if normalized_selection == "no":
                return not both_scored

            return False

        if normalized_market == "totals":
            if normalized_selection == "over 1.5":
                return total_goals >= 2

            if normalized_selection == "over 2.5":
                return total_goals >= 3

            if normalized_selection == "under 2.5":
                return total_goals <= 2

            if normalized_selection == "under 3.5":
                return total_goals <= 3

        return False


    @classmethod
    def _is_primary_probability(
        cls,
        market_name,
        selection: str,
        recommended_market,
        recommended_selection,
    ) -> bool:
        if (
            recommended_market is None
            or recommended_selection is None
            or pd.isna(
                recommended_market
            )
            or pd.isna(
                recommended_selection
            )
        ):
            return False

        normalized_market = (
            cls._normalize_market_text(
                market_name
            )
        )

        normalized_recommended_market = (
            cls._normalize_market_text(
                recommended_market
            )
        )

        if (
            normalized_market
            != normalized_recommended_market
        ):
            return False

        normalized_selection = (
            cls._normalize_selection_text(
                selection
            )
        )

        normalized_recommended_selection = (
            cls._normalize_selection_text(
                recommended_selection
            )
        )

        return (
            normalized_selection
            == normalized_recommended_selection
        )

    @staticmethod
    def _normalize_market_text(
        value,
    ) -> str:
        normalized = str(
            value
        ).strip().casefold()

        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            normalized,
        )

        aliases = {
            "h2h": "match result",
            "match result": "match result",
            "total goals": "total goals",
            "totals": "total goals",
            "both teams to score": "btts",
            "btts": "btts",
        }

        cleaned = " ".join(
            normalized.split()
        )

        return aliases.get(
            cleaned,
            cleaned,
        )

    @classmethod
    def _normalize_selection_text(
        cls,
        value,
    ) -> str:
        raw_value = str(
            value
        ).strip()

        normalized_team = (
            cls._normalize_team_name(
                raw_value
            )
        )

        normalized = re.sub(
            r"[^a-z0-9.]+",
            " ",
            raw_value.casefold(),
        )

        ignored_words = {
            "goal",
            "goals",
            "win",
            "wins",
        }

        normalized_words = [
            word
            for word in normalized.split()
            if word not in ignored_words
        ]

        normalized_selection = (
            " ".join(
                normalized_words
            )
            .strip()
        )

        standard_selections = {
            "draw",
            "yes",
            "no",
            "over 1.5",
            "over 2.5",
            "under 2.5",
            "under 3.5",
        }

        if normalized_selection in (
            standard_selections
        ):
            return normalized_selection

        return normalized_team

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
                "home",
                "home win",
            } or PredictionHistoryService._team_names_match(
                selection,
                api_home_team,
            ):
                return (
                    actual_home_goals
                    > actual_away_goals
                )

            if normalized_selection in {
                "away",
                "away win",
            } or PredictionHistoryService._team_names_match(
                selection,
                api_away_team,
            ):
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

    @staticmethod
    def _team_names_match(
        first_team: str,
        second_team: str,
    ) -> bool:
        first_normalized = (
            PredictionHistoryService
            ._normalize_team_name(
                first_team
            )
        )

        second_normalized = (
            PredictionHistoryService
            ._normalize_team_name(
                second_team
            )
        )

        return (
            bool(first_normalized)
            and first_normalized
            == second_normalized
        )

    @staticmethod
    def _normalize_team_name(
        team_name: str,
    ) -> str:
        normalized = unicodedata.normalize(
            "NFKD",
            str(team_name).casefold(),
        )

        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(
                character
            )
        )

        normalized = normalized.replace(
            "&",
            " and ",
        )

        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            normalized,
        )

        ignored_words = {
            "afc",
            "bk",
            "cd",
            "cf",
            "club",
            "fc",
            "fk",
            "football",
            "if",
            "jk",
            "kv",
            "rc",
            "sc",
            "sk",
            "sl",
            "ssc",
            "sv",
            "town",
            "ud",
        }

        words = [
            word
            for word in normalized.split()
            if word not in ignored_words
        ]

        return " ".join(words).strip()
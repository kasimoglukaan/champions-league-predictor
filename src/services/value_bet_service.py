from __future__ import annotations

from typing import Optional

from src.models.bookmaker_odds import (
    BookmakerPrice,
    MatchOdds,
)
from src.models.value_bet import (
    ValueBet,
    ValueBetReport,
)


class ValueBetService:
    def __init__(
        self,
        minimum_model_probability: float = 0.45,
        minimum_edge: float = 0.03,
        minimum_expected_value: float = 0.03,
        maximum_decimal_odds: float = 8.00,
    ) -> None:
        self.minimum_model_probability = (
            minimum_model_probability
        )

        self.minimum_edge = minimum_edge

        self.minimum_expected_value = (
            minimum_expected_value
        )

        self.maximum_decimal_odds = (
            maximum_decimal_odds
        )

    def analyse(
        self,
        prediction,
        match_odds: MatchOdds,
    ) -> ValueBetReport:
        candidates = []

        candidates.extend(
            self._build_match_result_candidates(
                prediction=prediction,
                match_odds=match_odds,
            )
        )

        candidates.extend(
            self._build_total_goals_candidates(
                prediction=prediction,
                match_odds=match_odds,
            )
        )

        opportunities = [
            candidate
            for candidate in candidates
            if self._is_value_opportunity(
                candidate
            )
        ]

        rejected_markets = [
            candidate
            for candidate in candidates
            if candidate not in opportunities
        ]

        opportunities.sort(
            key=self._opportunity_sort_key,
            reverse=True,
        )

        rejected_markets.sort(
            key=self._opportunity_sort_key,
            reverse=True,
        )

        best_opportunity = (
            opportunities[0]
            if opportunities
            else None
        )

        return ValueBetReport(
            opportunities=opportunities,
            rejected_markets=rejected_markets,
            best_opportunity=(
                best_opportunity
            ),
        )

    def _build_match_result_candidates(
        self,
        prediction,
        match_odds: MatchOdds,
    ) -> list[ValueBet]:
        hybrid_home_probability = (
            prediction.home_win_probability
            + prediction.poisson_home_probability
        ) / 2.0

        hybrid_draw_probability = (
            prediction.draw_probability
            + prediction.poisson_draw_probability
        ) / 2.0

        hybrid_away_probability = (
            prediction.away_win_probability
            + prediction.poisson_away_probability
        ) / 2.0

        result_candidates = [
            (
                match_odds.home_team,
                hybrid_home_probability,
                prediction.home_team,
            ),
            (
                "Draw",
                hybrid_draw_probability,
                "Draw",
            ),
            (
                match_odds.away_team,
                hybrid_away_probability,
                prediction.away_team,
            ),
        ]

        value_bets = []

        for (
            api_selection,
            model_probability,
            display_selection,
        ) in result_candidates:
            best_price = (
                match_odds.get_best_price(
                    market_key="h2h",
                    selection=api_selection,
                )
            )

            if best_price is None:
                continue

            value_bet = self._create_value_bet(
                price=best_price,
                model_probability=(
                    model_probability
                ),
                display_selection=(
                    display_selection
                ),
            )

            if value_bet is not None:
                value_bets.append(
                    value_bet
                )

        return value_bets

    def _build_total_goals_candidates(
        self,
        prediction,
        match_odds: MatchOdds,
    ) -> list[ValueBet]:
        total_candidates = [
            (
                "Over",
                1.5,
                prediction.over_1_5_probability,
                "Over 1.5 goals",
            ),
            (
                "Over",
                2.5,
                prediction.over_2_5_probability,
                "Over 2.5 goals",
            ),
            (
                "Under",
                2.5,
                prediction.under_2_5_probability,
                "Under 2.5 goals",
            ),
            (
                "Under",
                3.5,
                prediction.under_3_5_probability,
                "Under 3.5 goals",
            ),
        ]

        value_bets = []

        for (
            api_selection,
            point,
            model_probability,
            display_selection,
        ) in total_candidates:
            best_price = (
                match_odds.get_best_price(
                    market_key="totals",
                    selection=api_selection,
                    point=point,
                )
            )

            if best_price is None:
                continue

            value_bet = self._create_value_bet(
                price=best_price,
                model_probability=(
                    model_probability
                ),
                display_selection=(
                    display_selection
                ),
            )

            if value_bet is not None:
                value_bets.append(
                    value_bet
                )

        return value_bets

    def _create_value_bet(
        self,
        price: BookmakerPrice,
        model_probability: float,
        display_selection: str,
    ) -> Optional[ValueBet]:
        if price.odds <= 1.0:
            return None

        if (
            price.odds
            > self.maximum_decimal_odds
        ):
            return None

        safe_probability = min(
            max(
                float(model_probability),
                0.0,
            ),
            1.0,
        )

        implied_probability = (
            1.0 / price.odds
        )

        edge = (
            safe_probability
            - implied_probability
        )

        expected_value = (
            safe_probability
            * price.odds
            - 1.0
        )

        return ValueBet(
            market_key=price.market_key,
            market_name=price.market_name,
            selection=display_selection,
            bookmaker=price.bookmaker_title,
            decimal_odds=price.odds,
            model_probability=(
                safe_probability
            ),
            implied_probability=(
                implied_probability
            ),
            edge=edge,
            expected_value=(
                expected_value
            ),
            point=price.point,
        )

    def _is_value_opportunity(
        self,
        value_bet: ValueBet,
    ) -> bool:
        return (
            value_bet.model_probability
            >= self.minimum_model_probability
            and value_bet.edge
            >= self.minimum_edge
            and value_bet.expected_value
            >= self.minimum_expected_value
        )

    @staticmethod
    def _opportunity_sort_key(
        value_bet: ValueBet,
    ) -> tuple[float, float, float]:
        return (
            value_bet.expected_value,
            value_bet.edge,
            value_bet.model_probability,
        )
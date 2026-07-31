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
        maximum_bookmaker_margin: float = 0.15,
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

        self.maximum_bookmaker_margin = (
            maximum_bookmaker_margin
        )

    def analyse(
        self,
        prediction,
        match_odds: MatchOdds,
    ) -> ValueBetReport:
        candidates: list[ValueBet] = []

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
            best_opportunity=best_opportunity,
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

        value_bets: list[ValueBet] = []

        for (
            api_selection,
            model_probability,
            display_selection,
        ) in result_candidates:
            best_price = match_odds.get_best_price(
                market_key="h2h",
                selection=api_selection,
            )

            if best_price is None:
                continue

            value_bet = self._create_value_bet(
                match_odds=match_odds,
                price=best_price,
                model_probability=model_probability,
                display_selection=display_selection,
            )

            if value_bet is not None:
                value_bets.append(value_bet)

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

        value_bets: list[ValueBet] = []

        for (
            api_selection,
            point,
            model_probability,
            display_selection,
        ) in total_candidates:
            best_price = match_odds.get_best_price(
                market_key="totals",
                selection=api_selection,
                point=point,
            )

            if best_price is None:
                continue

            value_bet = self._create_value_bet(
                match_odds=match_odds,
                price=best_price,
                model_probability=model_probability,
                display_selection=display_selection,
            )

            if value_bet is not None:
                value_bets.append(value_bet)

        return value_bets

    def _create_value_bet(
        self,
        match_odds: MatchOdds,
        price: BookmakerPrice,
        model_probability: float,
        display_selection: str,
    ) -> Optional[ValueBet]:
        if price.odds <= 1.0:
            return None

        if price.odds > self.maximum_decimal_odds:
            return None

        safe_probability = min(
            max(
                float(model_probability),
                0.0,
            ),
            1.0,
        )

        market_prices = self._get_same_bookmaker_market(
            match_odds=match_odds,
            selected_price=price,
        )

        market_result = (
            self._calculate_fair_market_probability(
                selected_price=price,
                market_prices=market_prices,
            )
        )

        if market_result is None:
            return None

        (
            raw_implied_probability,
            fair_market_probability,
            bookmaker_margin,
        ) = market_result

        edge = (
            safe_probability
            - fair_market_probability
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
            bookmaker_key=price.bookmaker_key,
            bookmaker=price.bookmaker_title,
            decimal_odds=price.odds,
            model_probability=safe_probability,
            raw_implied_probability=(
                raw_implied_probability
            ),
            fair_market_probability=(
                fair_market_probability
            ),
            bookmaker_margin=bookmaker_margin,
            edge=edge,
            expected_value=expected_value,
            point=price.point,
        )

    @staticmethod
    def _get_same_bookmaker_market(
        match_odds: MatchOdds,
        selected_price: BookmakerPrice,
    ) -> list[BookmakerPrice]:
        matching_prices: list[BookmakerPrice] = []

        for candidate in match_odds.prices:
            if (
                candidate.bookmaker_key
                != selected_price.bookmaker_key
            ):
                continue

            if (
                candidate.market_key
                != selected_price.market_key
            ):
                continue

            if not ValueBetService._same_point(
                candidate.point,
                selected_price.point,
            ):
                continue

            matching_prices.append(candidate)

        return matching_prices

    @staticmethod
    def _calculate_fair_market_probability(
        selected_price: BookmakerPrice,
        market_prices: list[BookmakerPrice],
    ) -> Optional[
        tuple[float, float, float]
    ]:
        if not market_prices:
            return None

        unique_outcomes: dict[
            tuple[str, Optional[float]],
            BookmakerPrice,
        ] = {}

        for price in market_prices:
            outcome_key = (
                price.selection.strip().casefold(),
                price.point,
            )

            existing_price = unique_outcomes.get(
                outcome_key
            )

            if (
                existing_price is None
                or price.odds
                > existing_price.odds
            ):
                unique_outcomes[
                    outcome_key
                ] = price

        required_outcome_count = (
            3
            if selected_price.market_key == "h2h"
            else 2
        )

        if (
            len(unique_outcomes)
            < required_outcome_count
        ):
            return None

        total_implied_probability = 0.0

        for price in unique_outcomes.values():
            if price.odds <= 1.0:
                return None

            total_implied_probability += (
                1.0 / price.odds
            )

        if total_implied_probability <= 0:
            return None

        raw_implied_probability = (
            1.0 / selected_price.odds
        )

        fair_market_probability = (
            raw_implied_probability
            / total_implied_probability
        )

        bookmaker_margin = (
            total_implied_probability
            - 1.0
        )

        return (
            raw_implied_probability,
            fair_market_probability,
            bookmaker_margin,
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
            and value_bet.bookmaker_margin
            <= self.maximum_bookmaker_margin
        )

    @staticmethod
    def _same_point(
        first_point: Optional[float],
        second_point: Optional[float],
    ) -> bool:
        if (
            first_point is None
            and second_point is None
        ):
            return True

        if (
            first_point is None
            or second_point is None
        ):
            return False

        return abs(
            first_point
            - second_point
        ) < 0.001

    @staticmethod
    def _opportunity_sort_key(
        value_bet: ValueBet,
    ) -> tuple[float, float, float]:
        return (
            value_bet.expected_value,
            value_bet.edge,
            value_bet.model_probability,
        )
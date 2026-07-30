from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BookmakerPrice:
    bookmaker_key: str
    bookmaker_title: str
    market_key: str
    market_name: str
    selection: str
    odds: float
    point: Optional[float] = None
    last_update: Optional[str] = None

    @property
    def implied_probability(self) -> float:
        if self.odds <= 0:
            return 0.0

        return 1.0 / self.odds


@dataclass
class MatchOdds:
    event_id: str
    sport_key: str
    commence_time: str
    home_team: str
    away_team: str

    prices: list[BookmakerPrice] = field(
        default_factory=list
    )

    requests_remaining: Optional[int] = None
    requests_used: Optional[int] = None
    request_cost: Optional[int] = None

    def get_market_prices(
        self,
        market_key: str,
    ) -> list[BookmakerPrice]:
        return [
            price
            for price in self.prices
            if price.market_key == market_key
        ]

    def get_selection_prices(
        self,
        market_key: str,
        selection: str,
        point: Optional[float] = None,
    ) -> list[BookmakerPrice]:
        normalized_selection = (
            selection.strip().casefold()
        )

        matching_prices = []

        for price in self.prices:
            if price.market_key != market_key:
                continue

            if (
                price.selection.strip().casefold()
                != normalized_selection
            ):
                continue

            if point is not None:
                if price.point is None:
                    continue

                if abs(price.point - point) > 0.001:
                    continue

            matching_prices.append(
                price
            )

        return matching_prices

    def get_best_price(
        self,
        market_key: str,
        selection: str,
        point: Optional[float] = None,
    ) -> Optional[BookmakerPrice]:
        prices = self.get_selection_prices(
            market_key=market_key,
            selection=selection,
            point=point,
        )

        if not prices:
            return None

        return max(
            prices,
            key=lambda price: price.odds,
        )


@dataclass(frozen=True)
class OddsEvent:
    event_id: str
    sport_key: str
    sport_title: str
    commence_time: str
    home_team: str
    away_team: str
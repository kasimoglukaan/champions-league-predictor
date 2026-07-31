from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ValueBet:
    market_key: str
    market_name: str
    selection: str

    bookmaker_key: str
    bookmaker: str
    decimal_odds: float

    model_probability: float

    raw_implied_probability: float
    fair_market_probability: float
    bookmaker_margin: float

    edge: float
    expected_value: float

    point: Optional[float] = None

    @property
    def implied_probability(self) -> float:
        """
        Backward-compatible alias used by the UI.

        The value returned here is the bookmaker-margin-adjusted
        market probability rather than the raw 1 / odds value.
        """
        return self.fair_market_probability

    @property
    def edge_percent(self) -> float:
        return self.edge * 100.0

    @property
    def expected_value_percent(self) -> float:
        return self.expected_value * 100.0

    @property
    def bookmaker_margin_percent(self) -> float:
        return self.bookmaker_margin * 100.0

    @property
    def fair_odds(self) -> float:
        if self.model_probability <= 0:
            return 0.0

        return 1.0 / self.model_probability

    @property
    def market_fair_odds(self) -> float:
        if self.fair_market_probability <= 0:
            return 0.0

        return 1.0 / self.fair_market_probability

    @property
    def confidence_label(self) -> str:
        if (
            self.edge >= 0.10
            and self.expected_value >= 0.15
        ):
            return "STRONG"

        if (
            self.edge >= 0.06
            and self.expected_value >= 0.08
        ):
            return "MODERATE"

        if (
            self.edge >= 0.03
            and self.expected_value >= 0.03
        ):
            return "SMALL"

        return "NO VALUE"


@dataclass
class ValueBetReport:
    opportunities: list[ValueBet] = field(
        default_factory=list
    )

    rejected_markets: list[ValueBet] = field(
        default_factory=list
    )

    best_opportunity: Optional[ValueBet] = None

    @property
    def has_value_bet(self) -> bool:
        return bool(self.opportunities)
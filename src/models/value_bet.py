from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ValueBet:
    market_key: str
    market_name: str
    selection: str

    bookmaker: str
    decimal_odds: float

    model_probability: float
    implied_probability: float

    edge: float
    expected_value: float

    point: Optional[float] = None

    @property
    def edge_percent(self) -> float:
        return self.edge * 100.0

    @property
    def expected_value_percent(self) -> float:
        return self.expected_value * 100.0

    @property
    def fair_odds(self) -> float:
        if self.model_probability <= 0:
            return 0.0

        return 1.0 / self.model_probability

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
        return bool(
            self.opportunities
        )
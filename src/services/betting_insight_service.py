from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BettingInsight:
    market: str
    selection: str
    probability: float
    confidence_label: str
    explanation: str


@dataclass(frozen=True)
class BettingInsightReport:
    best_insight: BettingInsight | None
    insights: List[BettingInsight]
    no_strong_signal: bool


class BettingInsightService:
    def __init__(
        self,
        strong_threshold: float = 0.72,
        moderate_threshold: float = 0.62,
    ) -> None:
        if not 0.0 < moderate_threshold < 1.0:
            raise ValueError(
                "moderate_threshold must be "
                "between 0 and 1."
            )

        if not 0.0 < strong_threshold < 1.0:
            raise ValueError(
                "strong_threshold must be "
                "between 0 and 1."
            )

        if strong_threshold <= moderate_threshold:
            raise ValueError(
                "strong_threshold must be greater "
                "than moderate_threshold."
            )

        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold

    def generate(
        self,
        home_team: str,
        away_team: str,
        home_win_probability: float,
        draw_probability: float,
        away_win_probability: float,
        btts_probability: float,
        over_1_5_probability: float,
        over_2_5_probability: float,
        under_2_5_probability: float,
        under_3_5_probability: float,
    ) -> BettingInsightReport:
        home_or_draw_probability = (
            home_win_probability
            + draw_probability
        )

        away_or_draw_probability = (
            away_win_probability
            + draw_probability
        )

        candidates = [
            self._create_insight(
                market="Double chance",
                selection=f"{home_team} or draw (1X)",
                probability=home_or_draw_probability,
                explanation=(
                    f"The model estimates that "
                    f"{home_team} avoids defeat in "
                    f"{home_or_draw_probability:.1%} "
                    "of simulated outcomes."
                ),
            ),
            self._create_insight(
                market="Double chance",
                selection=f"{away_team} or draw (X2)",
                probability=away_or_draw_probability,
                explanation=(
                    f"The model estimates that "
                    f"{away_team} avoids defeat in "
                    f"{away_or_draw_probability:.1%} "
                    "of simulated outcomes."
                ),
            ),
            self._create_insight(
                market="Total goals",
                selection="Over 1.5 goals",
                probability=over_1_5_probability,
                explanation=(
                    "At least two total goals occurred in "
                    f"{over_1_5_probability:.1%} of the "
                    "simulated matches."
                ),
            ),
            self._create_insight(
                market="Total goals",
                selection="Over 2.5 goals",
                probability=over_2_5_probability,
                explanation=(
                    "At least three total goals occurred in "
                    f"{over_2_5_probability:.1%} of the "
                    "simulated matches."
                ),
            ),
            self._create_insight(
                market="Total goals",
                selection="Under 2.5 goals",
                probability=under_2_5_probability,
                explanation=(
                    "Fewer than three total goals occurred in "
                    f"{under_2_5_probability:.1%} of the "
                    "simulated matches."
                ),
            ),
            self._create_insight(
                market="Total goals",
                selection="Under 3.5 goals",
                probability=under_3_5_probability,
                explanation=(
                    "Fewer than four total goals occurred in "
                    f"{under_3_5_probability:.1%} of the "
                    "simulated matches."
                ),
            ),
            self._create_insight(
                market="Both teams to score",
                selection="Both teams to score — Yes",
                probability=btts_probability,
                explanation=(
                    "Both teams scored in "
                    f"{btts_probability:.1%} of the "
                    "simulated matches."
                ),
            ),
            self._create_insight(
                market="Both teams to score",
                selection="Both teams to score — No",
                probability=1.0 - btts_probability,
                explanation=(
                    "At least one team failed to score in "
                    f"{1.0 - btts_probability:.1%} of the "
                    "simulated matches."
                ),
            ),
            self._create_insight(
                market="Match result",
                selection=f"{home_team} to win",
                probability=home_win_probability,
                explanation=(
                    f"{home_team} won in "
                    f"{home_win_probability:.1%} of the "
                    "modelled outcomes."
                ),
            ),
            self._create_insight(
                market="Match result",
                selection="Draw",
                probability=draw_probability,
                explanation=(
                    "The match ended in a draw in "
                    f"{draw_probability:.1%} of the "
                    "modelled outcomes."
                ),
            ),
            self._create_insight(
                market="Match result",
                selection=f"{away_team} to win",
                probability=away_win_probability,
                explanation=(
                    f"{away_team} won in "
                    f"{away_win_probability:.1%} of the "
                    "modelled outcomes."
                ),
            ),
        ]

        ranked_insights = sorted(
            candidates,
            key=lambda insight: insight.probability,
            reverse=True,
        )

        meaningful_insights = [
            insight
            for insight in ranked_insights
            if insight.confidence_label
            in {
                "STRONG",
                "MODERATE",
            }
        ]

        best_insight = (
            meaningful_insights[0]
            if meaningful_insights
            else None
        )

        return BettingInsightReport(
            best_insight=best_insight,
            insights=meaningful_insights[:6],
            no_strong_signal=best_insight is None,
        )

    def _create_insight(
        self,
        market: str,
        selection: str,
        probability: float,
        explanation: str,
    ) -> BettingInsight:
        safe_probability = float(
            min(
                max(
                    probability,
                    0.0,
                ),
                1.0,
            )
        )

        return BettingInsight(
            market=market,
            selection=selection,
            probability=safe_probability,
            confidence_label=(
                self._confidence_label(
                    safe_probability
                )
            ),
            explanation=explanation,
        )

    def _confidence_label(
        self,
        probability: float,
    ) -> str:
        if probability >= self.strong_threshold:
            return "STRONG"

        if probability >= self.moderate_threshold:
            return "MODERATE"

        return "WEAK"
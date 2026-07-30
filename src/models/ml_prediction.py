from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PredictionFactor:
    title: str
    explanation: str
    strength: float
    supports_prediction: bool


@dataclass
class MLPrediction:
    home_team: str
    away_team: str

    home_win_probability: float
    draw_probability: float
    away_win_probability: float

    predicted_result: str
    confidence: float
    confidence_label: str

    home_elo: float
    away_elo: float

    poisson_home_probability: float = 0.0
    poisson_draw_probability: float = 0.0
    poisson_away_probability: float = 0.0

    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0

    most_likely_home_goals: int = 0
    most_likely_away_goals: int = 0

    positive_factors: List[
        PredictionFactor
    ] = field(default_factory=list)

    negative_factors: List[
        PredictionFactor
    ] = field(default_factory=list)

    @property
    def predicted_result_text(self) -> str:
        if self.predicted_result == "H":
            return f"{self.home_team} wins"

        if self.predicted_result == "A":
            return f"{self.away_team} wins"

        return "Draw"

    @property
    def most_likely_score_text(self) -> str:
        return (
            f"{self.home_team} "
            f"{self.most_likely_home_goals}"
            "-"
            f"{self.most_likely_away_goals} "
            f"{self.away_team}"
        )
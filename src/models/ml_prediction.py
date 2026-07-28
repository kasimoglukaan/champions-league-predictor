from dataclasses import dataclass


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

    @property
    def predicted_result_text(self) -> str:
        if self.predicted_result == "H":
            return f"{self.home_team} wins"

        if self.predicted_result == "A":
            return f"{self.away_team} wins"

        return "Draw"
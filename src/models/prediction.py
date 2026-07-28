from dataclasses import dataclass


@dataclass
class MatchPrediction:
    home_team: str
    away_team: str

    home_win_probability: float
    draw_probability: float
    away_win_probability: float

    expected_home_goals: float
    expected_away_goals: float

    most_likely_home_goals: int
    most_likely_away_goals: int

    @property
    def most_likely_score(self) -> str:
        return (
            f"{self.most_likely_home_goals}-"
            f"{self.most_likely_away_goals}"
        )

    def as_dict(self) -> dict:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_win_probability": self.home_win_probability,
            "draw_probability": self.draw_probability,
            "away_win_probability": self.away_win_probability,
            "expected_home_goals": self.expected_home_goals,
            "expected_away_goals": self.expected_away_goals,
            "most_likely_score": self.most_likely_score,
        }
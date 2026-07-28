from dataclasses import dataclass
from datetime import datetime


@dataclass
class Match:
    date: datetime
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int

    @property
    def result(self) -> str:
        """
        H = Home win
        D = Draw
        A = Away win
        """
        if self.home_goals > self.away_goals:
            return "H"

        if self.home_goals < self.away_goals:
            return "A"

        return "D"

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals
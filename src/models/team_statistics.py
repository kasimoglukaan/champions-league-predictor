from collections import defaultdict

from src.models.match import Match


class TeamStatistics:
    def __init__(self) -> None:
        self.home_goals_scored: dict[str, list[int]] = (
            defaultdict(list)
        )

        self.home_goals_conceded: dict[str, list[int]] = (
            defaultdict(list)
        )

        self.away_goals_scored: dict[str, list[int]] = (
            defaultdict(list)
        )

        self.away_goals_conceded: dict[str, list[int]] = (
            defaultdict(list)
        )

        self.global_home_goal_average = 1.5
        self.global_away_goal_average = 1.2

    def calculate(
        self,
        matches: list[Match],
    ) -> None:
        if not matches:
            raise ValueError(
                "At least one match is required."
            )

        total_home_goals = 0
        total_away_goals = 0

        for match in matches:
            total_home_goals += match.home_goals
            total_away_goals += match.away_goals

            self.home_goals_scored[
                match.home_team
            ].append(match.home_goals)

            self.home_goals_conceded[
                match.home_team
            ].append(match.away_goals)

            self.away_goals_scored[
                match.away_team
            ].append(match.away_goals)

            self.away_goals_conceded[
                match.away_team
            ].append(match.home_goals)

        match_count = len(matches)

        self.global_home_goal_average = (
            total_home_goals / match_count
        )

        self.global_away_goal_average = (
            total_away_goals / match_count
        )

    def get_home_attack_strength(
        self,
        team_name: str,
    ) -> float:
        goals = self.home_goals_scored.get(team_name)

        if not goals:
            return 1.0

        average = sum(goals) / len(goals)

        return self._safe_divide(
            average,
            self.global_home_goal_average,
        )

    def get_home_defence_strength(
        self,
        team_name: str,
    ) -> float:
        goals = self.home_goals_conceded.get(team_name)

        if not goals:
            return 1.0

        average = sum(goals) / len(goals)

        return self._safe_divide(
            average,
            self.global_away_goal_average,
        )

    def get_away_attack_strength(
        self,
        team_name: str,
    ) -> float:
        goals = self.away_goals_scored.get(team_name)

        if not goals:
            return 1.0

        average = sum(goals) / len(goals)

        return self._safe_divide(
            average,
            self.global_away_goal_average,
        )

    def get_away_defence_strength(
        self,
        team_name: str,
    ) -> float:
        goals = self.away_goals_conceded.get(team_name)

        if not goals:
            return 1.0

        average = sum(goals) / len(goals)

        return self._safe_divide(
            average,
            self.global_home_goal_average,
        )

    @staticmethod
    def _safe_divide(
        numerator: float,
        denominator: float,
    ) -> float:
        if denominator == 0:
            return 1.0

        return numerator / denominator
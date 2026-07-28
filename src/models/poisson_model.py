import math

from src.models.elo_model import EloModel
from src.models.prediction import MatchPrediction
from src.models.team_statistics import TeamStatistics


class PoissonModel:
    def __init__(
        self,
        team_statistics: TeamStatistics,
        elo_model: EloModel,
        max_goals: int = 7,
        elo_weight: float = 0.20,
    ) -> None:
        self.team_statistics = team_statistics
        self.elo_model = elo_model
        self.max_goals = max_goals
        self.elo_weight = elo_weight

    def poisson_probability(
        self,
        goals: int,
        expected_goals: float,
    ) -> float:
        return (
            math.exp(-expected_goals)
            * expected_goals ** goals
            / math.factorial(goals)
        )

    def calculate_expected_goals(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float]:
        home_attack = (
            self.team_statistics
            .get_home_attack_strength(home_team)
        )

        home_defence = (
            self.team_statistics
            .get_home_defence_strength(home_team)
        )

        away_attack = (
            self.team_statistics
            .get_away_attack_strength(away_team)
        )

        away_defence = (
            self.team_statistics
            .get_away_defence_strength(away_team)
        )

        expected_home_goals = (
            home_attack
            * away_defence
            * self.team_statistics
            .global_home_goal_average
        )

        expected_away_goals = (
            away_attack
            * home_defence
            * self.team_statistics
            .global_away_goal_average
        )

        elo_difference = (
            self.elo_model.get_rating_difference(
                home_team,
                away_team,
            )
        )

        elo_adjustment = (
            elo_difference / 400
        ) * self.elo_weight

        expected_home_goals *= max(
            0.5,
            1 + elo_adjustment,
        )

        expected_away_goals *= max(
            0.5,
            1 - elo_adjustment,
        )

        expected_home_goals = self._limit_expected_goals(
            expected_home_goals
        )

        expected_away_goals = self._limit_expected_goals(
            expected_away_goals
        )

        return (
            expected_home_goals,
            expected_away_goals,
        )

    def predict(
        self,
        home_team: str,
        away_team: str,
    ) -> MatchPrediction:
        if home_team == away_team:
            raise ValueError(
                "Home team and away team must be different."
            )

        (
            expected_home_goals,
            expected_away_goals,
        ) = self.calculate_expected_goals(
            home_team,
            away_team,
        )

        home_win_probability = 0.0
        draw_probability = 0.0
        away_win_probability = 0.0

        highest_score_probability = 0.0
        most_likely_home_goals = 0
        most_likely_away_goals = 0

        total_probability = 0.0

        for home_goals in range(
            self.max_goals + 1
        ):
            home_probability = (
                self.poisson_probability(
                    home_goals,
                    expected_home_goals,
                )
            )

            for away_goals in range(
                self.max_goals + 1
            ):
                away_probability = (
                    self.poisson_probability(
                        away_goals,
                        expected_away_goals,
                    )
                )

                score_probability = (
                    home_probability
                    * away_probability
                )

                total_probability += score_probability

                if home_goals > away_goals:
                    home_win_probability += (
                        score_probability
                    )

                elif home_goals == away_goals:
                    draw_probability += score_probability

                else:
                    away_win_probability += (
                        score_probability
                    )

                if (
                    score_probability
                    > highest_score_probability
                ):
                    highest_score_probability = (
                        score_probability
                    )

                    most_likely_home_goals = home_goals
                    most_likely_away_goals = away_goals

        if total_probability > 0:
            home_win_probability /= total_probability
            draw_probability /= total_probability
            away_win_probability /= total_probability

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_win_probability=home_win_probability,
            draw_probability=draw_probability,
            away_win_probability=away_win_probability,
            expected_home_goals=expected_home_goals,
            expected_away_goals=expected_away_goals,
            most_likely_home_goals=(
                most_likely_home_goals
            ),
            most_likely_away_goals=(
                most_likely_away_goals
            ),
        )

    @staticmethod
    def _limit_expected_goals(
        value: float,
    ) -> float:
        return max(0.15, min(value, 4.5))
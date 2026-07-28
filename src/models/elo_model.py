from collections import defaultdict

from src.models.match import Match


class EloModel:
    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 30.0,
        home_advantage: float = 80.0,
    ) -> None:
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage

        self.ratings: dict[str, float] = defaultdict(
            lambda: self.initial_rating
        )

    def expected_score(
        self,
        team_rating: float,
        opponent_rating: float,
    ) -> float:
        return 1 / (
            1
            + 10
            ** (
                (opponent_rating - team_rating)
                / 400
            )
        )

    def update_ratings(self, match: Match) -> None:
        home_rating = self.get_rating(match.home_team)
        away_rating = self.get_rating(match.away_team)

        adjusted_home_rating = (
            home_rating + self.home_advantage
        )

        expected_home = self.expected_score(
            adjusted_home_rating,
            away_rating,
        )

        expected_away = 1 - expected_home

        actual_home, actual_away = (
            self._get_actual_scores(match)
        )

        new_home_rating = (
            home_rating
            + self.k_factor
            * (actual_home - expected_home)
        )

        new_away_rating = (
            away_rating
            + self.k_factor
            * (actual_away - expected_away)
        )

        self.ratings[match.home_team] = new_home_rating
        self.ratings[match.away_team] = new_away_rating

    def train(self, matches: list[Match]) -> None:
        self.reset()

        sorted_matches = sorted(
            matches,
            key=lambda match: match.date,
        )

        for match in sorted_matches:
            self.update_ratings(match)

    def get_rating(self, team_name: str) -> float:
        return float(self.ratings[team_name])

    def get_all_ratings(self) -> dict[str, float]:
        return dict(
            sorted(
                self.ratings.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def get_rating_difference(
        self,
        home_team: str,
        away_team: str,
    ) -> float:
        home_rating = (
            self.get_rating(home_team)
            + self.home_advantage
        )

        away_rating = self.get_rating(away_team)

        return home_rating - away_rating

    def reset(self) -> None:
        self.ratings = defaultdict(
            lambda: self.initial_rating
        )

    @staticmethod
    def _get_actual_scores(
        match: Match,
    ) -> tuple[float, float]:
        if match.home_goals > match.away_goals:
            return 1.0, 0.0

        if match.home_goals < match.away_goals:
            return 0.0, 1.0

        return 0.5, 0.5
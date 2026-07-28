from src.data.match_repository import MatchRepository
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.models.prediction import MatchPrediction
from src.models.team_statistics import TeamStatistics
from src.services.monte_carlo_simulator import (
    MonteCarloSimulator,
    SimulationResult,
)


class PredictorService:
    def __init__(
        self,
        csv_path: str,
    ) -> None:
        self.repository = MatchRepository(csv_path)

        self.elo_model = EloModel(
            initial_rating=1500,
            k_factor=30,
            home_advantage=80,
        )

        self.team_statistics = TeamStatistics()

        self.poisson_model = PoissonModel(
            team_statistics=self.team_statistics,
            elo_model=self.elo_model,
            max_goals=7,
        )

        self.simulator = MonteCarloSimulator(
            simulations=10_000,
            random_seed=42,
        )

        self.matches = []
        self.is_trained = False

    def train(self) -> None:
        self.matches = (
            self.repository.load_matches()
        )

        if not self.matches:
            raise ValueError(
                "No matches were found in the dataset."
            )

        self.elo_model.train(self.matches)

        self.team_statistics.calculate(
            self.matches
        )

        self.is_trained = True

    def predict_match(
        self,
        home_team: str,
        away_team: str,
    ) -> MatchPrediction:
        self._ensure_trained()

        return self.poisson_model.predict(
            home_team,
            away_team,
        )

    def simulate_match(
        self,
        home_team: str,
        away_team: str,
    ) -> SimulationResult:
        prediction = self.predict_match(
            home_team,
            away_team,
        )

        return self.simulator.simulate(
            prediction.expected_home_goals,
            prediction.expected_away_goals,
        )

    def get_teams(self) -> list[str]:
        return self.repository.get_team_names()

    def get_rankings(self) -> list[dict]:
        self._ensure_trained()

        ratings = (
            self.elo_model.get_all_ratings()
        )

        ranking = []

        for position, (
            team,
            rating,
        ) in enumerate(
            ratings.items(),
            start=1,
        ):
            ranking.append(
                {
                    "position": position,
                    "team": team,
                    "elo_rating": round(
                        rating,
                        2,
                    ),
                }
            )

        return ranking

    def _ensure_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "The model must be trained first."
            )
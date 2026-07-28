from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SimulationResult:
    home_wins: int
    draws: int
    away_wins: int
    simulations: int

    @property
    def home_win_probability(self) -> float:
        return self.home_wins / self.simulations

    @property
    def draw_probability(self) -> float:
        return self.draws / self.simulations

    @property
    def away_win_probability(self) -> float:
        return self.away_wins / self.simulations


class MonteCarloSimulator:
    def __init__(
        self,
        simulations: int = 10_000,
        random_seed: Optional[int] = 42,
    ) -> None:
        if simulations <= 0:
            raise ValueError(
                "Number of simulations must be positive."
            )

        self.simulations = simulations
        self.random_generator = (
            np.random.default_rng(random_seed)
        )

    def simulate(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
    ) -> SimulationResult:
        home_goals = self.random_generator.poisson(
            expected_home_goals,
            self.simulations,
        )

        away_goals = self.random_generator.poisson(
            expected_away_goals,
            self.simulations,
        )

        home_wins = int(
            np.sum(home_goals > away_goals)
        )

        draws = int(
            np.sum(home_goals == away_goals)
        )

        away_wins = int(
            np.sum(home_goals < away_goals)
        )

        return SimulationResult(
            home_wins=home_wins,
            draws=draws,
            away_wins=away_wins,
            simulations=self.simulations,
        )
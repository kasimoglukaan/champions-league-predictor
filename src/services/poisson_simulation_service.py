from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    home_win_probability: float
    draw_probability: float
    away_win_probability: float

    btts_probability: float

    over_1_5_probability: float
    over_2_5_probability: float
    under_2_5_probability: float
    under_3_5_probability: float

    expected_home_goals: float
    expected_away_goals: float

    most_likely_home_goals: int
    most_likely_away_goals: int

    simulations: int


class PoissonSimulationService:
    def __init__(
        self,
        simulations: int = 20_000,
        random_seed: int = 42,
    ) -> None:
        if simulations <= 0:
            raise ValueError(
                "simulations must be greater "
                "than zero."
            )

        self.simulations = simulations
        self.random_seed = random_seed

    def simulate(
        self,
        home_summary: Dict[str, float],
        away_summary: Dict[str, float],
        home_elo: float,
        away_elo: float,
    ) -> SimulationResult:
        expected_home_goals = (
            self._expected_home_goals(
                home_summary=home_summary,
                away_summary=away_summary,
                home_elo=home_elo,
                away_elo=away_elo,
            )
        )

        expected_away_goals = (
            self._expected_away_goals(
                home_summary=home_summary,
                away_summary=away_summary,
                home_elo=home_elo,
                away_elo=away_elo,
            )
        )

        random_generator = (
            np.random.default_rng(
                self.random_seed
            )
        )

        home_goals = (
            random_generator.poisson(
                lam=expected_home_goals,
                size=self.simulations,
            )
        )

        away_goals = (
            random_generator.poisson(
                lam=expected_away_goals,
                size=self.simulations,
            )
        )

        total_goals = (
            home_goals
            + away_goals
        )

        home_wins = int(
            np.sum(
                home_goals
                > away_goals
            )
        )

        draws = int(
            np.sum(
                home_goals
                == away_goals
            )
        )

        away_wins = int(
            np.sum(
                home_goals
                < away_goals
            )
        )

        both_teams_score = int(
            np.sum(
                (
                    home_goals > 0
                )
                & (
                    away_goals > 0
                )
            )
        )

        over_1_5 = int(
            np.sum(
                total_goals >= 2
            )
        )

        over_2_5 = int(
            np.sum(
                total_goals >= 3
            )
        )

        under_2_5 = int(
            np.sum(
                total_goals <= 2
            )
        )

        under_3_5 = int(
            np.sum(
                total_goals <= 3
            )
        )

        score_pairs = np.column_stack(
            [
                home_goals,
                away_goals,
            ]
        )

        unique_scores, score_counts = (
            np.unique(
                score_pairs,
                axis=0,
                return_counts=True,
            )
        )

        most_likely_index = int(
            np.argmax(
                score_counts
            )
        )

        most_likely_score = (
            unique_scores[
                most_likely_index
            ]
        )

        simulation_count = float(
            self.simulations
        )

        return SimulationResult(
            home_win_probability=float(
                home_wins
                / simulation_count
            ),
            draw_probability=float(
                draws
                / simulation_count
            ),
            away_win_probability=float(
                away_wins
                / simulation_count
            ),
            btts_probability=float(
                both_teams_score
                / simulation_count
            ),
            over_1_5_probability=float(
                over_1_5
                / simulation_count
            ),
            over_2_5_probability=float(
                over_2_5
                / simulation_count
            ),
            under_2_5_probability=float(
                under_2_5
                / simulation_count
            ),
            under_3_5_probability=float(
                under_3_5
                / simulation_count
            ),
            expected_home_goals=float(
                expected_home_goals
            ),
            expected_away_goals=float(
                expected_away_goals
            ),
            most_likely_home_goals=int(
                most_likely_score[0]
            ),
            most_likely_away_goals=int(
                most_likely_score[1]
            ),
            simulations=self.simulations,
        )

    @staticmethod
    def _expected_home_goals(
        home_summary: Dict[str, float],
        away_summary: Dict[str, float],
        home_elo: float,
        away_elo: float,
    ) -> float:
        home_attack = float(
            home_summary[
                "goals_scored"
            ]
        )

        away_defence = float(
            away_summary[
                "goals_conceded"
            ]
        )

        base_expectation = (
            0.55
            * home_attack
            + 0.45
            * away_defence
        )

        elo_difference = (
            home_elo
            - away_elo
        )

        elo_multiplier = np.exp(
            np.clip(
                elo_difference
                / 900.0,
                -0.45,
                0.45,
            )
        )

        home_advantage_multiplier = 1.10

        expected_goals = (
            base_expectation
            * elo_multiplier
            * home_advantage_multiplier
        )

        return float(
            np.clip(
                expected_goals,
                0.20,
                4.50,
            )
        )

    @staticmethod
    def _expected_away_goals(
        home_summary: Dict[str, float],
        away_summary: Dict[str, float],
        home_elo: float,
        away_elo: float,
    ) -> float:
        away_attack = float(
            away_summary[
                "goals_scored"
            ]
        )

        home_defence = float(
            home_summary[
                "goals_conceded"
            ]
        )

        base_expectation = (
            0.55
            * away_attack
            + 0.45
            * home_defence
        )

        elo_difference = (
            away_elo
            - home_elo
        )

        elo_multiplier = np.exp(
            np.clip(
                elo_difference
                / 900.0,
                -0.45,
                0.45,
            )
        )

        away_penalty_multiplier = 0.94

        expected_goals = (
            base_expectation
            * elo_multiplier
            * away_penalty_multiplier
        )

        return float(
            np.clip(
                expected_goals,
                0.20,
                4.50,
            )
        )
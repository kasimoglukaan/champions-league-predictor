from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

import numpy as np
import pandas as pd

from src.features.feature_builder import FeatureBuilder


class LiveFeatureBuilder:
    """
    Replays all historical matches chronologically and creates
    the current pre-match feature row for a selected fixture.
    """

    def __init__(
        self,
        initial_elo: float = 1500.0,
        k_factor: float = 25.0,
        home_advantage: float = 60.0,
        form_window: int = 8,
    ) -> None:
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.form_window = form_window

        self.ratings: Dict[str, float] = defaultdict(
            lambda: self.initial_elo
        )

        self.histories: Dict[
            str,
            Deque[Tuple[int, int, int]]
        ] = defaultdict(
            lambda: deque(maxlen=self.form_window)
        )

        self.last_match_dates: Dict[
            str,
            pd.Timestamp
        ] = {}

        self.latest_data_date = pd.Timestamp.now(
            tz="UTC"
        )

        self.is_fitted = False

    def fit(
        self,
        matches: pd.DataFrame,
    ) -> None:
        dataframe = matches.copy()

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            utc=True,
        )

        dataframe = dataframe.sort_values(
            "date"
        ).reset_index(drop=True)

        self.ratings = defaultdict(
            lambda: self.initial_elo
        )

        self.histories = defaultdict(
            lambda: deque(maxlen=self.form_window)
        )

        self.last_match_dates = {}

        for row in dataframe.itertuples(
            index=False
        ):
            home_goals = int(row.home_goals)
            away_goals = int(row.away_goals)

            self._update_elo(
                home_team=row.home_team,
                away_team=row.away_team,
                home_goals=home_goals,
                away_goals=away_goals,
            )

            self._update_history(
                team=row.home_team,
                goals_scored=home_goals,
                goals_conceded=away_goals,
            )

            self._update_history(
                team=row.away_team,
                goals_scored=away_goals,
                goals_conceded=home_goals,
            )

            self.last_match_dates[
                row.home_team
            ] = row.date

            self.last_match_dates[
                row.away_team
            ] = row.date

        if not dataframe.empty:
            self.latest_data_date = dataframe[
                "date"
            ].max()

        self.is_fitted = True

    def build_match_features(
        self,
        home_team: str,
        away_team: str,
        competition: str = "CL",
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "LiveFeatureBuilder must be fitted first."
            )

        if home_team == away_team:
            raise ValueError(
                "Home and away teams must be different."
            )

        home_elo = self.ratings[home_team]
        away_elo = self.ratings[away_team]

        home_stats = self._history_features(
            self.histories[home_team]
        )

        away_stats = self._history_features(
            self.histories[away_team]
        )

        prediction_date = max(
            pd.Timestamp.now(tz="UTC"),
            self.latest_data_date
            + pd.Timedelta(days=1),
        )

        home_rest_days = self._rest_days(
            home_team,
            prediction_date,
        )

        away_rest_days = self._rest_days(
            away_team,
            prediction_date,
        )

        feature_row = {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": (
                home_elo
                + self.home_advantage
                - away_elo
            ),
            "home_form_points": (
                home_stats["points"]
            ),
            "away_form_points": (
                away_stats["points"]
            ),
            "form_difference": (
                home_stats["points"]
                - away_stats["points"]
            ),
            "home_goals_scored": (
                home_stats["goals_scored"]
            ),
            "away_goals_scored": (
                away_stats["goals_scored"]
            ),
            "home_goals_conceded": (
                home_stats["goals_conceded"]
            ),
            "away_goals_conceded": (
                away_stats["goals_conceded"]
            ),
            "home_goal_difference": (
                home_stats["goal_difference"]
            ),
            "away_goal_difference": (
                away_stats["goal_difference"]
            ),
            "home_win_rate": (
                home_stats["win_rate"]
            ),
            "away_win_rate": (
                away_stats["win_rate"]
            ),
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "is_champions_league": int(
                competition == "CL"
            ),
        }

        dataframe = pd.DataFrame(
            [feature_row]
        )

        return dataframe[
            FeatureBuilder.FEATURE_COLUMNS
        ]

    def get_teams(self) -> list:
        teams = set(self.ratings.keys())
        teams.update(self.histories.keys())

        return sorted(teams)

    def get_team_summary(
        self,
        team_name: str,
    ) -> dict:
        statistics = self._history_features(
            self.histories[team_name]
        )

        return {
            "team": team_name,
            "elo": round(
                self.ratings[team_name],
                1,
            ),
            "form_points": round(
                statistics["points"],
                2,
            ),
            "goals_scored": round(
                statistics["goals_scored"],
                2,
            ),
            "goals_conceded": round(
                statistics["goals_conceded"],
                2,
            ),
            "win_rate": round(
                statistics["win_rate"],
                3,
            ),
        }

    def _history_features(
        self,
        history: Deque[Tuple[int, int, int]],
    ) -> dict:
        if not history:
            return {
                "points": 0.0,
                "goals_scored": 1.2,
                "goals_conceded": 1.2,
                "goal_difference": 0.0,
                "win_rate": 0.0,
            }

        matches = list(history)

        points = np.mean(
            [match[2] for match in matches]
        )

        goals_scored = np.mean(
            [match[0] for match in matches]
        )

        goals_conceded = np.mean(
            [match[1] for match in matches]
        )

        wins = sum(
            1
            for match in matches
            if match[2] == 3
        )

        return {
            "points": float(points),
            "goals_scored": float(
                goals_scored
            ),
            "goals_conceded": float(
                goals_conceded
            ),
            "goal_difference": float(
                goals_scored
                - goals_conceded
            ),
            "win_rate": (
                wins / len(matches)
            ),
        }

    def _update_history(
        self,
        team: str,
        goals_scored: int,
        goals_conceded: int,
    ) -> None:
        if goals_scored > goals_conceded:
            points = 3

        elif goals_scored == goals_conceded:
            points = 1

        else:
            points = 0

        self.histories[team].append(
            (
                goals_scored,
                goals_conceded,
                points,
            )
        )

    def _update_elo(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> None:
        home_rating = self.ratings[
            home_team
        ]

        away_rating = self.ratings[
            away_team
        ]

        expected_home = 1.0 / (
            1.0
            + 10.0
            ** (
                (
                    away_rating
                    - home_rating
                    - self.home_advantage
                )
                / 400.0
            )
        )

        if home_goals > away_goals:
            actual_home = 1.0

        elif home_goals == away_goals:
            actual_home = 0.5

        else:
            actual_home = 0.0

        goal_margin = abs(
            home_goals - away_goals
        )

        if goal_margin > 0:
            margin_multiplier = (
                1.0
                + np.log1p(goal_margin)
            )

        else:
            margin_multiplier = 1.0

        rating_change = (
            self.k_factor
            * margin_multiplier
            * (
                actual_home
                - expected_home
            )
        )

        self.ratings[home_team] = (
            home_rating + rating_change
        )

        self.ratings[away_team] = (
            away_rating - rating_change
        )

    def _rest_days(
        self,
        team_name: str,
        prediction_date: pd.Timestamp,
    ) -> float:
        previous_date = (
            self.last_match_dates.get(
                team_name
            )
        )

        if previous_date is None:
            return 7.0

        difference = (
            prediction_date
            - previous_date
        ).days

        return float(
            max(1, min(difference, 30))
        )
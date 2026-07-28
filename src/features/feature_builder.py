from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd


class FeatureBuilder:
    FEATURE_COLUMNS = [
        "home_elo",
        "away_elo",
        "elo_difference",
        "home_form_points",
        "away_form_points",
        "form_difference",
        "home_goals_scored",
        "away_goals_scored",
        "home_goals_conceded",
        "away_goals_conceded",
        "home_goal_difference",
        "away_goal_difference",
        "home_win_rate",
        "away_win_rate",
        "home_rest_days",
        "away_rest_days",
        "is_champions_league",
    ]

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

    def build(
        self,
        matches: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = matches.copy()

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            utc=True,
        )

        dataframe = dataframe.sort_values(
            "date"
        ).reset_index(drop=True)

        ratings: Dict[str, float] = defaultdict(
            lambda: self.initial_elo
        )

        histories: Dict[
            str,
            Deque[Tuple[int, int, int]]
        ] = defaultdict(
            lambda: deque(
                maxlen=self.form_window
            )
        )

        last_match_dates: Dict[
            str,
            pd.Timestamp
        ] = {}

        feature_rows: List[Dict] = []

        for row in dataframe.itertuples(
            index=False
        ):
            home_team = row.home_team
            away_team = row.away_team
            match_date = row.date

            home_elo = ratings[home_team]
            away_elo = ratings[away_team]

            home_stats = self._history_features(
                histories[home_team]
            )

            away_stats = self._history_features(
                histories[away_team]
            )

            home_rest_days = self._rest_days(
                team=home_team,
                match_date=match_date,
                last_match_dates=last_match_dates,
            )

            away_rest_days = self._rest_days(
                team=away_team,
                match_date=match_date,
                last_match_dates=last_match_dates,
            )

            feature_rows.append(
                {
                    "match_id": row.match_id,
                    "date": match_date,
                    "competition": row.competition,
                    "home_team": home_team,
                    "away_team": away_team,
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
                        row.competition == "CL"
                    ),
                    "target": row.winner,
                }
            )

            self._update_elo(
                home_team=home_team,
                away_team=away_team,
                home_goals=int(row.home_goals),
                away_goals=int(row.away_goals),
                ratings=ratings,
            )

            self._update_history(
                team=home_team,
                goals_scored=int(row.home_goals),
                goals_conceded=int(row.away_goals),
                history=histories[home_team],
            )

            self._update_history(
                team=away_team,
                goals_scored=int(row.away_goals),
                goals_conceded=int(row.home_goals),
                history=histories[away_team],
            )

            last_match_dates[home_team] = match_date
            last_match_dates[away_team] = match_date

        return pd.DataFrame(feature_rows)

    def _history_features(
        self,
        history: Deque[Tuple[int, int, int]],
    ) -> Dict[str, float]:
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
                goals_scored - goals_conceded
            ),
            "win_rate": wins / len(matches),
        }

    def _update_history(
        self,
        team: str,
        goals_scored: int,
        goals_conceded: int,
        history: Deque[Tuple[int, int, int]],
    ) -> None:
        if goals_scored > goals_conceded:
            points = 3
        elif goals_scored == goals_conceded:
            points = 1
        else:
            points = 0

        history.append(
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
        ratings: Dict[str, float],
    ) -> None:
        home_rating = ratings[home_team]
        away_rating = ratings[away_team]

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

        margin_multiplier = (
            1.0
            + np.log1p(goal_margin)
            if goal_margin > 0
            else 1.0
        )

        change = (
            self.k_factor
            * margin_multiplier
            * (actual_home - expected_home)
        )

        ratings[home_team] = (
            home_rating + change
        )

        ratings[away_team] = (
            away_rating - change
        )

    @staticmethod
    def _rest_days(
        team: str,
        match_date: pd.Timestamp,
        last_match_dates: Dict[
            str,
            pd.Timestamp
        ],
    ) -> float:
        previous_date = last_match_dates.get(
            team
        )

        if previous_date is None:
            return 7.0

        difference = (
            match_date - previous_date
        ).days

        return float(
            max(1, min(difference, 30))
        )
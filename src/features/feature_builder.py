from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd


MatchHistory = Tuple[int, int, int]


class FeatureBuilder:
    FEATURE_COLUMNS = [
        "home_elo",
        "away_elo",
        "elo_difference",
        "absolute_elo_difference",

        "home_form_points_5",
        "away_form_points_5",
        "form_points_difference_5",

        "home_form_points_10",
        "away_form_points_10",
        "form_points_difference_10",

        "home_goals_scored_5",
        "away_goals_scored_5",
        "home_goals_conceded_5",
        "away_goals_conceded_5",

        "home_home_points_5",
        "away_away_points_5",
        "home_home_win_rate_5",
        "away_away_win_rate_5",

        "home_goal_difference_5",
        "away_goal_difference_5",

        "home_rest_days",
        "away_rest_days",

        "home_league_strength",
        "away_league_strength",
        "league_strength_difference",

        "is_champions_league",
    ]

    LEAGUE_STRENGTHS = {
        "PL": 1.00,
        "PD": 0.97,
        "SA": 0.95,
        "BL1": 0.94,
        "FL1": 0.90,
        "PPL": 0.84,
        "DED": 0.82,
        "CL": 1.00,
    }

    def __init__(
        self,
        initial_elo: float = 1500.0,
        k_factor: float = 25.0,
        home_advantage: float = 60.0,
    ) -> None:
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.home_advantage = home_advantage

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

        overall_histories: Dict[
            str,
            Deque[MatchHistory]
        ] = defaultdict(
            lambda: deque(maxlen=10)
        )

        home_histories: Dict[
            str,
            Deque[MatchHistory]
        ] = defaultdict(
            lambda: deque(maxlen=10)
        )

        away_histories: Dict[
            str,
            Deque[MatchHistory]
        ] = defaultdict(
            lambda: deque(maxlen=10)
        )

        last_match_dates: Dict[
            str,
            pd.Timestamp
        ] = {}

        team_domestic_competition: Dict[
            str,
            str
        ] = {}

        feature_rows: List[Dict] = []

        for row in dataframe.itertuples(
            index=False
        ):
            home_team = str(row.home_team)
            away_team = str(row.away_team)
            competition = str(row.competition)
            match_date = row.date

            home_elo = ratings[home_team]
            away_elo = ratings[away_team]

            home_form_5 = self._history_features(
                overall_histories[home_team],
                window=5,
            )

            away_form_5 = self._history_features(
                overall_histories[away_team],
                window=5,
            )

            home_form_10 = self._history_features(
                overall_histories[home_team],
                window=10,
            )

            away_form_10 = self._history_features(
                overall_histories[away_team],
                window=10,
            )

            home_at_home = self._history_features(
                home_histories[home_team],
                window=5,
            )

            away_as_away = self._history_features(
                away_histories[away_team],
                window=5,
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

            home_league_strength = (
                self._get_team_league_strength(
                    home_team,
                    team_domestic_competition,
                )
            )

            away_league_strength = (
                self._get_team_league_strength(
                    away_team,
                    team_domestic_competition,
                )
            )

            feature_rows.append(
                {
                    "match_id": row.match_id,
                    "date": match_date,
                    "competition": competition,
                    "home_team": home_team,
                    "away_team": away_team,

                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "elo_difference": (
                        home_elo
                        + self.home_advantage
                        - away_elo
                    ),
                    "absolute_elo_difference": abs(
                        home_elo - away_elo
                    ),

                    "home_form_points_5": (
                        home_form_5["points"]
                    ),
                    "away_form_points_5": (
                        away_form_5["points"]
                    ),
                    "form_points_difference_5": (
                        home_form_5["points"]
                        - away_form_5["points"]
                    ),

                    "home_form_points_10": (
                        home_form_10["points"]
                    ),
                    "away_form_points_10": (
                        away_form_10["points"]
                    ),
                    "form_points_difference_10": (
                        home_form_10["points"]
                        - away_form_10["points"]
                    ),

                    "home_goals_scored_5": (
                        home_form_5["goals_scored"]
                    ),
                    "away_goals_scored_5": (
                        away_form_5["goals_scored"]
                    ),
                    "home_goals_conceded_5": (
                        home_form_5["goals_conceded"]
                    ),
                    "away_goals_conceded_5": (
                        away_form_5["goals_conceded"]
                    ),

                    "home_home_points_5": (
                        home_at_home["points"]
                    ),
                    "away_away_points_5": (
                        away_as_away["points"]
                    ),
                    "home_home_win_rate_5": (
                        home_at_home["win_rate"]
                    ),
                    "away_away_win_rate_5": (
                        away_as_away["win_rate"]
                    ),

                    "home_goal_difference_5": (
                        home_form_5["goal_difference"]
                    ),
                    "away_goal_difference_5": (
                        away_form_5["goal_difference"]
                    ),

                    "home_rest_days": home_rest_days,
                    "away_rest_days": away_rest_days,

                    "home_league_strength": (
                        home_league_strength
                    ),
                    "away_league_strength": (
                        away_league_strength
                    ),
                    "league_strength_difference": (
                        home_league_strength
                        - away_league_strength
                    ),

                    "is_champions_league": int(
                        competition == "CL"
                    ),

                    "target": row.winner,
                }
            )

            home_goals = int(row.home_goals)
            away_goals = int(row.away_goals)

            self._update_elo(
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                ratings=ratings,
            )

            self._append_history(
                overall_histories[home_team],
                home_goals,
                away_goals,
            )

            self._append_history(
                overall_histories[away_team],
                away_goals,
                home_goals,
            )

            self._append_history(
                home_histories[home_team],
                home_goals,
                away_goals,
            )

            self._append_history(
                away_histories[away_team],
                away_goals,
                home_goals,
            )

            last_match_dates[home_team] = match_date
            last_match_dates[away_team] = match_date

            if competition != "CL":
                team_domestic_competition[
                    home_team
                ] = competition

                team_domestic_competition[
                    away_team
                ] = competition

        return pd.DataFrame(feature_rows)

    def _history_features(
        self,
        history: Deque[MatchHistory],
        window: int,
    ) -> Dict[str, float]:
        matches = list(history)[-window:]

        if not matches:
            return {
                "points": 1.0,
                "goals_scored": 1.2,
                "goals_conceded": 1.2,
                "goal_difference": 0.0,
                "win_rate": 0.33,
            }

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
            "win_rate": float(
                wins / len(matches)
            ),
        }

    @staticmethod
    def _append_history(
        history: Deque[MatchHistory],
        goals_scored: int,
        goals_conceded: int,
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
            1.0 + np.log1p(goal_margin)
            if goal_margin > 0
            else 1.0
        )

        rating_change = (
            self.k_factor
            * margin_multiplier
            * (
                actual_home
                - expected_home
            )
        )

        ratings[home_team] = (
            home_rating + rating_change
        )

        ratings[away_team] = (
            away_rating - rating_change
        )

    def _get_team_league_strength(
        self,
        team: str,
        team_domestic_competition: Dict[
            str,
            str
        ],
    ) -> float:
        competition = (
            team_domestic_competition.get(team)
        )

        if competition is None:
            return 0.85

        return self.LEAGUE_STRENGTHS.get(
            competition,
            0.85,
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
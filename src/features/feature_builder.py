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

        # Successful Elo-derived features
        "absolute_elo_difference",
        "squared_elo_difference",
        "close_elo_match",

        "home_form_points",
        "away_form_points",
        "form_difference",

        "home_goals_scored",
        "away_goals_scored",
        "home_goals_conceded",
        "away_goals_conceded",

        # Successful attack/defence matchup features
        "home_attack_matchup",
        "away_attack_matchup",
        "attack_matchup_difference",

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
        if form_window < 1:
            raise ValueError(
                "form_window must be at least 1."
            )

        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.form_window = form_window

    def build(
        self,
        matches: pd.DataFrame,
    ) -> pd.DataFrame:
        required_columns = {
            "match_id",
            "date",
            "competition",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "winner",
        }

        missing_columns = (
            required_columns
            - set(matches.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing match columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        dataframe = matches.copy()

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            utc=True,
            errors="raise",
        )

        dataframe = dataframe.sort_values(
            ["date", "match_id"]
        ).reset_index(drop=True)

        ratings: Dict[str, float] = defaultdict(
            lambda: self.initial_elo
        )

        histories: Dict[
            str,
            Deque[MatchHistory],
        ] = defaultdict(
            lambda: deque(
                maxlen=self.form_window
            )
        )

        last_match_dates: Dict[
            str,
            pd.Timestamp,
        ] = {}

        feature_rows: List[Dict] = []

        for row in dataframe.itertuples(
            index=False
        ):
            home_team = str(row.home_team)
            away_team = str(row.away_team)
            competition = str(
                row.competition
            )
            match_date = row.date

            home_elo = float(
                ratings[home_team]
            )

            away_elo = float(
                ratings[away_team]
            )

            home_stats = (
                self._history_features(
                    histories[home_team]
                )
            )

            away_stats = (
                self._history_features(
                    histories[away_team]
                )
            )

            elo_difference = (
                home_elo
                + self.home_advantage
                - away_elo
            )

            absolute_elo_difference = abs(
                home_elo - away_elo
            )

            squared_elo_difference = (
                elo_difference ** 2
            )

            close_elo_match = int(
                absolute_elo_difference < 75
            )

            home_attack_matchup = (
                home_stats[
                    "goals_scored"
                ]
                - away_stats[
                    "goals_conceded"
                ]
            )

            away_attack_matchup = (
                away_stats[
                    "goals_scored"
                ]
                - home_stats[
                    "goals_conceded"
                ]
            )

            attack_matchup_difference = (
                home_attack_matchup
                - away_attack_matchup
            )

            home_rest_days = (
                self._rest_days(
                    team=home_team,
                    match_date=match_date,
                    last_match_dates=(
                        last_match_dates
                    ),
                )
            )

            away_rest_days = (
                self._rest_days(
                    team=away_team,
                    match_date=match_date,
                    last_match_dates=(
                        last_match_dates
                    ),
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
                        elo_difference
                    ),
                    "absolute_elo_difference": (
                        absolute_elo_difference
                    ),
                    "squared_elo_difference": (
                        squared_elo_difference
                    ),
                    "close_elo_match": (
                        close_elo_match
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
                        home_stats[
                            "goals_scored"
                        ]
                    ),
                    "away_goals_scored": (
                        away_stats[
                            "goals_scored"
                        ]
                    ),
                    "home_goals_conceded": (
                        home_stats[
                            "goals_conceded"
                        ]
                    ),
                    "away_goals_conceded": (
                        away_stats[
                            "goals_conceded"
                        ]
                    ),

                    "home_attack_matchup": (
                        home_attack_matchup
                    ),
                    "away_attack_matchup": (
                        away_attack_matchup
                    ),
                    "attack_matchup_difference": (
                        attack_matchup_difference
                    ),

                    "home_goal_difference": (
                        home_stats[
                            "goal_difference"
                        ]
                    ),
                    "away_goal_difference": (
                        away_stats[
                            "goal_difference"
                        ]
                    ),

                    "home_win_rate": (
                        home_stats[
                            "win_rate"
                        ]
                    ),
                    "away_win_rate": (
                        away_stats[
                            "win_rate"
                        ]
                    ),

                    "home_rest_days": (
                        home_rest_days
                    ),
                    "away_rest_days": (
                        away_rest_days
                    ),

                    "is_champions_league": int(
                        competition == "CL"
                    ),

                    "target": str(row.winner),
                }
            )

            home_goals = int(
                row.home_goals
            )

            away_goals = int(
                row.away_goals
            )

            self._update_elo(
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                ratings=ratings,
            )

            self._append_history(
                history=histories[
                    home_team
                ],
                goals_scored=home_goals,
                goals_conceded=away_goals,
            )

            self._append_history(
                history=histories[
                    away_team
                ],
                goals_scored=away_goals,
                goals_conceded=home_goals,
            )

            last_match_dates[
                home_team
            ] = match_date

            last_match_dates[
                away_team
            ] = match_date

        result = pd.DataFrame(
            feature_rows
        )

        numeric_columns = (
            self.FEATURE_COLUMNS
        )

        result[
            numeric_columns
        ] = result[
            numeric_columns
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return result

    def _history_features(
        self,
        history: Deque[MatchHistory],
    ) -> Dict[str, float]:
        if not history:
            return {
                "points": 1.0,
                "goals_scored": 1.2,
                "goals_conceded": 1.2,
                "goal_difference": 0.0,
                "win_rate": 0.33,
            }

        matches = list(history)

        points = np.mean(
            [
                match[2]
                for match in matches
            ]
        )

        goals_scored = np.mean(
            [
                match[0]
                for match in matches
            ]
        )

        goals_conceded = np.mean(
            [
                match[1]
                for match in matches
            ]
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
        home_rating = ratings[
            home_team
        ]

        away_rating = ratings[
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

        margin_multiplier = (
            1.0
            + np.log1p(goal_margin)
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
            home_rating
            + rating_change
        )

        ratings[away_team] = (
            away_rating
            - rating_change
        )

    @staticmethod
    def _rest_days(
        team: str,
        match_date: pd.Timestamp,
        last_match_dates: Dict[
            str,
            pd.Timestamp,
        ],
    ) -> float:
        previous_date = (
            last_match_dates.get(team)
        )

        if previous_date is None:
            return 7.0

        difference = (
            match_date
            - previous_date
        ).days

        return float(
            max(
                1,
                min(difference, 30),
            )
        )
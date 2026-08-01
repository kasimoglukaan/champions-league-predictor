from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict

import numpy as np
import pandas as pd

from src.features.feature_builder import (
    FeatureBuilder,
)
from src.features.league_strength import (
    LeagueStrengthResolver,
)
from src.features.match_statistics import (
    MatchHistory,
    append_result,
    summarize_history,
)


class LiveFeatureBuilder:
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
        self.home_advantage = (
            home_advantage
        )
        self.form_window = form_window

        self.ratings: Dict[
            str,
            float,
        ] = defaultdict(
            lambda: self.initial_elo
        )

        self.overall_histories: Dict[
            str,
            Deque[MatchHistory],
        ] = defaultdict(
            lambda: deque(
                maxlen=self.form_window
            )
        )

        self.home_histories: Dict[
            str,
            Deque[MatchHistory],
        ] = defaultdict(
            lambda: deque(
                maxlen=self.form_window
            )
        )

        self.away_histories: Dict[
            str,
            Deque[MatchHistory],
        ] = defaultdict(
            lambda: deque(
                maxlen=self.form_window
            )
        )

        self.last_match_dates: Dict[
            str,
            pd.Timestamp,
        ] = {}

        self.team_leagues: Dict[
            str,
            str,
        ] = {}

        self.latest_data_date = (
            pd.Timestamp.now(
                tz="UTC"
            )
        )

        self.is_fitted = False

    @property
    def histories(
        self,
    ) -> Dict[
        str,
        Deque[MatchHistory],
    ]:
        return self.overall_histories

    def fit(
        self,
        matches: pd.DataFrame,
    ) -> None:
        required_columns = {
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        }

        missing_columns = (
            required_columns
            - set(
                matches.columns
            )
        )

        if missing_columns:
            raise ValueError(
                "Missing match columns: "
                + ", ".join(
                    sorted(
                        missing_columns
                    )
                )
            )

        dataframe = matches.copy()

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            utc=True,
            errors="raise",
        )

        if (
            "competition"
            not in dataframe.columns
        ):
            dataframe[
                "competition"
            ] = "UNKNOWN"

        dataframe = dataframe.sort_values(
            "date"
        ).reset_index(
            drop=True
        )

        self.ratings = defaultdict(
            lambda: self.initial_elo
        )

        self.overall_histories = defaultdict(
            lambda: deque(
                maxlen=self.form_window
            )
        )

        self.home_histories = defaultdict(
            lambda: deque(
                maxlen=self.form_window
            )
        )

        self.away_histories = defaultdict(
            lambda: deque(
                maxlen=self.form_window
            )
        )

        self.last_match_dates = {}
        self.team_leagues = {}

        for row in dataframe.itertuples(
            index=False
        ):
            home_team = str(
                row.home_team
            )

            away_team = str(
                row.away_team
            )

            home_goals = int(
                row.home_goals
            )

            away_goals = int(
                row.away_goals
            )

            competition = str(
                row.competition
            )

            self._update_elo(
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
            )

            append_result(
                history=(
                    self.overall_histories[
                        home_team
                    ]
                ),
                goals_scored=(
                    home_goals
                ),
                goals_conceded=(
                    away_goals
                ),
            )

            append_result(
                history=(
                    self.overall_histories[
                        away_team
                    ]
                ),
                goals_scored=(
                    away_goals
                ),
                goals_conceded=(
                    home_goals
                ),
            )

            append_result(
                history=(
                    self.home_histories[
                        home_team
                    ]
                ),
                goals_scored=(
                    home_goals
                ),
                goals_conceded=(
                    away_goals
                ),
            )

            append_result(
                history=(
                    self.away_histories[
                        away_team
                    ]
                ),
                goals_scored=(
                    away_goals
                ),
                goals_conceded=(
                    home_goals
                ),
            )

            self.last_match_dates[
                home_team
            ] = row.date

            self.last_match_dates[
                away_team
            ] = row.date

            if (
                LeagueStrengthResolver
                .is_domestic(
                    competition
                )
            ):
                canonical_league = (
                    LeagueStrengthResolver
                    .canonical_competition(
                        competition
                    )
                )

                self.team_leagues[
                    home_team
                ] = canonical_league

                self.team_leagues[
                    away_team
                ] = canonical_league

        if not dataframe.empty:
            self.latest_data_date = (
                dataframe[
                    "date"
                ].max()
            )

        self.is_fitted = True

    def build_match_features(
        self,
        home_team: str,
        away_team: str,
        competition: str = "CL",
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "LiveFeatureBuilder must "
                "be fitted first."
            )

        if home_team == away_team:
            raise ValueError(
                "Home and away teams "
                "must be different."
            )

        home_elo = float(
            self.ratings[
                home_team
            ]
        )

        away_elo = float(
            self.ratings[
                away_team
            ]
        )

        home_stats = (
            summarize_history(
                self.overall_histories[
                    home_team
                ]
            )
        )

        away_stats = (
            summarize_history(
                self.overall_histories[
                    away_team
                ]
            )
        )

        home_home_stats = (
            summarize_history(
                self.home_histories[
                    home_team
                ]
            )
        )

        away_away_stats = (
            summarize_history(
                self.away_histories[
                    away_team
                ]
            )
        )

        home_league = (
            self.team_leagues.get(
                home_team
            )
        )

        away_league = (
            self.team_leagues.get(
                away_team
            )
        )

        home_league_strength = (
            LeagueStrengthResolver
            .resolve_team_strength(
                team_league=(
                    home_league
                ),
                current_competition=(
                    competition
                ),
            )
        )

        away_league_strength = (
            LeagueStrengthResolver
            .resolve_team_strength(
                team_league=(
                    away_league
                ),
                current_competition=(
                    competition
                ),
            )
        )

        league_strength_difference = (
            home_league_strength
            - away_league_strength
        )

        cross_league_match = int(
            bool(
                home_league
                and away_league
                and home_league
                != away_league
            )
        )

        elo_difference = (
            home_elo
            + self.home_advantage
            - away_elo
        )

        absolute_elo_difference = abs(
            home_elo
            - away_elo
        )

        squared_elo_difference = (
            elo_difference ** 2
        )

        close_elo_match = int(
            absolute_elo_difference
            < 75
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

        prediction_date = max(
            pd.Timestamp.now(
                tz="UTC"
            ),
            self.latest_data_date
            + pd.Timedelta(
                days=1
            ),
        )

        feature_row = {
            "home_elo": (
                home_elo
            ),
            "away_elo": (
                away_elo
            ),
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

            "home_league_strength": (
                home_league_strength
            ),
            "away_league_strength": (
                away_league_strength
            ),
            "league_strength_difference": (
                league_strength_difference
            ),
            "cross_league_match": (
                cross_league_match
            ),

            "home_form_points": (
                home_stats[
                    "points"
                ]
            ),
            "away_form_points": (
                away_stats[
                    "points"
                ]
            ),
            "form_difference": (
                home_stats[
                    "points"
                ]
                - away_stats[
                    "points"
                ]
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

            "home_home_form_points": (
                home_home_stats[
                    "points"
                ]
            ),
            "away_away_form_points": (
                away_away_stats[
                    "points"
                ]
            ),
            "home_away_form_difference": (
                home_home_stats[
                    "points"
                ]
                - away_away_stats[
                    "points"
                ]
            ),

            "home_home_goals_scored": (
                home_home_stats[
                    "goals_scored"
                ]
            ),
            "away_away_goals_scored": (
                away_away_stats[
                    "goals_scored"
                ]
            ),
            "home_home_goals_conceded": (
                home_home_stats[
                    "goals_conceded"
                ]
            ),
            "away_away_goals_conceded": (
                away_away_stats[
                    "goals_conceded"
                ]
            ),

            "home_home_goal_difference": (
                home_home_stats[
                    "goal_difference"
                ]
            ),
            "away_away_goal_difference": (
                away_away_stats[
                    "goal_difference"
                ]
            ),

            "home_home_win_rate": (
                home_home_stats[
                    "win_rate"
                ]
            ),
            "away_away_win_rate": (
                away_away_stats[
                    "win_rate"
                ]
            ),

            "home_rest_days": (
                self._rest_days(
                    team=home_team,
                    prediction_date=(
                        prediction_date
                    ),
                )
            ),
            "away_rest_days": (
                self._rest_days(
                    team=away_team,
                    prediction_date=(
                        prediction_date
                    ),
                )
            ),

            "is_champions_league": int(
                LeagueStrengthResolver
                .is_champions_league(
                    competition
                )
            ),
        }

        dataframe = pd.DataFrame(
            [
                feature_row
            ]
        )

        dataframe = dataframe.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        return dataframe[
            FeatureBuilder.FEATURE_COLUMNS
        ]

    def get_teams(
        self,
    ) -> list:
        teams = set(
            self.ratings.keys()
        )

        teams.update(
            self.overall_histories.keys()
        )

        return sorted(
            teams
        )

    def get_team_summary(
        self,
        team_name: str,
    ) -> dict:
        overall_statistics = (
            summarize_history(
                self.overall_histories[
                    team_name
                ]
            )
        )

        home_statistics = (
            summarize_history(
                self.home_histories[
                    team_name
                ]
            )
        )

        away_statistics = (
            summarize_history(
                self.away_histories[
                    team_name
                ]
            )
        )

        league = (
            self.team_leagues.get(
                team_name
            )
        )

        return {
            "team": (
                team_name
            ),
            "elo": round(
                self.ratings[
                    team_name
                ],
                1,
            ),
            "league": (
                league
                or "UNKNOWN"
            ),
            "league_strength": round(
                LeagueStrengthResolver
                .resolve_team_strength(
                    team_league=league,
                    current_competition=(
                        "UNKNOWN"
                    ),
                ),
                3,
            ),
            "form_points": round(
                overall_statistics[
                    "points"
                ],
                2,
            ),
            "goals_scored": round(
                overall_statistics[
                    "goals_scored"
                ],
                2,
            ),
            "goals_conceded": round(
                overall_statistics[
                    "goals_conceded"
                ],
                2,
            ),
            "win_rate": round(
                overall_statistics[
                    "win_rate"
                ],
                3,
            ),
            "home_form_points": round(
                home_statistics[
                    "points"
                ],
                2,
            ),
            "home_win_rate": round(
                home_statistics[
                    "win_rate"
                ],
                3,
            ),
            "away_form_points": round(
                away_statistics[
                    "points"
                ],
                2,
            ),
            "away_win_rate": round(
                away_statistics[
                    "win_rate"
                ],
                3,
            ),
        }

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
            home_goals
            - away_goals
        )

        margin_multiplier = (
            1.0
            + np.log1p(
                goal_margin
            )
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

        self.ratings[
            home_team
        ] = (
            home_rating
            + rating_change
        )

        self.ratings[
            away_team
        ] = (
            away_rating
            - rating_change
        )

    def _rest_days(
        self,
        team: str,
        prediction_date: pd.Timestamp,
    ) -> float:
        previous_date = (
            self.last_match_dates.get(
                team
            )
        )

        if previous_date is None:
            return 7.0

        difference = (
            prediction_date
            - previous_date
        ).days

        return float(
            max(
                1,
                min(
                    difference,
                    30,
                ),
            )
        )
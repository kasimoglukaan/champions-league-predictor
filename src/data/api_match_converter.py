from typing import Any, Dict, List

import pandas as pd


class ApiMatchConverter:
    COLUMNS = [
        "match_id",
        "date",
        "competition",
        "season",
        "stage",
        "matchday",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "winner",
    ]

    def convert(
        self,
        api_matches: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        rows = []

        for match in api_matches:
            score = match.get("score", {})
            full_time = score.get("fullTime", {})

            home_goals = full_time.get("home")
            away_goals = full_time.get("away")

            if (
                home_goals is None
                or away_goals is None
            ):
                continue

            competition = match.get(
                "competition",
                {},
            )

            season = match.get("season", {})
            home_team = match.get("homeTeam", {})
            away_team = match.get("awayTeam", {})

            rows.append(
                {
                    "match_id": match.get("id"),
                    "date": match.get("utcDate"),
                    "competition": competition.get(
                        "code"
                    ),
                    "season": self._get_season_year(
                        season
                    ),
                    "stage": match.get("stage"),
                    "matchday": match.get("matchday"),
                    "home_team": home_team.get(
                        "name"
                    ),
                    "away_team": away_team.get(
                        "name"
                    ),
                    "home_goals": int(home_goals),
                    "away_goals": int(away_goals),
                    "winner": self._get_winner(
                        int(home_goals),
                        int(away_goals),
                    ),
                }
            )

        dataframe = pd.DataFrame(
            rows,
            columns=self.COLUMNS,
        )

        if dataframe.empty:
            return dataframe

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            utc=True,
        )

        dataframe = (
            dataframe
            .drop_duplicates(subset=["match_id"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        return dataframe

    @staticmethod
    def _get_season_year(
        season: Dict[str, Any],
    ) -> int:
        start_date = season.get("startDate")

        if not start_date:
            return 0

        return int(str(start_date)[:4])

    @staticmethod
    def _get_winner(
        home_goals: int,
        away_goals: int,
    ) -> str:
        if home_goals > away_goals:
            return "H"

        if home_goals < away_goals:
            return "A"

        return "D"
from pathlib import Path

import pandas as pd

from src.models.match import Match


class MatchRepository:
    REQUIRED_COLUMNS = {
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    }

    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)

    def load_matches(self) -> list[Match]:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Match data file was not found: {self.csv_path}"
            )

        dataframe = pd.read_csv(self.csv_path)

        self._validate_columns(dataframe)

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="raise",
        )

        dataframe = dataframe.sort_values("date")

        matches: list[Match] = []

        for row in dataframe.itertuples(index=False):
            match = Match(
                date=row.date.to_pydatetime(),
                home_team=str(row.home_team).strip(),
                away_team=str(row.away_team).strip(),
                home_goals=int(row.home_goals),
                away_goals=int(row.away_goals),
            )

            matches.append(match)

        return matches

    def get_team_names(self) -> list[str]:
        matches = self.load_matches()

        teams = set()

        for match in matches:
            teams.add(match.home_team)
            teams.add(match.away_team)

        return sorted(teams)

    def _validate_columns(self, dataframe: pd.DataFrame) -> None:
        missing_columns = (
            self.REQUIRED_COLUMNS - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "CSV file is missing these columns: "
                + ", ".join(sorted(missing_columns))
            )
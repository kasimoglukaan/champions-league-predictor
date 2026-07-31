from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


EXISTING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_matches.csv"
)

LEAGUE_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "leagues"
)

EXPANDED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_matches_expanded_candidate.csv"
)

DOWNLOAD_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_league_download_report.csv"
)


BASE_URL = (
    "https://www.football-data.co.uk/"
    "mmz4281/{season}/{division}.csv"
)


LEAGUES = {
    "premier_league": {
        "name": "Premier League",
        "country": "England",
        "division": "E0",
        "competition": "EPL",
    },
    "championship": {
        "name": "EFL Championship",
        "country": "England",
        "division": "E1",
        "competition": "EFL_CHAMP",
    },
    "bundesliga": {
        "name": "Bundesliga",
        "country": "Germany",
        "division": "D1",
        "competition": "BUNDESLIGA",
    },
    "la_liga": {
        "name": "La Liga",
        "country": "Spain",
        "division": "SP1",
        "competition": "LA_LIGA",
    },
    "serie_a": {
        "name": "Serie A",
        "country": "Italy",
        "division": "I1",
        "competition": "SERIE_A",
    },
    "ligue_1": {
        "name": "Ligue 1",
        "country": "France",
        "division": "F1",
        "competition": "LIGUE_1",
    },
    "eredivisie": {
        "name": "Eredivisie",
        "country": "Netherlands",
        "division": "N1",
        "competition": "EREDIVISIE",
    },
    "primeira_liga": {
        "name": "Primeira Liga",
        "country": "Portugal",
        "division": "P1",
        "competition": "PRIMEIRA_LIGA",
    },
    "super_lig": {
        "name": "Turkish Super League",
        "country": "Turkey",
        "division": "T1",
        "competition": "SUPER_LIG",
    },
    "belgian_first_division": {
        "name": "Belgian First Division",
        "country": "Belgium",
        "division": "B1",
        "competition": "BELGIUM_1",
    },
}


SEASONS = [
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
]


OUTPUT_COLUMNS = [
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
    "country",
    "league_name",
    "data_source",
]


TEAM_NAME_ALIASES = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    "Brighton": "Brighton and Hove Albion",
    "Tottenham": "Tottenham Hotspur",

    "Bayern Munich": "FC Bayern München",
    "Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer 04 Leverkusen",
    "M'gladbach": "Borussia Mönchengladbach",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Mainz": "FSV Mainz 05",
    "RB Leipzig": "RB Leipzig",
    "Schalke 04": "FC Schalke 04",

    "Barcelona": "FC Barcelona",
    "Real Madrid": "Real Madrid CF",
    "Ath Madrid": "Club Atlético de Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Sociedad": "Real Sociedad",
    "Betis": "Real Betis",
    "Celta": "Celta Vigo",
    "Vallecano": "Rayo Vallecano",

    "Inter": "Inter Milan",
    "Milan": "AC Milan",
    "Roma": "AS Roma",
    "Lazio": "SS Lazio",
    "Verona": "Hellas Verona",

    "Paris SG": "Paris Saint-Germain FC",
    "Marseille": "Olympique Marseille",
    "Lyon": "Olympique Lyonnais",
    "Monaco": "AS Monaco",
    "St Etienne": "AS Saint-Étienne",

    "Ajax": "Ajax",
    "PSV Eindhoven": "PSV Eindhoven",
    "Feyenoord": "Feyenoord Rotterdam",

    "Sp Lisbon": "Sporting CP",
    "Porto": "FC Porto",
    "Benfica": "Benfica",
    "Braga": "Sporting Braga",

    "Besiktas": "Beşiktaş JK",
    "Fenerbahce": "Fenerbahçe",
    "Galatasaray": "Galatasaray SK",
    "Trabzonspor": "Trabzonspor",
    "Basaksehir": "İstanbul Başakşehir",
    "Kasimpasa": "Kasımpaşa SK",
    "Goztep": "Göztepe",
    "Rizespor": "Çaykur Rizespor",
    "Gaziantep": "Gaziantep FK",

    "Anderlecht": "RSC Anderlecht",
    "Club Brugge": "Club Brugge KV",
    "Genk": "KRC Genk",
    "Gent": "KAA Gent",
    "Standard": "Standard Liège",
}


def season_label(
    season_code: str,
) -> str:
    if len(season_code) != 4:
        return season_code

    start_year = int(
        season_code[:2]
    )

    end_year = int(
        season_code[2:]
    )

    start_full_year = (
        2000 + start_year
        if start_year < 70
        else 1900 + start_year
    )

    end_full_year = (
        2000 + end_year
        if end_year < 70
        else 1900 + end_year
    )

    return (
        f"{start_full_year}-"
        f"{str(end_full_year)[-2:]}"
    )


def canonical_team_name(
    team_name: object,
) -> str:
    cleaned_name = str(
        team_name
    ).strip()

    if not cleaned_name:
        return ""

    return TEAM_NAME_ALIASES.get(
        cleaned_name,
        cleaned_name,
    )


def determine_winner(
    home_goals: int,
    away_goals: int,
) -> str:
    if home_goals > away_goals:
        return "H"

    if away_goals > home_goals:
        return "A"

    return "D"


def create_match_id(
    competition: str,
    season: str,
    date: str,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> str:
    identity_text = "|".join(
        [
            competition,
            season,
            date,
            home_team,
            away_team,
            str(home_goals),
            str(away_goals),
        ]
    )

    digest = hashlib.sha1(
        identity_text.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        "league_"
        + digest[:16]
    )


def request_csv(
    url: str,
    timeout_seconds: int = 30,
) -> Optional[pd.DataFrame]:
    response = requests.get(
        url,
        timeout=timeout_seconds,
        headers={
            "User-Agent": (
                "EdgeXI-Football-Analytics/1.0"
            ),
        },
    )

    if response.status_code == 404:
        return None

    response.raise_for_status()

    content = response.content

    if not content:
        return None

    try:
        dataframe = pd.read_csv(
            io.BytesIO(content)
        )

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        dataframe = pd.read_csv(
            io.BytesIO(content),
            encoding="latin-1",
            on_bad_lines="skip",
        )

    if dataframe.empty:
        return None

    return dataframe


def normalize_downloaded_data(
    source_data: pd.DataFrame,
    league_key: str,
    league_config: dict,
    season_code: str,
) -> pd.DataFrame:
    required_columns = {
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    }

    missing_columns = (
        required_columns
        - set(source_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    normalized = source_data[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
        ]
    ].copy()

    normalized = normalized.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        }
    )

    normalized["date"] = pd.to_datetime(
        normalized["date"],
        dayfirst=True,
        errors="coerce",
    )

    normalized["home_goals"] = (
        pd.to_numeric(
            normalized["home_goals"],
            errors="coerce",
        )
    )

    normalized["away_goals"] = (
        pd.to_numeric(
            normalized["away_goals"],
            errors="coerce",
        )
    )

    normalized = normalized.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    normalized["home_goals"] = (
        normalized[
            "home_goals"
        ].astype(int)
    )

    normalized["away_goals"] = (
        normalized[
            "away_goals"
        ].astype(int)
    )

    normalized["home_team"] = (
        normalized[
            "home_team"
        ].map(
            canonical_team_name
        )
    )

    normalized["away_team"] = (
        normalized[
            "away_team"
        ].map(
            canonical_team_name
        )
    )

    normalized = normalized[
        (
            normalized["home_team"] != ""
        )
        & (
            normalized["away_team"] != ""
        )
        & (
            normalized["home_team"]
            != normalized["away_team"]
        )
    ].copy()

    normalized["date"] = (
        normalized["date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    normalized["competition"] = (
        league_config[
            "competition"
        ]
    )

    normalized["season"] = (
        season_label(
            season_code
        )
    )

    normalized["stage"] = (
        "LEAGUE"
    )

    normalized["matchday"] = pd.NA

    normalized["winner"] = [
        determine_winner(
            int(home_goals),
            int(away_goals),
        )
        for home_goals, away_goals
        in zip(
            normalized["home_goals"],
            normalized["away_goals"],
        )
    ]

    normalized["country"] = (
        league_config["country"]
    )

    normalized["league_name"] = (
        league_config["name"]
    )

    normalized["data_source"] = (
        "football-data.co.uk"
    )

    normalized["match_id"] = [
        create_match_id(
            competition=(
                league_config[
                    "competition"
                ]
            ),
            season=(
                season_label(
                    season_code
                )
            ),
            date=str(date),
            home_team=str(
                home_team
            ),
            away_team=str(
                away_team
            ),
            home_goals=int(
                home_goals
            ),
            away_goals=int(
                away_goals
            ),
        )
        for (
            date,
            home_team,
            away_team,
            home_goals,
            away_goals,
        ) in zip(
            normalized["date"],
            normalized["home_team"],
            normalized["away_team"],
            normalized["home_goals"],
            normalized["away_goals"],
        )
    ]

    normalized = normalized[
        OUTPUT_COLUMNS
    ].copy()

    normalized = (
        normalized.drop_duplicates(
            subset=[
                "competition",
                "season",
                "date",
                "home_team",
                "away_team",
            ],
            keep="last",
        )
    )

    return normalized.reset_index(
        drop=True
    )


def load_existing_matches(
) -> pd.DataFrame:
    if not EXISTING_DATA_PATH.exists():
        raise FileNotFoundError(
            "Existing dataset was not found: "
            f"{EXISTING_DATA_PATH}"
        )

    existing = pd.read_csv(
        EXISTING_DATA_PATH
    )

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    }

    missing_columns = (
        required_columns
        - set(existing.columns)
    )

    if missing_columns:
        raise ValueError(
            "Existing dataset is missing: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    existing = existing.copy()

    existing["date"] = pd.to_datetime(
        existing["date"],
        errors="coerce",
        utc=True,
    )

    existing["date"] = (
        existing["date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    existing["home_team"] = (
        existing[
            "home_team"
        ].map(
            canonical_team_name
        )
    )

    existing["away_team"] = (
        existing[
            "away_team"
        ].map(
            canonical_team_name
        )
    )

    existing["home_goals"] = (
        pd.to_numeric(
            existing["home_goals"],
            errors="coerce",
        )
    )

    existing["away_goals"] = (
        pd.to_numeric(
            existing["away_goals"],
            errors="coerce",
        )
    )

    existing = existing.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    existing["home_goals"] = (
        existing[
            "home_goals"
        ].astype(int)
    )

    existing["away_goals"] = (
        existing[
            "away_goals"
        ].astype(int)
    )

    if "competition" not in existing:
        existing["competition"] = (
            "UNKNOWN"
        )

    if "season" not in existing:
        existing["season"] = (
            existing["date"]
            .astype(str)
            .str[:4]
        )

    if "stage" not in existing:
        existing["stage"] = (
            "UNKNOWN"
        )

    if "matchday" not in existing:
        existing["matchday"] = (
            pd.NA
        )

    if "winner" not in existing:
        existing["winner"] = [
            determine_winner(
                home_goals,
                away_goals,
            )
            for home_goals, away_goals
            in zip(
                existing["home_goals"],
                existing["away_goals"],
            )
        ]

    if "country" not in existing:
        existing["country"] = (
            ""
        )

    if "league_name" not in existing:
        existing["league_name"] = (
            existing[
                "competition"
            ].astype(str)
        )

    if "data_source" not in existing:
        existing["data_source"] = (
            "existing_project_data"
        )

    if "match_id" not in existing:
        existing["match_id"] = [
            create_match_id(
                competition=str(
                    competition
                ),
                season=str(
                    season
                ),
                date=str(
                    date
                ),
                home_team=str(
                    home_team
                ),
                away_team=str(
                    away_team
                ),
                home_goals=int(
                    home_goals
                ),
                away_goals=int(
                    away_goals
                ),
            )
            for (
                competition,
                season,
                date,
                home_team,
                away_team,
                home_goals,
                away_goals,
            ) in zip(
                existing["competition"],
                existing["season"],
                existing["date"],
                existing["home_team"],
                existing["away_team"],
                existing["home_goals"],
                existing["away_goals"],
            )
        ]

    return existing[
        OUTPUT_COLUMNS
    ].copy()


def create_summary(
    master_data: pd.DataFrame,
) -> None:
    print()
    print("=" * 84)
    print("EXPANDED DATASET SUMMARY")
    print("=" * 84)

    print(
        f"Total matches: "
        f"{len(master_data):,}"
    )

    unique_teams = sorted(
        set(
            master_data[
                "home_team"
            ].astype(str)
        )
        | set(
            master_data[
                "away_team"
            ].astype(str)
        )
    )

    print(
        f"Unique teams: "
        f"{len(unique_teams):,}"
    )

    print(
        f"Competitions: "
        f"{master_data['competition'].nunique():,}"
    )

    print()
    print("MATCHES BY COMPETITION")
    print("-" * 84)

    competition_counts = (
        master_data[
            "league_name"
        ]
        .value_counts()
    )

    for (
        competition_name,
        match_count,
    ) in competition_counts.items():
        print(
            f"{competition_name:<35}"
            f"{match_count:>8,}"
        )

    print()
    print(
        "Candidate master dataset saved to:"
    )
    print(
        EXPANDED_DATA_PATH
    )

    print("=" * 84)


def main() -> None:
    LEAGUE_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPANDED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_downloaded_frames = []
    report_rows = []

    print()
    print("=" * 84)
    print("MULTI-LEAGUE HISTORICAL DATA DOWNLOAD")
    print("=" * 84)

    for (
        league_key,
        league_config,
    ) in LEAGUES.items():
        league_frames = []

        print()
        print(
            league_config["name"]
        )
        print(
            "-" * len(
                league_config["name"]
            )
        )

        for season_code in SEASONS:
            url = BASE_URL.format(
                season=season_code,
                division=(
                    league_config[
                        "division"
                    ]
                ),
            )

            try:
                source_data = (
                    request_csv(
                        url
                    )
                )

                if source_data is None:
                    print(
                        f"[NOT FOUND] "
                        f"{season_label(season_code)}"
                    )

                    report_rows.append(
                        {
                            "league": (
                                league_config[
                                    "name"
                                ]
                            ),
                            "league_key": (
                                league_key
                            ),
                            "season": (
                                season_label(
                                    season_code
                                )
                            ),
                            "url": url,
                            "status": (
                                "NOT_FOUND"
                            ),
                            "matches": 0,
                            "error": "",
                        }
                    )

                    continue

                normalized_data = (
                    normalize_downloaded_data(
                        source_data=(
                            source_data
                        ),
                        league_key=(
                            league_key
                        ),
                        league_config=(
                            league_config
                        ),
                        season_code=(
                            season_code
                        ),
                    )
                )

                league_frames.append(
                    normalized_data
                )

                all_downloaded_frames.append(
                    normalized_data
                )

                print(
                    f"[OK] "
                    f"{season_label(season_code)}: "
                    f"{len(normalized_data):,} matches"
                )

                report_rows.append(
                    {
                        "league": (
                            league_config[
                                "name"
                            ]
                        ),
                        "league_key": (
                            league_key
                        ),
                        "season": (
                            season_label(
                                season_code
                            )
                        ),
                        "url": url,
                        "status": "OK",
                        "matches": (
                            len(
                                normalized_data
                            )
                        ),
                        "error": "",
                    }
                )

            except Exception as error:
                print(
                    f"[ERROR] "
                    f"{season_label(season_code)}: "
                    f"{error}"
                )

                report_rows.append(
                    {
                        "league": (
                            league_config[
                                "name"
                            ]
                        ),
                        "league_key": (
                            league_key
                        ),
                        "season": (
                            season_label(
                                season_code
                            )
                        ),
                        "url": url,
                        "status": "ERROR",
                        "matches": 0,
                        "error": str(
                            error
                        ),
                    }
                )

        if league_frames:
            league_dataset = pd.concat(
                league_frames,
                ignore_index=True,
            )

            league_dataset = (
                league_dataset
                .drop_duplicates(
                    subset=[
                        "competition",
                        "season",
                        "date",
                        "home_team",
                        "away_team",
                    ],
                    keep="last",
                )
                .sort_values(
                    by=[
                        "date",
                        "home_team",
                        "away_team",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            league_output_path = (
                LEAGUE_DATA_DIRECTORY
                / f"{league_key}.csv"
            )

            league_dataset.to_csv(
                league_output_path,
                index=False,
            )

            print(
                f"Saved: "
                f"{league_output_path}"
            )

    report = pd.DataFrame(
        report_rows
    )

    report.to_csv(
        DOWNLOAD_REPORT_PATH,
        index=False,
    )

    if not all_downloaded_frames:
        raise RuntimeError(
            "No league data could be downloaded."
        )

    downloaded_master = pd.concat(
        all_downloaded_frames,
        ignore_index=True,
    )

    existing_matches = (
        load_existing_matches()
    )

    master_data = pd.concat(
        [
            existing_matches,
            downloaded_master,
        ],
        ignore_index=True,
    )

    master_data["date"] = (
        pd.to_datetime(
            master_data["date"],
            errors="coerce",
        )
    )

    master_data = master_data.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    master_data["date"] = (
        master_data["date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    master_data = (
        master_data
        .drop_duplicates(
            subset=[
                "competition",
                "season",
                "date",
                "home_team",
                "away_team",
            ],
            keep="last",
        )
        .sort_values(
            by=[
                "date",
                "competition",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    master_data.to_csv(
        EXPANDED_DATA_PATH,
        index=False,
    )

    create_summary(
        master_data
    )


if __name__ == "__main__":
    main()
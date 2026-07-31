from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import tomli


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.services.odds_api_service import (
    OddsAPIService,
)
from src.services.team_matching_service import (
    TeamMatchingService,
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_matches.csv"
)

SECRETS_PATH = (
    PROJECT_ROOT
    / ".streamlit"
    / "secrets.toml"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "active_team_coverage.csv"
)


TARGET_LEAGUE_KEYS = {
    "soccer_epl": "Premier League",
    "soccer_efl_champ": "EFL Championship",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_spain_la_liga": "La Liga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
    "soccer_netherlands_eredivisie": "Eredivisie",
    "soccer_portugal_primeira_liga": "Primeira Liga",
    "soccer_turkey_super_league": "Turkish Super League",
    "soccer_belgium_first_div": "Belgian First Division",
    "soccer_scotland_premiership": "Scottish Premiership",
    "soccer_uefa_champs_league": "UEFA Champions League",
    "soccer_uefa_champs_league_qualification": (
        "Champions League Qualification"
    ),
    "soccer_uefa_europa_league": "UEFA Europa League",
    "soccer_uefa_europa_conference_league": (
        "UEFA Conference League"
    ),
}


def load_api_key() -> str:
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            "Secrets file was not found: "
            f"{SECRETS_PATH}"
        )

    with SECRETS_PATH.open(
        "rb"
    ) as secrets_file:
        secrets = tomli.load(
            secrets_file
        )

    api_key = str(
        secrets.get(
            "ODDS_API_KEY",
            "",
        )
    ).strip()

    if not api_key:
        raise ValueError(
            "ODDS_API_KEY is missing from "
            ".streamlit/secrets.toml."
        )

    return api_key


def load_dataset_teams() -> list[str]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Match dataset was not found: "
            f"{DATA_PATH}"
        )

    matches = pd.read_csv(
        DATA_PATH
    )

    required_columns = {
        "home_team",
        "away_team",
    }

    missing_columns = (
        required_columns
        - set(matches.columns)
    )

    if missing_columns:
        raise ValueError(
            "The dataset is missing required "
            "columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    home_teams = (
        matches["home_team"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    away_teams = (
        matches["away_team"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    teams = set(
        home_teams
    )

    teams.update(
        away_teams
    )

    teams.discard("")

    return sorted(
        teams
    )


def collect_unique_event_teams(
    events,
) -> list[str]:
    teams: set[str] = set()

    for event in events:
        home_team = str(
            event.home_team
        ).strip()

        away_team = str(
            event.away_team
        ).strip()

        if home_team:
            teams.add(
                home_team
            )

        if away_team:
            teams.add(
                away_team
            )

    return sorted(
        teams
    )


def create_result_row(
    league_name: str,
    sport_key: str,
    api_team: str,
    match_result,
) -> dict:
    best_candidate = ""
    best_overlap = ""

    if match_result.candidates:
        first_candidate = (
            match_result.candidates[0]
        )

        best_candidate = (
            first_candidate.team_name
        )

        best_overlap = ", ".join(
            first_candidate
            .distinctive_overlap
        )

    return {
        "league": league_name,
        "sport_key": sport_key,
        "api_team": api_team,
        "status": (
            "MATCHED"
            if match_result.accepted
            else "MISSING"
        ),
        "matched_model_team": (
            match_result.matched_team_name
            or ""
        ),
        "best_candidate": (
            best_candidate
        ),
        "best_score": (
            match_result.score
        ),
        "second_best_score": (
            match_result
            .second_best_score
        ),
        "score_margin": (
            match_result.score_margin
        ),
        "distinctive_overlap": (
            best_overlap
        ),
        "reason": (
            match_result.reason
        ),
    }


def print_missing_teams(
    report: pd.DataFrame,
) -> None:
    missing_report = report[
        report["status"]
        == "MISSING"
    ]

    print()
    print("=" * 80)
    print("MISSING OR UNSAFE TEAMS")
    print("=" * 80)

    if missing_report.empty:
        print(
            "No missing teams were found."
        )
        return

    for league_name, league_group in (
        missing_report.groupby(
            "league",
            sort=True,
        )
    ):
        print()
        print(
            league_name
        )
        print(
            "-" * len(
                league_name
            )
        )

        for team_name in sorted(
            league_group[
                "api_team"
            ].tolist()
        ):
            print(
                f"- {team_name}"
            )


def main() -> None:
    api_key = load_api_key()

    dataset_teams = (
        load_dataset_teams()
    )

    odds_service = (
        OddsAPIService(
            api_key=api_key,
            region="eu",
        )
    )

    matching_service = (
        TeamMatchingService(
            minimum_score=0.84,
            minimum_margin=0.12,
            candidate_limit=5,
        )
    )

    active_leagues = (
        odds_service
        .get_active_soccer_leagues()
    )

    available_target_leagues = {
        sport_key: (
            active_leagues.get(
                sport_key,
                display_name,
            )
        )
        for sport_key, display_name
        in TARGET_LEAGUE_KEYS.items()
        if sport_key in active_leagues
    }

    print()
    print("=" * 80)
    print("ACTIVE TEAM COVERAGE ANALYSIS")
    print("=" * 80)
    print(
        f"Dataset teams: "
        f"{len(dataset_teams):,}"
    )
    print(
        f"Target active leagues: "
        f"{len(available_target_leagues):,}"
    )
    print()

    rows: list[dict] = []

    for (
        sport_key,
        league_name,
    ) in available_target_leagues.items():
        try:
            events = (
                odds_service
                .get_league_events(
                    sport_key=(
                        sport_key
                    )
                )
            )

        except Exception as error:
            print(
                f"[ERROR] {league_name}: "
                f"{error}"
            )
            continue

        league_teams = (
            collect_unique_event_teams(
                events
            )
        )

        matched_count = 0
        missing_count = 0

        for api_team in league_teams:
            match_result = (
                matching_service.match(
                    api_team_name=(
                        api_team
                    ),
                    model_teams=(
                        dataset_teams
                    ),
                )
            )

            if match_result.accepted:
                matched_count += 1

            else:
                missing_count += 1

            rows.append(
                create_result_row(
                    league_name=(
                        league_name
                    ),
                    sport_key=(
                        sport_key
                    ),
                    api_team=api_team,
                    match_result=(
                        match_result
                    ),
                )
            )

        print(
            f"{league_name}: "
            f"{len(league_teams):>3} teams | "
            f"{matched_count:>3} matched | "
            f"{missing_count:>3} missing"
        )

    report = pd.DataFrame(
        rows
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if report.empty:
        report.to_csv(
            REPORT_PATH,
            index=False,
        )

        print()
        print(
            "No active team data was returned."
        )
        return

    report = report.sort_values(
        by=[
            "status",
            "league",
            "api_team",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    total_teams = len(
        report
    )

    matched_teams = int(
        (
            report["status"]
            == "MATCHED"
        ).sum()
    )

    missing_teams = (
        total_teams
        - matched_teams
    )

    coverage = (
        matched_teams
        / total_teams
        if total_teams
        else 0.0
    )

    print()
    print("=" * 80)
    print("COVERAGE SUMMARY")
    print("=" * 80)
    print(
        f"Active team entries: "
        f"{total_teams:,}"
    )
    print(
        f"Safely matched: "
        f"{matched_teams:,}"
    )
    print(
        f"Missing or unsafe: "
        f"{missing_teams:,}"
    )
    print(
        f"Coverage: "
        f"{coverage:.2%}"
    )

    print_missing_teams(
        report
    )

    print()
    print("=" * 80)
    print("REPORT")
    print("=" * 80)
    print(
        f"Saved to: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
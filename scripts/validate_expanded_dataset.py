from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_matches_expanded_candidate.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded_dataset_validation_report.json"
)

DUPLICATE_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded_dataset_duplicates.csv"
)

INVALID_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded_dataset_invalid_rows.csv"
)


REQUIRED_COLUMNS = {
    "match_id",
    "date",
    "competition",
    "season",
    "stage",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "winner",
    "league_name",
    "data_source",
}


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expanded dataset not found: {DATA_PATH}"
        )

    dataframe = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return dataframe


def normalize_for_validation(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    validated = dataframe.copy()

    validated["date"] = pd.to_datetime(
        validated["date"],
        errors="coerce",
    )

    validated["home_team"] = (
        validated["home_team"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    validated["away_team"] = (
        validated["away_team"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    validated["competition"] = (
        validated["competition"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    validated["league_name"] = (
        validated["league_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    validated["home_goals"] = pd.to_numeric(
        validated["home_goals"],
        errors="coerce",
    )

    validated["away_goals"] = pd.to_numeric(
        validated["away_goals"],
        errors="coerce",
    )

    validated["winner"] = (
        validated["winner"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return validated


def determine_expected_winner(
    home_goals: float,
    away_goals: float,
) -> str:
    if home_goals > away_goals:
        return "H"

    if away_goals > home_goals:
        return "A"

    return "D"


def create_invalid_row_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    invalid_reasons = []

    for index, row in dataframe.iterrows():
        row_reasons = []

        if pd.isna(row["date"]):
            row_reasons.append(
                "invalid_date"
            )

        if not row["home_team"]:
            row_reasons.append(
                "missing_home_team"
            )

        if not row["away_team"]:
            row_reasons.append(
                "missing_away_team"
            )

        if (
            row["home_team"]
            and row["away_team"]
            and row["home_team"]
            == row["away_team"]
        ):
            row_reasons.append(
                "same_home_and_away_team"
            )

        if pd.isna(row["home_goals"]):
            row_reasons.append(
                "invalid_home_goals"
            )

        elif row["home_goals"] < 0:
            row_reasons.append(
                "negative_home_goals"
            )

        if pd.isna(row["away_goals"]):
            row_reasons.append(
                "invalid_away_goals"
            )

        elif row["away_goals"] < 0:
            row_reasons.append(
                "negative_away_goals"
            )

        if not row["competition"]:
            row_reasons.append(
                "missing_competition"
            )

        if not row["league_name"]:
            row_reasons.append(
                "missing_league_name"
            )

        if (
            row["winner"]
            not in {"H", "D", "A"}
        ):
            row_reasons.append(
                "invalid_winner_label"
            )

        if (
            pd.notna(row["home_goals"])
            and pd.notna(row["away_goals"])
            and row["winner"]
            in {"H", "D", "A"}
        ):
            expected_winner = (
                determine_expected_winner(
                    row["home_goals"],
                    row["away_goals"],
                )
            )

            if row["winner"] != expected_winner:
                row_reasons.append(
                    "winner_score_mismatch"
                )

        if row_reasons:
            invalid_reasons.append(
                {
                    "row_index": index,
                    "invalid_reasons": (
                        ", ".join(row_reasons)
                    ),
                }
            )

    if not invalid_reasons:
        return pd.DataFrame()

    reasons_dataframe = pd.DataFrame(
        invalid_reasons
    )

    invalid_rows = dataframe.loc[
        reasons_dataframe["row_index"]
    ].copy()

    invalid_rows.insert(
        0,
        "row_index",
        reasons_dataframe[
            "row_index"
        ].to_numpy(),
    )

    invalid_rows.insert(
        1,
        "invalid_reasons",
        reasons_dataframe[
            "invalid_reasons"
        ].to_numpy(),
    )

    return invalid_rows


def create_duplicate_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    duplicate_columns = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    duplicate_mask = dataframe.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    duplicates = dataframe[
        duplicate_mask
    ].copy()

    if duplicates.empty:
        return duplicates

    return duplicates.sort_values(
        by=[
            "date",
            "home_team",
            "away_team",
        ]
    )


def calculate_team_coverage(
    dataframe: pd.DataFrame,
) -> dict:
    home_teams = set(
        dataframe["home_team"]
    )

    away_teams = set(
        dataframe["away_team"]
    )

    all_teams = sorted(
        (
            home_teams
            | away_teams
        )
        - {""}
    )

    team_match_counts = pd.concat(
        [
            dataframe["home_team"],
            dataframe["away_team"],
        ],
        ignore_index=True,
    ).value_counts()

    low_sample_teams = (
        team_match_counts[
            team_match_counts < 10
        ]
        .sort_values()
        .to_dict()
    )

    return {
        "unique_team_count": (
            len(all_teams)
        ),
        "teams_with_fewer_than_10_matches": (
            low_sample_teams
        ),
    }


def calculate_summary(
    dataframe: pd.DataFrame,
    invalid_rows: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> dict:
    valid_dates = dataframe[
        "date"
    ].dropna()

    competition_counts = (
        dataframe[
            "league_name"
        ]
        .value_counts()
        .sort_values(
            ascending=False
        )
        .to_dict()
    )

    season_counts = (
        dataframe[
            "season"
        ]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    source_counts = (
        dataframe[
            "data_source"
        ]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    winner_distribution = (
        dataframe[
            "winner"
        ]
        .value_counts(
            normalize=True
        )
        .to_dict()
    )

    return {
        "dataset_path": str(
            DATA_PATH
        ),
        "total_rows": int(
            len(dataframe)
        ),
        "invalid_rows": int(
            len(invalid_rows)
        ),
        "duplicate_rows": int(
            len(duplicates)
        ),
        "date_min": (
            valid_dates.min().date().isoformat()
            if not valid_dates.empty
            else None
        ),
        "date_max": (
            valid_dates.max().date().isoformat()
            if not valid_dates.empty
            else None
        ),
        "competition_count": int(
            dataframe[
                "competition"
            ].nunique()
        ),
        "league_name_count": int(
            dataframe[
                "league_name"
            ].nunique()
        ),
        "competition_matches": (
            competition_counts
        ),
        "season_matches": (
            season_counts
        ),
        "source_matches": (
            source_counts
        ),
        "winner_distribution": (
            winner_distribution
        ),
        "team_coverage": (
            calculate_team_coverage(
                dataframe
            )
        ),
    }


def print_summary(
    report: dict,
) -> None:
    print()
    print("=" * 86)
    print("EXPANDED DATASET VALIDATION")
    print("=" * 86)

    print(
        f"Rows: {report['total_rows']:,}"
    )
    print(
        f"Invalid rows: "
        f"{report['invalid_rows']:,}"
    )
    print(
        f"Duplicate rows: "
        f"{report['duplicate_rows']:,}"
    )
    print(
        f"Date range: "
        f"{report['date_min']} "
        f"to {report['date_max']}"
    )
    print(
        f"Competitions: "
        f"{report['competition_count']:,}"
    )
    print(
        f"Unique teams: "
        f"{report['team_coverage']['unique_team_count']:,}"
    )

    print()
    print("WINNER DISTRIBUTION")
    print("-" * 86)

    for label, probability in (
        report[
            "winner_distribution"
        ].items()
    ):
        print(
            f"{label}: {probability:.2%}"
        )

    print()
    print("MATCHES BY LEAGUE")
    print("-" * 86)

    for league_name, match_count in (
        report[
            "competition_matches"
        ].items()
    ):
        print(
            f"{league_name:<36}"
            f"{match_count:>8,}"
        )

    low_sample_teams = (
        report[
            "team_coverage"
        ][
            "teams_with_fewer_than_10_matches"
        ]
    )

    print()
    print(
        "TEAMS WITH FEWER THAN 10 MATCHES"
    )
    print("-" * 86)
    print(
        f"Count: "
        f"{len(low_sample_teams):,}"
    )

    print()
    print("FILES")
    print("-" * 86)
    print(
        f"Validation report: {REPORT_PATH}"
    )
    print(
        f"Duplicate report: {DUPLICATE_REPORT_PATH}"
    )
    print(
        f"Invalid-row report: {INVALID_REPORT_PATH}"
    )
    print("=" * 86)


def main() -> None:
    dataframe = load_dataset()

    validated = normalize_for_validation(
        dataframe
    )

    invalid_rows = (
        create_invalid_row_report(
            validated
        )
    )

    duplicates = (
        create_duplicate_report(
            validated
        )
    )

    INVALID_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if invalid_rows.empty:
        pd.DataFrame(
            columns=[
                "row_index",
                "invalid_reasons",
            ]
        ).to_csv(
            INVALID_REPORT_PATH,
            index=False,
        )

    else:
        invalid_rows.to_csv(
            INVALID_REPORT_PATH,
            index=False,
        )

    duplicates.to_csv(
        DUPLICATE_REPORT_PATH,
        index=False,
    )

    report = calculate_summary(
        dataframe=validated,
        invalid_rows=invalid_rows,
        duplicates=duplicates,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(
        report
    )


if __name__ == "__main__":
    main()
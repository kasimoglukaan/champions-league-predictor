from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_matches_expanded_candidate.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_matches_expanded_clean.csv"
)

REMOVED_DUPLICATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded_dataset_removed_duplicates.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded_dataset_cleaning_report.json"
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


SOURCE_PRIORITY = {
    "football-data.co.uk": 2,
    "existing_project_data": 1,
}


def load_dataset() -> pd.DataFrame:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            "Expanded candidate dataset was not found: "
            f"{SOURCE_PATH}"
        )

    dataframe = pd.read_csv(
        SOURCE_PATH,
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


def normalize_dataset(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    cleaned = dataframe.copy()

    cleaned["date"] = pd.to_datetime(
        cleaned["date"],
        errors="coerce",
    )

    cleaned["home_team"] = (
        cleaned["home_team"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned["away_team"] = (
        cleaned["away_team"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned["competition"] = (
        cleaned["competition"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned["season"] = (
        cleaned["season"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned["stage"] = (
        cleaned["stage"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned["league_name"] = (
        cleaned["league_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned["data_source"] = (
        cleaned["data_source"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned["home_goals"] = pd.to_numeric(
        cleaned["home_goals"],
        errors="coerce",
    )

    cleaned["away_goals"] = pd.to_numeric(
        cleaned["away_goals"],
        errors="coerce",
    )

    cleaned["winner"] = (
        cleaned["winner"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    cleaned = cleaned.dropna(
        subset=[
            "date",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    cleaned = cleaned[
        (cleaned["home_team"] != "")
        & (cleaned["away_team"] != "")
        & (
            cleaned["home_team"]
            != cleaned["away_team"]
        )
    ].copy()

    cleaned["home_goals"] = (
        cleaned["home_goals"]
        .astype(int)
    )

    cleaned["away_goals"] = (
        cleaned["away_goals"]
        .astype(int)
    )

    cleaned["source_priority"] = (
        cleaned["data_source"]
        .map(SOURCE_PRIORITY)
        .fillna(0)
        .astype(int)
    )

    return cleaned


def create_match_fingerprint(
    row: pd.Series,
) -> str:
    identity = "|".join(
        [
            row["date"].strftime(
                "%Y-%m-%d"
            ),
            str(row["home_team"]),
            str(row["away_team"]),
            str(row["home_goals"]),
            str(row["away_goals"]),
        ]
    )

    return hashlib.sha1(
        identity.encode("utf-8")
    ).hexdigest()


def remove_duplicates(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deduplication_columns = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    ordered = dataframe.sort_values(
        by=[
            "source_priority",
            "competition",
            "season",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).copy()

    duplicate_mask = ordered.duplicated(
        subset=deduplication_columns,
        keep="first",
    )

    removed_duplicates = ordered[
        duplicate_mask
    ].copy()

    cleaned = ordered[
        ~duplicate_mask
    ].copy()

    return (
        cleaned,
        removed_duplicates,
    )


def rebuild_match_ids(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rebuilt = dataframe.copy()

    rebuilt["match_id"] = [
        (
            "expanded_"
            + create_match_fingerprint(row)[:16]
        )
        for _, row in rebuilt.iterrows()
    ]

    return rebuilt


def validate_cleaned_dataset(
    dataframe: pd.DataFrame,
) -> None:
    duplicate_mask = dataframe.duplicated(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(
            duplicate_mask.sum()
        )

        raise ValueError(
            "Duplicate rows remain after cleaning: "
            f"{duplicate_count}"
        )

    invalid_winner_count = int(
        (
            ~dataframe["winner"]
            .isin({"H", "D", "A"})
        ).sum()
    )

    if invalid_winner_count:
        raise ValueError(
            "Invalid winner labels remain: "
            f"{invalid_winner_count}"
        )


def create_report(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    removed_duplicates: pd.DataFrame,
) -> dict:
    valid_dates = cleaned[
        "date"
    ].dropna()

    unique_teams = (
        set(
            cleaned["home_team"]
        )
        | set(
            cleaned["away_team"]
        )
    )

    return {
        "source_path": str(
            SOURCE_PATH
        ),
        "output_path": str(
            OUTPUT_PATH
        ),
        "original_rows": int(
            len(original)
        ),
        "cleaned_rows": int(
            len(cleaned)
        ),
        "removed_duplicate_rows": int(
            len(removed_duplicates)
        ),
        "date_min": (
            valid_dates.min()
            .date()
            .isoformat()
            if not valid_dates.empty
            else None
        ),
        "date_max": (
            valid_dates.max()
            .date()
            .isoformat()
            if not valid_dates.empty
            else None
        ),
        "unique_teams": int(
            len(unique_teams)
        ),
        "competitions": int(
            cleaned[
                "competition"
            ].nunique()
        ),
        "league_names": int(
            cleaned[
                "league_name"
            ].nunique()
        ),
        "matches_by_league": (
            cleaned[
                "league_name"
            ]
            .value_counts()
            .to_dict()
        ),
        "matches_by_source": (
            cleaned[
                "data_source"
            ]
            .value_counts()
            .to_dict()
        ),
    }


def print_report(
    report: dict,
) -> None:
    print()
    print("=" * 84)
    print("EXPANDED DATASET CLEANING")
    print("=" * 84)

    print(
        f"Original rows: "
        f"{report['original_rows']:,}"
    )

    print(
        f"Removed duplicates: "
        f"{report['removed_duplicate_rows']:,}"
    )

    print(
        f"Cleaned rows: "
        f"{report['cleaned_rows']:,}"
    )

    print(
        f"Date range: "
        f"{report['date_min']} "
        f"to {report['date_max']}"
    )

    print(
        f"Unique teams: "
        f"{report['unique_teams']:,}"
    )

    print(
        f"Competitions: "
        f"{report['competitions']:,}"
    )

    print()
    print("MATCHES BY SOURCE")
    print("-" * 84)

    for (
        source_name,
        match_count,
    ) in report[
        "matches_by_source"
    ].items():
        print(
            f"{source_name:<35}"
            f"{match_count:>8,}"
        )

    print()
    print("FILES")
    print("-" * 84)

    print(
        f"Clean dataset: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Removed duplicates: "
        f"{REMOVED_DUPLICATES_PATH}"
    )

    print(
        f"Cleaning report: "
        f"{REPORT_PATH}"
    )

    print("=" * 84)


def main() -> None:
    original = load_dataset()

    normalized = normalize_dataset(
        original
    )

    (
        cleaned,
        removed_duplicates,
    ) = remove_duplicates(
        normalized
    )

    cleaned = rebuild_match_ids(
        cleaned
    )

    cleaned = cleaned.drop(
        columns=[
            "source_priority",
        ],
        errors="ignore",
    )

    removed_duplicates = (
        removed_duplicates.drop(
            columns=[
                "source_priority",
            ],
            errors="ignore",
        )
    )

    cleaned = cleaned.sort_values(
        by=[
            "date",
            "competition",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    removed_duplicates = (
        removed_duplicates.sort_values(
            by=[
                "date",
                "home_team",
                "away_team",
            ]
        ).reset_index(
            drop=True
        )
    )

    validate_cleaned_dataset(
        cleaned
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_to_save = cleaned.copy()

    cleaned_to_save["date"] = (
        cleaned_to_save["date"]
        .dt.strftime("%Y-%m-%d")
    )

    duplicates_to_save = (
        removed_duplicates.copy()
    )

    if not duplicates_to_save.empty:
        duplicates_to_save["date"] = (
            duplicates_to_save["date"]
            .dt.strftime("%Y-%m-%d")
        )

    cleaned_to_save.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    duplicates_to_save.to_csv(
        REMOVED_DUPLICATES_PATH,
        index=False,
    )

    report = create_report(
        original=original,
        cleaned=cleaned,
        removed_duplicates=(
            removed_duplicates
        ),
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

    print_report(
        report
    )


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATCHES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_matches.csv"
)

COEFFICIENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "uefa_coefficients.csv"
)


def main() -> None:
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"Match data not found: {MATCHES_PATH}"
        )

    if not COEFFICIENTS_PATH.exists():
        raise FileNotFoundError(
            f"Coefficient data not found: {COEFFICIENTS_PATH}"
        )

    matches = pd.read_csv(MATCHES_PATH)
    coefficients = pd.read_csv(COEFFICIENTS_PATH)

    matches["date"] = pd.to_datetime(
        matches["date"],
        utc=True,
        errors="raise",
    )

    matches["season"] = matches["date"].dt.year

    coefficient_keys = set(
        zip(
            coefficients["season"].astype(int),
            coefficients["team"].astype(str),
        )
    )

    home_found = matches.apply(
        lambda row: (
            int(row["season"]),
            str(row["home_team"]),
        )
        in coefficient_keys,
        axis=1,
    )

    away_found = matches.apply(
        lambda row: (
            int(row["season"]),
            str(row["away_team"]),
        )
        in coefficient_keys,
        axis=1,
    )

    both_found = home_found & away_found

    all_teams = sorted(
        set(matches["home_team"].dropna())
        | set(matches["away_team"].dropna())
    )

    coefficient_teams = set(
        coefficients["team"].dropna()
    )

    missing_teams = [
        team
        for team in all_teams
        if team not in coefficient_teams
    ]

    cl_matches = matches[
        matches["competition"] == "CL"
    ].copy()

    cl_home_found = home_found.loc[
        cl_matches.index
    ]

    cl_away_found = away_found.loc[
        cl_matches.index
    ]

    cl_both_found = (
        cl_home_found
        & cl_away_found
    )

    print("=" * 65)
    print("UEFA COEFFICIENT COVERAGE")
    print("=" * 65)

    print(f"Total matches: {len(matches):,}")
    print(f"Total teams: {len(all_teams):,}")
    print(
        "Teams in coefficient file: "
        f"{len(coefficient_teams):,}"
    )

    print()
    print("ALL MATCHES")
    print("-" * 65)

    print(
        "Home coefficient coverage: "
        f"{home_found.mean():.2%}"
    )

    print(
        "Away coefficient coverage: "
        f"{away_found.mean():.2%}"
    )

    print(
        "Both teams covered: "
        f"{both_found.mean():.2%}"
    )

    print()
    print("CHAMPIONS LEAGUE MATCHES")
    print("-" * 65)

    print(f"CL matches: {len(cl_matches):,}")

    print(
        "Home coefficient coverage: "
        f"{cl_home_found.mean():.2%}"
    )

    print(
        "Away coefficient coverage: "
        f"{cl_away_found.mean():.2%}"
    )

    print(
        "Both teams covered: "
        f"{cl_both_found.mean():.2%}"
    )

    print()
    print(
        f"Missing unique teams: "
        f"{len(missing_teams):,}"
    )

    print()
    print("First 50 missing teams:")
    print("-" * 65)

    for team in missing_teams[:50]:
        print(team)

    print("=" * 65)


if __name__ == "__main__":
    main()
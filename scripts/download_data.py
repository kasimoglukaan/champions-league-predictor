import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.clients.football_data_client import (
    FootballDataClient,
)
from src.data.api_match_converter import (
    ApiMatchConverter,
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_matches.csv"
)


COMPETITIONS: Dict[str, List[int]] = {
    "CL": list(range(2018, 2026)),
    "PL": list(range(2018, 2026)),
    "PD": list(range(2018, 2026)),
    "BL1": list(range(2018, 2026)),
    "SA": list(range(2018, 2026)),
    "FL1": list(range(2018, 2026)),
    "PPL": list(range(2018, 2026)),
    "DED": list(range(2018, 2026)),
}


def main() -> None:
    load_dotenv()

    api_key = os.getenv(
        "FOOTBALL_DATA_API_KEY"
    )

    client = FootballDataClient(
        api_key=api_key or "",
        request_delay=6.5,
    )

    converter = ApiMatchConverter()

    dataframes = []

    for competition_code, seasons in (
        COMPETITIONS.items()
    ):
        matches = client.get_multiple_seasons(
            competition_code=competition_code,
            seasons=seasons,
        )

        dataframe = converter.convert(matches)

        if not dataframe.empty:
            dataframes.append(dataframe)

    if not dataframes:
        raise RuntimeError(
            "No match data was downloaded."
        )

    combined = pd.concat(
        dataframes,
        ignore_index=True,
    )

    combined = (
        combined
        .drop_duplicates(subset=["match_id"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print(f"Saved {len(combined)} matches.")
    print(f"File: {OUTPUT_PATH}")
    print(
        f"From: {combined['date'].min()}"
    )
    print(
        f"To:   {combined['date'].max()}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
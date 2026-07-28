import time
from typing import Any, Dict, List, Optional

import requests


class FootballDataClient:
    BASE_URL = "https://api.football-data.org/v4"

    def __init__(
        self,
        api_key: str,
        request_delay: float = 6.5,
        timeout: int = 30,
    ) -> None:
        if not api_key:
            raise ValueError(
                "FOOTBALL_DATA_API_KEY is missing."
            )

        self.api_key = api_key
        self.request_delay = request_delay
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-Auth-Token": self.api_key,
            }
        )

    def get_competition_matches(
        self,
        competition_code: str,
        season: Optional[int] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}

        if season is not None:
            params["season"] = season

        if status is not None:
            params["status"] = status

        endpoint = (
            f"/competitions/{competition_code}/matches"
        )

        response_data = self._get(
            endpoint=endpoint,
            params=params,
        )

        matches = response_data.get("matches", [])

        if not isinstance(matches, list):
            raise ValueError(
                "Unexpected matches response from API."
            )

        return matches

    def get_multiple_seasons(
        self,
        competition_code: str,
        seasons: List[int],
    ) -> List[Dict[str, Any]]:
        all_matches: List[Dict[str, Any]] = []

        for index, season in enumerate(seasons):
            print(
                f"Downloading {competition_code} "
                f"season {season}..."
            )

            try:
                matches = self.get_competition_matches(
                    competition_code=competition_code,
                    season=season,
                    status="FINISHED",
                )

                all_matches.extend(matches)

                print(
                    f"Downloaded {len(matches)} matches."
                )

            except requests.HTTPError as error:
                print(
                    f"Could not download "
                    f"{competition_code} {season}: {error}"
                )

            if index < len(seasons) - 1:
                time.sleep(self.request_delay)

        return all_matches

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{endpoint}"

        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
        )

        if response.status_code == 429:
            raise requests.HTTPError(
                "API rate limit reached. "
                "Wait before trying again."
            )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise ValueError(
                "Unexpected API response format."
            )

        return data
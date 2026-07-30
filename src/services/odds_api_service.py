from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Optional

import requests

from src.models.bookmaker_odds import (
    BookmakerPrice,
    MatchOdds,
    OddsEvent,
)


class OddsAPIError(RuntimeError):
    """Raised when The Odds API request fails."""


class MatchOddsNotFoundError(LookupError):
    """Raised when no suitable odds event can be found."""


class OddsAPIService:
    BASE_URL = (
        "https://api.the-odds-api.com/v4"
    )

    DEFAULT_TIMEOUT_SECONDS = 15

    PREFERRED_LEAGUES = {
        "soccer_epl": (
            "Premier League"
        ),
        "soccer_efl_champ": (
            "EFL Championship"
        ),
        "soccer_germany_bundesliga": (
            "Bundesliga"
        ),
        "soccer_spain_la_liga": (
            "La Liga"
        ),
        "soccer_italy_serie_a": (
            "Serie A"
        ),
        "soccer_france_ligue_one": (
            "Ligue 1"
        ),
        "soccer_netherlands_eredivisie": (
            "Eredivisie"
        ),
        "soccer_portugal_primeira_liga": (
            "Primeira Liga"
        ),
        "soccer_turkey_super_league": (
            "Turkish Super League"
        ),
        "soccer_belgium_first_div": (
            "Belgian First Division"
        ),
        "soccer_scotland_premiership": (
            "Scottish Premiership"
        ),
        "soccer_uefa_champs_league": (
            "UEFA Champions League"
        ),
        "soccer_uefa_champs_league_qualification": (
            "Champions League Qualification"
        ),
        "soccer_uefa_europa_league": (
            "UEFA Europa League"
        ),
        "soccer_uefa_europa_conference_league": (
            "UEFA Conference League"
        ),
    }

    TEAM_ALIASES = {
        "barcelona": {
            "barcelona",
            "fc barcelona",
        },
        "paris saint germain": {
            "paris saint germain",
            "paris saint-germain",
            "psg",
        },
        "inter milan": {
            "inter milan",
            "internazionale",
            "inter",
            "fc internazionale",
        },
        "ac milan": {
            "ac milan",
            "milan",
        },
        "manchester city": {
            "manchester city",
            "man city",
        },
        "manchester united": {
            "manchester united",
            "man united",
            "man utd",
        },
        "bayern munich": {
            "bayern munich",
            "bayern münchen",
            "fc bayern munich",
            "fc bayern münchen",
        },
        "borussia dortmund": {
            "borussia dortmund",
            "dortmund",
        },
        "atletico madrid": {
            "atletico madrid",
            "atlético madrid",
            "club atletico de madrid",
        },
        "real madrid": {
            "real madrid",
            "real madrid cf",
        },
        "sporting cp": {
            "sporting cp",
            "sporting lisbon",
            "sporting clube de portugal",
        },
        "benfica": {
            "benfica",
            "sl benfica",
        },
        "psv eindhoven": {
            "psv eindhoven",
            "psv",
        },
        "ajax": {
            "ajax",
            "afc ajax",
        },
        "red bull salzburg": {
            "red bull salzburg",
            "rb salzburg",
            "fc salzburg",
        },
        "shakhtar donetsk": {
            "shakhtar donetsk",
            "shakhtar",
        },
        "celtic": {
            "celtic",
            "celtic fc",
        },
        "rangers": {
            "rangers",
            "rangers fc",
        },
        "club brugge": {
            "club brugge",
            "club brugge kv",
        },
        "olympique marseille": {
            "olympique marseille",
            "marseille",
        },
        "olympique lyonnais": {
            "olympique lyonnais",
            "lyon",
        },
        "juventus": {
            "juventus",
            "juventus fc",
        },
        "napoli": {
            "napoli",
            "ssc napoli",
        },
        "atalanta": {
            "atalanta",
            "atalanta bc",
        },
        "roma": {
            "roma",
            "as roma",
        },
        "arsenal": {
            "arsenal",
            "arsenal fc",
        },
        "chelsea": {
            "chelsea",
            "chelsea fc",
        },
        "liverpool": {
            "liverpool",
            "liverpool fc",
        },
        "tottenham hotspur": {
            "tottenham hotspur",
            "tottenham",
            "spurs",
        },
        "newcastle united": {
            "newcastle united",
            "newcastle",
        },
        "bayer leverkusen": {
            "bayer leverkusen",
            "bayer 04 leverkusen",
            "leverkusen",
        },
        "rb leipzig": {
            "rb leipzig",
            "rasenballsport leipzig",
        },
        "athletic bilbao": {
            "athletic bilbao",
            "athletic club",
        },
        "real sociedad": {
            "real sociedad",
            "real sociedad san sebastian",
        },
        "porto": {
            "porto",
            "fc porto",
        },
        "feyenoord": {
            "feyenoord",
            "feyenoord rotterdam",
        },
        "galatasaray": {
            "galatasaray",
            "galatasaray sk",
        },
        "fenerbahce": {
            "fenerbahce",
            "fenerbahçe",
            "fenerbahce sk",
            "fenerbahçe sk",
        },
        "besiktas": {
            "besiktas",
            "beşiktaş",
            "besiktas jk",
            "beşiktaş jk",
        },
    }

    def __init__(
        self,
        api_key: str,
        region: str = "eu",
        timeout_seconds: int = (
            DEFAULT_TIMEOUT_SECONDS
        ),
    ) -> None:
        cleaned_api_key = api_key.strip()

        if not cleaned_api_key:
            raise ValueError(
                "The Odds API key cannot be empty."
            )

        self.api_key = cleaned_api_key
        self.region = region
        self.timeout_seconds = timeout_seconds

        self.session = requests.Session()

    def get_active_soccer_leagues(
        self,
    ) -> dict[str, str]:
        url = (
            f"{self.BASE_URL}/sports/"
        )

        response = self._get(
            url=url,
            params={
                "apiKey": self.api_key,
            },
        )

        payload = self._parse_json(
            response
        )

        if not isinstance(payload, list):
            raise OddsAPIError(
                "Unexpected sports response "
                "received from The Odds API."
            )

        active_leagues = {}

        for sport_data in payload:
            if not isinstance(
                sport_data,
                dict,
            ):
                continue

            sport_key = str(
                sport_data.get(
                    "key",
                    "",
                )
            ).strip()

            group = str(
                sport_data.get(
                    "group",
                    "",
                )
            ).strip()

            active = bool(
                sport_data.get(
                    "active",
                    False,
                )
            )

            if not active:
                continue

            if not sport_key.startswith(
                "soccer_"
            ):
                continue

            if (
                group
                and group.casefold()
                != "soccer"
            ):
                continue

            title = str(
                sport_data.get(
                    "title",
                    sport_key,
                )
            ).strip()

            display_name = (
                self.PREFERRED_LEAGUES.get(
                    sport_key,
                    title,
                )
            )

            active_leagues[
                sport_key
            ] = display_name

        return dict(
            sorted(
                active_leagues.items(),
                key=lambda item: (
                    self._league_sort_key(
                        item[0],
                        item[1],
                    )
                ),
            )
        )

    def get_league_events(
        self,
        sport_key: str,
    ) -> list[OddsEvent]:
        cleaned_sport_key = (
            sport_key.strip()
        )

        if not cleaned_sport_key:
            raise ValueError(
                "Sport key cannot be empty."
            )

        if not cleaned_sport_key.startswith(
            "soccer_"
        ):
            raise ValueError(
                "Only soccer leagues are "
                "supported by this service."
            )

        url = (
            f"{self.BASE_URL}/sports/"
            f"{cleaned_sport_key}/events"
        )

        response = self._get(
            url=url,
            params={
                "apiKey": self.api_key,
                "dateFormat": "iso",
            },
        )

        payload = self._parse_json(
            response
        )

        if not isinstance(payload, list):
            raise OddsAPIError(
                "Unexpected events response "
                "received from The Odds API."
            )

        events = []

        for event_data in payload:
            if not isinstance(
                event_data,
                dict,
            ):
                continue

            event_id = str(
                event_data.get(
                    "id",
                    "",
                )
            ).strip()

            home_team = str(
                event_data.get(
                    "home_team",
                    "",
                )
            ).strip()

            away_team = str(
                event_data.get(
                    "away_team",
                    "",
                )
            ).strip()

            if not (
                event_id
                and home_team
                and away_team
            ):
                continue

            events.append(
                OddsEvent(
                    event_id=event_id,
                    sport_key=str(
                        event_data.get(
                            "sport_key",
                            cleaned_sport_key,
                        )
                    ),
                    sport_title=str(
                        event_data.get(
                            "sport_title",
                            self.PREFERRED_LEAGUES.get(
                                cleaned_sport_key,
                                cleaned_sport_key,
                            ),
                        )
                    ),
                    commence_time=str(
                        event_data.get(
                            "commence_time",
                            "",
                        )
                    ),
                    home_team=home_team,
                    away_team=away_team,
                )
            )

        return sorted(
            events,
            key=lambda event: (
                event.commence_time
            ),
        )

    def find_event(
        self,
        sport_key: str,
        home_team: str,
        away_team: str,
    ) -> Optional[OddsEvent]:
        events = self.get_league_events(
            sport_key=sport_key,
        )

        best_event = None
        best_score = 0.0

        for event in events:
            direct_score = (
                self._team_similarity(
                    home_team,
                    event.home_team,
                )
                + self._team_similarity(
                    away_team,
                    event.away_team,
                )
            ) / 2.0

            reversed_score = (
                self._team_similarity(
                    home_team,
                    event.away_team,
                )
                + self._team_similarity(
                    away_team,
                    event.home_team,
                )
            ) / 2.0

            event_score = max(
                direct_score,
                reversed_score,
            )

            if event_score > best_score:
                best_score = event_score
                best_event = event

        if best_score < 0.72:
            return None

        return best_event

    def get_match_odds(
        self,
        sport_key: str,
        home_team: str,
        away_team: str,
    ) -> MatchOdds:
        event = self.find_event(
            sport_key=sport_key,
            home_team=home_team,
            away_team=away_team,
        )

        if event is None:
            raise MatchOddsNotFoundError(
                "No active odds event was found "
                f"for {home_team} vs {away_team}."
            )

        return self.get_event_odds(
            event=event,
        )

    def get_event_odds(
        self,
        event: OddsEvent,
    ) -> MatchOdds:
        url = (
            f"{self.BASE_URL}/sports/"
            f"{event.sport_key}"
            f"/events/{event.event_id}/odds"
        )

        response = self._get(
            url=url,
            params={
                "apiKey": self.api_key,
                "regions": self.region,
                "markets": "h2h,totals",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )

        payload = self._parse_json(
            response
        )

        if not isinstance(payload, dict):
            raise OddsAPIError(
                "Unexpected odds response "
                "received from The Odds API."
            )

        prices = self._extract_prices(
            payload
        )

        return MatchOdds(
            event_id=str(
                payload.get(
                    "id",
                    event.event_id,
                )
            ),
            sport_key=str(
                payload.get(
                    "sport_key",
                    event.sport_key,
                )
            ),
            commence_time=str(
                payload.get(
                    "commence_time",
                    event.commence_time,
                )
            ),
            home_team=str(
                payload.get(
                    "home_team",
                    event.home_team,
                )
            ),
            away_team=str(
                payload.get(
                    "away_team",
                    event.away_team,
                )
            ),
            prices=prices,
            requests_remaining=(
                self._header_as_integer(
                    response,
                    "x-requests-remaining",
                )
            ),
            requests_used=(
                self._header_as_integer(
                    response,
                    "x-requests-used",
                )
            ),
            request_cost=(
                self._header_as_integer(
                    response,
                    "x-requests-last",
                )
            ),
        )

    def _extract_prices(
        self,
        payload: dict[str, Any],
    ) -> list[BookmakerPrice]:
        prices = []

        bookmakers = payload.get(
            "bookmakers",
            [],
        )

        if not isinstance(
            bookmakers,
            list,
        ):
            return prices

        for bookmaker in bookmakers:
            if not isinstance(
                bookmaker,
                dict,
            ):
                continue

            bookmaker_key = str(
                bookmaker.get(
                    "key",
                    "",
                )
            )

            bookmaker_title = str(
                bookmaker.get(
                    "title",
                    bookmaker_key,
                )
            )

            bookmaker_update = (
                bookmaker.get(
                    "last_update"
                )
            )

            markets = bookmaker.get(
                "markets",
                [],
            )

            if not isinstance(
                markets,
                list,
            ):
                continue

            for market in markets:
                if not isinstance(
                    market,
                    dict,
                ):
                    continue

                market_key = str(
                    market.get(
                        "key",
                        "",
                    )
                )

                if not market_key:
                    continue

                market_name = (
                    self._market_name(
                        market_key
                    )
                )

                market_update = (
                    market.get(
                        "last_update",
                        bookmaker_update,
                    )
                )

                outcomes = market.get(
                    "outcomes",
                    [],
                )

                if not isinstance(
                    outcomes,
                    list,
                ):
                    continue

                for outcome in outcomes:
                    if not isinstance(
                        outcome,
                        dict,
                    ):
                        continue

                    selection = str(
                        outcome.get(
                            "name",
                            "",
                        )
                    ).strip()

                    raw_odds = outcome.get(
                        "price"
                    )

                    if (
                        not selection
                        or raw_odds is None
                    ):
                        continue

                    try:
                        odds = float(
                            raw_odds
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    if odds <= 1.0:
                        continue

                    raw_point = outcome.get(
                        "point"
                    )

                    point = None

                    if raw_point is not None:
                        try:
                            point = float(
                                raw_point
                            )

                        except (
                            TypeError,
                            ValueError,
                        ):
                            point = None

                    prices.append(
                        BookmakerPrice(
                            bookmaker_key=(
                                bookmaker_key
                            ),
                            bookmaker_title=(
                                bookmaker_title
                            ),
                            market_key=(
                                market_key
                            ),
                            market_name=(
                                market_name
                            ),
                            selection=(
                                selection
                            ),
                            odds=odds,
                            point=point,
                            last_update=(
                                str(
                                    market_update
                                )
                                if market_update
                                else None
                            ),
                        )
                    )

        return prices

    def _get(
        self,
        url: str,
        params: dict[str, Any],
    ) -> requests.Response:
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=(
                    self.timeout_seconds
                ),
            )

        except requests.Timeout as error:
            raise OddsAPIError(
                "The Odds API request timed out."
            ) from error

        except requests.RequestException as error:
            raise OddsAPIError(
                "The Odds API could not be reached."
            ) from error

        if response.ok:
            return response

        error_message = (
            self._extract_error_message(
                response
            )
        )

        if response.status_code == 401:
            raise OddsAPIError(
                "The Odds API rejected the API "
                "key. Check ODDS_API_KEY in "
                ".streamlit/secrets.toml."
            )

        if response.status_code == 429:
            raise OddsAPIError(
                "The Odds API quota has been "
                "exhausted or too many requests "
                "were sent."
            )

        raise OddsAPIError(
            "The Odds API request failed "
            f"({response.status_code}): "
            f"{error_message}"
        )

    @staticmethod
    def _parse_json(
        response: requests.Response,
    ) -> Any:
        try:
            return response.json()

        except ValueError as error:
            raise OddsAPIError(
                "The Odds API returned an "
                "invalid JSON response."
            ) from error

    @staticmethod
    def _extract_error_message(
        response: requests.Response,
    ) -> str:
        try:
            payload = response.json()

        except ValueError:
            return (
                response.text.strip()
                or "Unknown API error"
            )

        if isinstance(payload, dict):
            return str(
                payload.get(
                    "message",
                    payload.get(
                        "error",
                        "Unknown API error",
                    ),
                )
            )

        return "Unknown API error"

    @staticmethod
    def _header_as_integer(
        response: requests.Response,
        header_name: str,
    ) -> Optional[int]:
        raw_value = response.headers.get(
            header_name
        )

        if raw_value is None:
            return None

        try:
            return int(raw_value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def _team_similarity(
        cls,
        first_team: str,
        second_team: str,
    ) -> float:
        first_forms = (
            cls._possible_team_names(
                first_team
            )
        )

        second_forms = (
            cls._possible_team_names(
                second_team
            )
        )

        best_score = 0.0

        for first_form in first_forms:
            for second_form in second_forms:
                if first_form == second_form:
                    return 1.0

                score = SequenceMatcher(
                    None,
                    first_form,
                    second_form,
                ).ratio()

                best_score = max(
                    best_score,
                    score,
                )

        return best_score

    @classmethod
    def _possible_team_names(
        cls,
        team_name: str,
    ) -> set[str]:
        normalized_name = (
            cls._normalize_team_name(
                team_name
            )
        )

        forms = {
            normalized_name,
        }

        for canonical_name, aliases in (
            cls.TEAM_ALIASES.items()
        ):
            normalized_aliases = {
                cls._normalize_team_name(
                    alias
                )
                for alias in aliases
            }

            if (
                normalized_name
                == canonical_name
                or normalized_name
                in normalized_aliases
            ):
                forms.add(
                    canonical_name
                )

                forms.update(
                    normalized_aliases
                )

        return forms

    @staticmethod
    def _normalize_team_name(
        team_name: str,
    ) -> str:
        normalized = (
            team_name
            .strip()
            .casefold()
        )

        replacements = {
            "á": "a",
            "à": "a",
            "â": "a",
            "ä": "a",
            "ã": "a",
            "å": "a",
            "ç": "c",
            "é": "e",
            "è": "e",
            "ê": "e",
            "ë": "e",
            "í": "i",
            "ì": "i",
            "î": "i",
            "ï": "i",
            "ñ": "n",
            "ó": "o",
            "ò": "o",
            "ô": "o",
            "ö": "o",
            "õ": "o",
            "ú": "u",
            "ù": "u",
            "û": "u",
            "ü": "u",
            "ğ": "g",
            "ı": "i",
            "ş": "s",
        }

        for source, target in (
            replacements.items()
        ):
            normalized = normalized.replace(
                source,
                target,
            )

        punctuation = (
            ".",
            ",",
            "-",
            "_",
            "/",
            "\\",
            "'",
            '"',
            "(",
            ")",
        )

        for character in punctuation:
            normalized = normalized.replace(
                character,
                " ",
            )

        ignored_words = {
            "fc",
            "cf",
            "afc",
            "sc",
            "sk",
            "fk",
            "club",
            "football",
            "calcio",
        }

        words = [
            word
            for word in normalized.split()
            if word not in ignored_words
        ]

        return " ".join(
            words
        ).strip()

    @classmethod
    def _league_sort_key(
        cls,
        sport_key: str,
        title: str,
    ) -> tuple[int, str]:
        preferred_keys = list(
            cls.PREFERRED_LEAGUES.keys()
        )

        if sport_key in preferred_keys:
            return (
                preferred_keys.index(
                    sport_key
                ),
                title.casefold(),
            )

        return (
            len(preferred_keys),
            title.casefold(),
        )

    @staticmethod
    def _market_name(
        market_key: str,
    ) -> str:
        names = {
            "h2h": "Match Result",
            "totals": "Total Goals",
            "btts": (
                "Both Teams To Score"
            ),
        }

        return names.get(
            market_key,
            market_key,
        )
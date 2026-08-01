from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

from rapidfuzz import fuzz


@dataclass(frozen=True)
class TeamCandidate:
    team_name: str
    score: float
    distinctive_overlap: tuple[str, ...]


@dataclass(frozen=True)
class TeamMatchResult:
    api_team_name: str
    matched_team_name: Optional[str]

    score: float
    second_best_score: float
    score_margin: float

    accepted: bool
    reason: str

    candidates: tuple[TeamCandidate, ...]

    @property
    def confidence_percent(self) -> float:
        return self.score * 100.0


@dataclass(frozen=True)
class NormalizedTeamName:
    original: str
    normalized: str
    canonical: str

    all_tokens: frozenset[str]
    distinctive_tokens: frozenset[str]


class TeamMatchingService:
    """
    Matches bookmaker/API team names to model team names.

    The matcher combines:

    - Unicode and punctuation normalization
    - Club-prefix and club-suffix removal
    - Verified aliases
    - Exact canonical-name matching
    - Distinctive-token overlap
    - RapidFuzz similarity metrics
    - Ambiguity protection
    - Generic-word protection

    The service rejects uncertain matches instead of forcing a
    potentially incorrect model team.
    """

    GENERIC_WORDS = {
        "1",
        "2",
        "11",
        "1900",
        "1901",
        "1902",
        "1903",
        "1904",
        "1905",
        "1906",
        "1907",
        "1908",
        "1909",
        "1910",
        "1911",
        "1912",
        "1913",
        "1914",
        "1915",
        "1916",
        "1917",
        "1918",
        "1919",
        "1920",
        "1921",
        "1922",
        "1923",
        "1924",
        "1925",
        "1926",
        "1927",
        "1928",
        "1929",
        "1930",
        "1931",
        "1932",
        "1933",
        "1934",
        "1935",
        "1936",
        "1937",
        "1938",
        "1939",
        "1940",
        "1941",
        "1942",
        "1943",
        "1944",
        "1945",
        "1946",
        "1947",
        "1948",
        "1949",
        "1950",
        "1951",
        "1952",
        "1953",
        "1954",
        "1955",
        "1956",
        "1957",
        "1958",
        "1959",
        "1960",
        "1961",
        "1962",
        "1963",
        "1964",
        "1965",
        "1966",
        "1967",
        "1968",
        "1969",
        "1970",
        "1971",
        "1972",
        "1973",
        "1974",
        "1975",
        "1976",
        "1977",
        "1978",
        "1979",
        "1980",
        "1981",
        "1982",
        "1983",
        "1984",
        "1985",
        "1986",
        "1987",
        "1988",
        "1989",
        "1990",
        "1991",
        "1992",
        "1993",
        "1994",
        "1995",
        "1996",
        "1997",
        "1998",
        "1999",
        "2000",
        "afc",
        "association",
        "as",
        "athletic",
        "athletico",
        "atletico",
        "bc",
        "bk",
        "calcio",
        "cd",
        "cf",
        "city",
        "club",
        "de",
        "des",
        "du",
        "fc",
        "fk",
        "football",
        "fsv",
        "if",
        "jk",
        "kv",
        "la",
        "le",
        "los",
        "olympic",
        "olympique",
        "real",
        "rb",
        "rc",
        "rcc",
        "rsc",
        "sc",
        "sk",
        "sl",
        "sport",
        "sporting",
        "ssc",
        "sv",
        "town",
        "ud",
        "united",
        "vfb",
        "vfl",
    }

    REMOVABLE_TOKENS = {
        "afc",
        "association",
        "bc",
        "bk",
        "calcio",
        "cd",
        "cf",
        "club",
        "fc",
        "fk",
        "football",
        "fsv",
        "jk",
        "kv",
        "rc",
        "rcc",
        "rsc",
        "sc",
        "sk",
        "sl",
        "ssc",
        "sv",
        "ud",
        "vfb",
        "vfl",
    }

    RESERVE_MARKERS = {
        "ii",
        "iii",
        "b",
        "res",
        "reserve",
        "reserves",
        "u18",
        "u19",
        "u20",
        "u21",
        "u23",
        "women",
        "womens",
        "woman",
        "ladies",
        "femenino",
        "feminine",
        "feminin",
    }

    COUNTRY_MARKERS = {
        "bel",
        "belgium",
        "den",
        "denmark",
        "eng",
        "england",
        "esp",
        "spain",
        "fra",
        "france",
        "ger",
        "germany",
        "ita",
        "italy",
        "ned",
        "netherlands",
        "nld",
        "por",
        "portugal",
        "sco",
        "scotland",
        "tur",
        "turkey",
    }

    DANGEROUS_GENERIC_ONLY_NAMES = {
        "athletic",
        "atletico",
        "city",
        "inter",
        "olympic",
        "olympique",
        "real",
        "sport",
        "sporting",
        "town",
        "united",
    }

    VERIFIED_ALIAS_GROUPS = (
        {
            "ac milan",
            "milan",
        },
        {
            "ajax",
            "afc ajax",
        },
        {
            "arsenal",
            "arsenal fc",
        },
        {
            "aston villa",
            "aston villa fc",
        },
        {
            "athletic bilbao",
            "ath bilbao",
        },
        {
            "atletico madrid",
            "athletico madrid",
            "club atletico de madrid",
        },
        {
            "barcelona",
            "fc barcelona",
        },
        {
            "bayer 04 leverkusen",
            "bayer leverkusen",
            "leverkusen",
        },
        {
            "bayern munich",
            "bayern munchen",
            "fc bayern munich",
            "fc bayern munchen",
        },
        {
            "benfica",
            "sl benfica",
        },
        {
            "besiktas",
            "besiktas jk",
        },
        {
            "borussia dortmund",
            "dortmund",
        },
        {
            "brighton",
            "brighton and hove albion",
            "brighton hove albion",
        },
        {
            "celtic",
            "celtic fc",
        },
        {
            "club brugge",
            "club brugge kv",
        },
        {
            "coventry",
            "coventry city",
            "coventry city fc",
        },
        {
            "crystal palace",
            "crystal palace fc",
        },
        {
            "fenerbahce",
            "fenerbahce sk",
        },
        {
            "galatasaray",
            "galatasaray sk",
        },
        {
            "inter",
            "inter milan",
            "internazionale",
            "fc internazionale",
            "fc internazionale milano",
        },
        {
            "istanbul basaksehir",
            "basaksehir",
            "istanbul basaksehir fk",
        },
        {
            "leicester",
            "leicester city",
            "leicester city fc",
        },
        {
            "liverpool",
            "liverpool fc",
        },
        {
            "manchester city",
            "man city",
            "man city fc",
        },
        {
            "manchester united",
            "man united",
            "man utd",
            "manchester utd",
        },
        {
            "newcastle",
            "newcastle united",
            "newcastle united fc",
        },
        {
            "nottingham forest",
            "nottm forest",
            "nott m forest",
            "nott forest",
        },
        {
            "olympique lyonnais",
            "lyon",
        },
        {
            "olympique marseille",
            "marseille",
        },
        {
            "paris saint germain",
            "paris saint-germain",
            "paris sg",
            "psg",
        },
        {
            "porto",
            "fc porto",
        },
        {
            "psv",
            "psv eindhoven",
        },
        {
            "rb leipzig",
            "rasenballsport leipzig",
        },
        {
            "union saint gilloise",
            "union saint-gilloise",
            "royale union saint gilloise",
            "royale union saint-gilloise",
            "r union saint gilloise",
            "r union saint-gilloise",
            "union sg",
            "usg",
        },
        {
            "real madrid",
            "real madrid cf",
        },
        {
            "real sociedad",
            "real sociedad san sebastian",
        },
        {
            "red bull salzburg",
            "rb salzburg",
            "salzburg",
            "fc salzburg",
        },
        {
            "shakhtar",
            "shakhtar donetsk",
            "fk shakhtar donetsk",
        },
        {
            "sporting braga",
            "sc braga",
            "braga",
        },
        {
            "sporting cp",
            "sporting lisbon",
            "sporting clube de portugal",
        },
        {
            "tottenham",
            "tottenham hotspur",
            "spurs",
        },
        {
            "trabzonspor",
            "trabzonspor sk",
        },
        {
            "west brom",
            "west bromwich albion",
        },
        {
            "west ham",
            "west ham united",
        },
        {
            "wolverhampton",
            "wolverhampton wanderers",
            "wolves",
        },
    )

    def __init__(
        self,
        minimum_score: float = 0.84,
        minimum_margin: float = 0.12,
        candidate_limit: int = 5,
    ) -> None:
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between 0 and 1."
            )

        if not 0.0 <= minimum_margin <= 1.0:
            raise ValueError(
                "minimum_margin must be between 0 and 1."
            )

        if candidate_limit < 1:
            raise ValueError(
                "candidate_limit must be at least 1."
            )

        self.minimum_score = minimum_score
        self.minimum_margin = minimum_margin
        self.candidate_limit = candidate_limit

        self.alias_lookup = (
            self._build_alias_lookup()
        )

    def match(
        self,
        api_team_name: str,
        model_teams: Iterable[str],
    ) -> TeamMatchResult:
        cleaned_api_name = str(
            api_team_name
        ).strip()

        if not cleaned_api_name:
            return self._rejected_result(
                api_team_name=(
                    str(api_team_name)
                ),
                reason=(
                    "API team name is empty."
                ),
            )

        model_team_list = sorted(
            {
                str(team).strip()
                for team in model_teams
                if str(team).strip()
            }
        )

        if not model_team_list:
            return self._rejected_result(
                api_team_name=(
                    cleaned_api_name
                ),
                reason=(
                    "The model team list is empty."
                ),
            )

        api_identity = self._prepare_name(
            cleaned_api_name
        )

        prepared_model_teams = [
            (
                model_team,
                self._prepare_name(
                    model_team
                ),
            )
            for model_team
            in model_team_list
        ]

        exact_candidate = (
            self._find_exact_candidate(
                api_identity=api_identity,
                prepared_model_teams=(
                    prepared_model_teams
                ),
            )
        )

        if exact_candidate is not None:
            return TeamMatchResult(
                api_team_name=(
                    cleaned_api_name
                ),
                matched_team_name=(
                    exact_candidate.team_name
                ),
                score=(
                    exact_candidate.score
                ),
                second_best_score=0.0,
                score_margin=(
                    exact_candidate.score
                ),
                accepted=True,
                reason=(
                    "Exact canonical team-name match."
                ),
                candidates=(
                    exact_candidate,
                ),
            )

        alias_candidate = (
            self._find_alias_candidate(
                api_identity=api_identity,
                prepared_model_teams=(
                    prepared_model_teams
                ),
            )
        )

        if alias_candidate is not None:
            other_candidates = [
                self._score_candidate(
                    api_identity=api_identity,
                    model_team_name=(
                        model_team
                    ),
                    model_identity=(
                        model_identity
                    ),
                )
                for (
                    model_team,
                    model_identity,
                )
                in prepared_model_teams
                if model_team
                != alias_candidate.team_name
            ]

            other_candidates.sort(
                key=lambda item: (
                    item.score,
                    item.team_name,
                ),
                reverse=True,
            )

            second_best_score = (
                other_candidates[0].score
                if other_candidates
                else 0.0
            )

            visible_candidates = tuple(
                [
                    alias_candidate,
                    *other_candidates[
                        : max(
                            0,
                            self.candidate_limit
                            - 1,
                        )
                    ],
                ]
            )

            return TeamMatchResult(
                api_team_name=(
                    cleaned_api_name
                ),
                matched_team_name=(
                    alias_candidate.team_name
                ),
                score=(
                    alias_candidate.score
                ),
                second_best_score=(
                    second_best_score
                ),
                score_margin=(
                    alias_candidate.score
                    - second_best_score
                ),
                accepted=True,
                reason=(
                    "Verified team alias match."
                ),
                candidates=(
                    visible_candidates
                ),
            )

        candidates = [
            self._score_candidate(
                api_identity=api_identity,
                model_team_name=model_team,
                model_identity=model_identity,
            )
            for (
                model_team,
                model_identity,
            )
            in prepared_model_teams
        ]

        candidates.sort(
            key=lambda item: (
                item.score,
                item.team_name,
            ),
            reverse=True,
        )

        best_candidate = candidates[0]

        second_best_score = (
            candidates[1].score
            if len(candidates) > 1
            else 0.0
        )

        score_margin = (
            best_candidate.score
            - second_best_score
        )

        accepted, reason = (
            self._validate_candidate(
                api_identity=api_identity,
                best_candidate=(
                    best_candidate
                ),
                second_best_score=(
                    second_best_score
                ),
                score_margin=(
                    score_margin
                ),
            )
        )

        return TeamMatchResult(
            api_team_name=(
                cleaned_api_name
            ),
            matched_team_name=(
                best_candidate.team_name
                if accepted
                else None
            ),
            score=(
                best_candidate.score
            ),
            second_best_score=(
                second_best_score
            ),
            score_margin=(
                score_margin
            ),
            accepted=accepted,
            reason=reason,
            candidates=tuple(
                candidates[
                    : self.candidate_limit
                ]
            ),
        )

    def _find_exact_candidate(
        self,
        api_identity: NormalizedTeamName,
        prepared_model_teams: list[
            tuple[
                str,
                NormalizedTeamName,
            ]
        ],
    ) -> Optional[TeamCandidate]:
        for (
            model_team,
            model_identity,
        ) in prepared_model_teams:
            if (
                api_identity.canonical
                == model_identity.canonical
                and api_identity.canonical
            ):
                overlap = (
                    api_identity
                    .distinctive_tokens
                    & model_identity
                    .distinctive_tokens
                )

                return TeamCandidate(
                    team_name=model_team,
                    score=1.0,
                    distinctive_overlap=tuple(
                        sorted(overlap)
                    ),
                )

        return None

    def _find_alias_candidate(
        self,
        api_identity: NormalizedTeamName,
        prepared_model_teams: list[
            tuple[
                str,
                NormalizedTeamName,
            ]
        ],
    ) -> Optional[TeamCandidate]:
        api_alias = (
            self.alias_lookup.get(
                api_identity.normalized
            )
            or self.alias_lookup.get(
                api_identity.canonical
            )
        )

        if api_alias is None:
            return None

        matches = []

        for (
            model_team,
            model_identity,
        ) in prepared_model_teams:
            model_alias = (
                self.alias_lookup.get(
                    model_identity.normalized
                )
                or self.alias_lookup.get(
                    model_identity.canonical
                )
            )

            if model_alias == api_alias:
                overlap = (
                    api_identity
                    .distinctive_tokens
                    & model_identity
                    .distinctive_tokens
                )

                matches.append(
                    TeamCandidate(
                        team_name=(
                            model_team
                        ),
                        score=0.995,
                        distinctive_overlap=tuple(
                            sorted(overlap)
                        ),
                    )
                )

        if len(matches) != 1:
            return None

        return matches[0]

    def _score_candidate(
        self,
        api_identity: NormalizedTeamName,
        model_team_name: str,
        model_identity: NormalizedTeamName,
    ) -> TeamCandidate:
        if (
            not api_identity.canonical
            or not model_identity.canonical
        ):
            return TeamCandidate(
                team_name=(
                    model_team_name
                ),
                score=0.0,
                distinctive_overlap=(),
            )

        api_distinctive = set(
            api_identity
            .distinctive_tokens
        )

        model_distinctive = set(
            model_identity
            .distinctive_tokens
        )

        distinctive_overlap = (
            api_distinctive
            & model_distinctive
        )

        all_distinctive = (
            api_distinctive
            | model_distinctive
        )

        if not distinctive_overlap:
            return TeamCandidate(
                team_name=(
                    model_team_name
                ),
                score=0.0,
                distinctive_overlap=(),
            )

        distinctive_jaccard = (
            len(distinctive_overlap)
            / len(all_distinctive)
            if all_distinctive
            else 0.0
        )

        api_tokens = set(
            api_identity.all_tokens
        )

        model_tokens = set(
            model_identity.all_tokens
        )

        all_token_union = (
            api_tokens
            | model_tokens
        )

        token_jaccard = (
            len(
                api_tokens
                & model_tokens
            )
            / len(all_token_union)
            if all_token_union
            else 0.0
        )

        canonical_ratio = (
            fuzz.ratio(
                api_identity.canonical,
                model_identity.canonical,
            )
            / 100.0
        )

        token_set_ratio = (
            fuzz.token_set_ratio(
                api_identity.canonical,
                model_identity.canonical,
            )
            / 100.0
        )

        weighted_ratio = (
            fuzz.WRatio(
                api_identity.normalized,
                model_identity.normalized,
            )
            / 100.0
        )

        partial_ratio = (
            fuzz.partial_ratio(
                api_identity.canonical,
                model_identity.canonical,
            )
            / 100.0
        )

        containment_bonus = (
            1.0
            if (
                api_identity.canonical
                in model_identity.canonical
                or model_identity.canonical
                in api_identity.canonical
            )
            else 0.0
        )

        score = (
            0.36
            * distinctive_jaccard
            + 0.15
            * token_jaccard
            + 0.16
            * canonical_ratio
            + 0.14
            * token_set_ratio
            + 0.09
            * weighted_ratio
            + 0.05
            * partial_ratio
            + 0.05
            * containment_bonus
        )

        if (
            api_distinctive
            == model_distinctive
            and api_distinctive
        ):
            score = max(
                score,
                0.95,
            )

        if (
            len(api_distinctive) == 1
            and len(model_distinctive) == 1
            and api_distinctive
            == model_distinctive
        ):
            score = max(
                score,
                0.97,
            )

        unmatched_api_tokens = (
            api_distinctive
            - model_distinctive
        )

        unmatched_model_tokens = (
            model_distinctive
            - api_distinctive
        )

        unmatched_count = (
            len(unmatched_api_tokens)
            + len(
                unmatched_model_tokens
            )
        )

        if unmatched_count >= 2:
            score *= 0.80

        elif unmatched_count == 1:
            score *= 0.92

        if (
            len(distinctive_overlap) == 1
            and len(all_distinctive) >= 3
        ):
            score *= 0.72

        if self._has_conflicting_identity(
            api_identity=api_identity,
            model_identity=(
                model_identity
            ),
        ):
            score *= 0.35

        return TeamCandidate(
            team_name=model_team_name,
            score=float(
                min(
                    max(
                        score,
                        0.0,
                    ),
                    1.0,
                )
            ),
            distinctive_overlap=tuple(
                sorted(
                    distinctive_overlap
                )
            ),
        )

    def _validate_candidate(
        self,
        api_identity: NormalizedTeamName,
        best_candidate: TeamCandidate,
        second_best_score: float,
        score_margin: float,
    ) -> tuple[bool, str]:
        if not best_candidate.distinctive_overlap:
            return (
                False,
                (
                    "No distinctive team-name "
                    "token matched. A match based "
                    "only on generic club words "
                    "was rejected."
                ),
            )

        if (
            api_identity.canonical
            in self.DANGEROUS_GENERIC_ONLY_NAMES
        ):
            return (
                False,
                (
                    "The API team name contains "
                    "only a generic club term and "
                    "cannot be matched safely."
                ),
            )

        if (
            best_candidate.score
            < self.minimum_score
        ):
            return (
                False,
                (
                    "The best candidate score is "
                    "below the required threshold "
                    f"of {self.minimum_score:.0%}."
                ),
            )

        if (
            score_margin
            < self.minimum_margin
        ):
            return (
                False,
                (
                    "The best and second-best "
                    "candidates are too close. "
                    "The match is ambiguous and "
                    "was rejected."
                ),
            )

        if (
            second_best_score >= 0.90
            and score_margin < 0.20
        ):
            return (
                False,
                (
                    "Multiple high-confidence "
                    "candidates were found. The "
                    "match was rejected to avoid "
                    "selecting the wrong club."
                ),
            )

        return (
            True,
            (
                "High-confidence distinctive-name "
                "match."
            ),
        )

    def _has_conflicting_identity(
        self,
        api_identity: NormalizedTeamName,
        model_identity: NormalizedTeamName,
    ) -> bool:
        api_distinctive = set(
            api_identity
            .distinctive_tokens
        )

        model_distinctive = set(
            model_identity
            .distinctive_tokens
        )

        shared = (
            api_distinctive
            & model_distinctive
        )

        if not shared:
            return True

        api_unique = (
            api_distinctive
            - model_distinctive
        )

        model_unique = (
            model_distinctive
            - api_distinctive
        )

        if (
            len(shared) == 1
            and api_unique
            and model_unique
        ):
            return True

        return False

    def _prepare_name(
        self,
        team_name: str,
    ) -> NormalizedTeamName:
        original = str(
            team_name
        ).strip()

        normalized = (
            self.normalize_name(
                original
            )
        )

        raw_tokens = [
            token
            for token
            in normalized.split()
            if token
        ]

        cleaned_tokens = [
            token
            for token
            in raw_tokens
            if (
                token
                not in self.COUNTRY_MARKERS
                and token
                not in self.RESERVE_MARKERS
            )
        ]

        canonical_tokens = [
            token
            for token
            in cleaned_tokens
            if token
            not in self.REMOVABLE_TOKENS
        ]

        canonical = " ".join(
            canonical_tokens
        ).strip()

        if not canonical:
            canonical = normalized

        distinctive_tokens = frozenset(
            token
            for token
            in canonical.split()
            if (
                token
                not in self.GENERIC_WORDS
                and len(token) >= 3
            )
        )

        return NormalizedTeamName(
            original=original,
            normalized=normalized,
            canonical=canonical,
            all_tokens=frozenset(
                canonical.split()
            ),
            distinctive_tokens=(
                distinctive_tokens
            ),
        )

    @classmethod
    def normalize_name(
        cls,
        team_name: str,
    ) -> str:
        normalized = (
            unicodedata.normalize(
                "NFKD",
                str(team_name)
                .casefold(),
            )
        )

        normalized = "".join(
            character
            for character
            in normalized
            if not unicodedata.combining(
                character
            )
        )

        normalized = (
            normalized.replace(
                "&",
                " and ",
            )
        )

        normalized = re.sub(
            r"\([^)]*\)",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"\[[^\]]*\]",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()

        return normalized

    def _build_alias_lookup(
        self,
    ) -> dict[str, str]:
        lookup: dict[str, str] = {}

        for (
            group_index,
            alias_group,
        ) in enumerate(
            self.VERIFIED_ALIAS_GROUPS
        ):
            canonical_identifier = (
                f"alias_group_{group_index}"
            )

            for alias in alias_group:
                normalized_alias = (
                    self.normalize_name(
                        alias
                    )
                )

                alias_identity = (
                    self._prepare_alias_name(
                        alias
                    )
                )

                if normalized_alias:
                    lookup[
                        normalized_alias
                    ] = (
                        canonical_identifier
                    )

                if alias_identity:
                    lookup[
                        alias_identity
                    ] = (
                        canonical_identifier
                    )

        return lookup

    def _prepare_alias_name(
        self,
        team_name: str,
    ) -> str:
        normalized = (
            self.normalize_name(
                team_name
            )
        )

        tokens = [
            token
            for token
            in normalized.split()
            if (
                token
                not in self.REMOVABLE_TOKENS
                and token
                not in self.COUNTRY_MARKERS
                and token
                not in self.RESERVE_MARKERS
            )
        ]

        return " ".join(
            tokens
        ).strip()

    def _rejected_result(
        self,
        api_team_name: str,
        reason: str,
    ) -> TeamMatchResult:
        return TeamMatchResult(
            api_team_name=(
                api_team_name
            ),
            matched_team_name=None,
            score=0.0,
            second_best_score=0.0,
            score_margin=0.0,
            accepted=False,
            reason=reason,
            candidates=(),
        )
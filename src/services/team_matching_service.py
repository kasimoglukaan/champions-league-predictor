from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Optional


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


class TeamMatchingService:
    """
    Safely matches bookmaker/API team names with model team names.

    Unlike a basic fuzzy matcher, this service does not allow generic
    words such as "Real", "City" or "United" to create false matches.
    """

    GENERIC_WORDS = {
        "1",
        "1900",
        "1904",
        "1905",
        "1907",
        "1908",
        "1910",
        "1912",
        "1913",
        "1919",
        "1920",
        "1924",
        "1927",
        "1929",
        "1946",
        "1967",
        "1970",
        "afc",
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
        "fc",
        "fk",
        "football",
        "fsv",
        "inter",
        "jk",
        "kv",
        "la",
        "le",
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
        "vfl",
        "vfb",
    }

    TEAM_ALIAS_GROUPS = (
        {
            "arsenal",
            "arsenal fc",
        },
        {
            "aston villa",
            "aston villa fc",
        },
        {
            "bayern munich",
            "bayern munchen",
            "fc bayern munich",
            "fc bayern munchen",
        },
        {
            "bayer leverkusen",
            "bayer 04 leverkusen",
            "leverkusen",
        },
        {
            "barcelona",
            "fc barcelona",
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
            "club atletico de madrid",
            "atletico madrid",
            "athletico madrid",
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
        },
        {
            "manchester united",
            "man united",
            "man utd",
        },
        {
            "newcastle",
            "newcastle united",
            "newcastle united fc",
        },
        {
            "paris saint germain",
            "paris saint-germain",
            "psg",
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
        },
        {
            "shakhtar",
            "shakhtar donetsk",
            "fk shakhtar donetsk",
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
        self.minimum_score = minimum_score
        self.minimum_margin = minimum_margin
        self.candidate_limit = candidate_limit

        self.alias_lookup = self._build_alias_lookup()

    def match(
        self,
        api_team_name: str,
        model_teams: Iterable[str],
    ) -> TeamMatchResult:
        cleaned_api_name = api_team_name.strip()

        if not cleaned_api_name:
            return TeamMatchResult(
                api_team_name=api_team_name,
                matched_team_name=None,
                score=0.0,
                second_best_score=0.0,
                score_margin=0.0,
                accepted=False,
                reason="API team name is empty.",
                candidates=(),
            )

        model_team_list = [
            str(team).strip()
            for team in model_teams
            if str(team).strip()
        ]

        if not model_team_list:
            return TeamMatchResult(
                api_team_name=cleaned_api_name,
                matched_team_name=None,
                score=0.0,
                second_best_score=0.0,
                score_margin=0.0,
                accepted=False,
                reason="The model team list is empty.",
                candidates=(),
            )

        candidates = [
            self._score_candidate(
                api_team_name=cleaned_api_name,
                model_team_name=model_team,
            )
            for model_team in model_team_list
        ]

        candidates.sort(
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        visible_candidates = tuple(
            candidates[: self.candidate_limit]
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

        accepted, reason = self._validate_match(
            api_team_name=cleaned_api_name,
            best_candidate=best_candidate,
            second_best_score=second_best_score,
            score_margin=score_margin,
        )

        return TeamMatchResult(
            api_team_name=cleaned_api_name,
            matched_team_name=(
                best_candidate.team_name
                if accepted
                else None
            ),
            score=best_candidate.score,
            second_best_score=second_best_score,
            score_margin=score_margin,
            accepted=accepted,
            reason=reason,
            candidates=visible_candidates,
        )

    def _score_candidate(
        self,
        api_team_name: str,
        model_team_name: str,
    ) -> TeamCandidate:
        api_normalized = self.normalize_name(
            api_team_name
        )

        model_normalized = self.normalize_name(
            model_team_name
        )

        if not api_normalized or not model_normalized:
            return TeamCandidate(
                team_name=model_team_name,
                score=0.0,
                distinctive_overlap=(),
            )

        if api_normalized == model_normalized:
            return TeamCandidate(
                team_name=model_team_name,
                score=1.0,
                distinctive_overlap=tuple(
                    sorted(
                        self.distinctive_tokens(
                            api_normalized
                        )
                    )
                ),
            )

        if self._same_alias_group(
            api_normalized,
            model_normalized,
        ):
            return TeamCandidate(
                team_name=model_team_name,
                score=0.99,
                distinctive_overlap=tuple(
                    sorted(
                        self.distinctive_tokens(
                            api_normalized
                        )
                        | self.distinctive_tokens(
                            model_normalized
                        )
                    )
                ),
            )

        api_tokens = set(
            api_normalized.split()
        )

        model_tokens = set(
            model_normalized.split()
        )

        api_distinctive = (
            self.distinctive_tokens(
                api_normalized
            )
        )

        model_distinctive = (
            self.distinctive_tokens(
                model_normalized
            )
        )

        distinctive_overlap = (
            api_distinctive
            & model_distinctive
        )

        # Critical protection:
        # Names sharing only generic terms such as Real, City or United
        # must never be accepted as the same team.
        if not distinctive_overlap:
            return TeamCandidate(
                team_name=model_team_name,
                score=0.0,
                distinctive_overlap=(),
            )

        all_distinctive_tokens = (
            api_distinctive
            | model_distinctive
        )

        distinctive_jaccard = (
            len(distinctive_overlap)
            / len(all_distinctive_tokens)
            if all_distinctive_tokens
            else 0.0
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

        sequence_score = SequenceMatcher(
            None,
            api_normalized,
            model_normalized,
        ).ratio()

        prefix_score = self._prefix_similarity(
            api_normalized,
            model_normalized,
        )

        score = (
            0.55 * distinctive_jaccard
            + 0.20 * token_jaccard
            + 0.15 * sequence_score
            + 0.10 * prefix_score
        )

        # One distinctive token is not sufficient when both names have
        # several different distinctive tokens.
        if (
            len(distinctive_overlap) == 1
            and len(all_distinctive_tokens) >= 3
        ):
            score *= 0.78

        return TeamCandidate(
            team_name=model_team_name,
            score=min(
                max(score, 0.0),
                1.0,
            ),
            distinctive_overlap=tuple(
                sorted(distinctive_overlap)
            ),
        )

    def _validate_match(
        self,
        api_team_name: str,
        best_candidate: TeamCandidate,
        second_best_score: float,
        score_margin: float,
    ) -> tuple[bool, str]:
        api_normalized = self.normalize_name(
            api_team_name
        )

        model_normalized = self.normalize_name(
            best_candidate.team_name
        )

        if api_normalized == model_normalized:
            return (
                True,
                "Exact normalized team-name match.",
            )

        if self._same_alias_group(
            api_normalized,
            model_normalized,
        ):
            return (
                True,
                "Verified team alias match.",
            )

        if not best_candidate.distinctive_overlap:
            return (
                False,
                "No distinctive team-name token matched. "
                "A match based only on generic words was rejected.",
            )

        if best_candidate.score < self.minimum_score:
            return (
                False,
                "The best candidate score is below the "
                f"required threshold of {self.minimum_score:.0%}.",
            )

        if score_margin < self.minimum_margin:
            return (
                False,
                "The best and second-best candidates are too close. "
                "The match is ambiguous and was rejected.",
            )

        return (
            True,
            "High-confidence distinctive-name match.",
        )

    @classmethod
    def normalize_name(
        cls,
        team_name: str,
    ) -> str:
        normalized = unicodedata.normalize(
            "NFKD",
            str(team_name).casefold(),
        )

        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(
                character
            )
        )

        normalized = normalized.replace(
            "&",
            " and ",
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

    @classmethod
    def distinctive_tokens(
        cls,
        normalized_name: str,
    ) -> set[str]:
        return {
            token
            for token in normalized_name.split()
            if (
                token not in cls.GENERIC_WORDS
                and len(token) >= 3
            )
        }

    def _same_alias_group(
        self,
        first_name: str,
        second_name: str,
    ) -> bool:
        first_canonical = self.alias_lookup.get(
            first_name
        )

        second_canonical = self.alias_lookup.get(
            second_name
        )

        return (
            first_canonical is not None
            and second_canonical is not None
            and first_canonical
            == second_canonical
        )

    def _build_alias_lookup(
        self,
    ) -> dict[str, str]:
        alias_lookup: dict[str, str] = {}

        for alias_group in self.TEAM_ALIAS_GROUPS:
            normalized_aliases = sorted(
                {
                    self.normalize_name(alias)
                    for alias in alias_group
                }
            )

            if not normalized_aliases:
                continue

            canonical_name = (
                normalized_aliases[0]
            )

            for alias in normalized_aliases:
                alias_lookup[alias] = (
                    canonical_name
                )

        return alias_lookup

    @staticmethod
    def _prefix_similarity(
        first_name: str,
        second_name: str,
    ) -> float:
        first_tokens = (
            first_name.split()
        )

        second_tokens = (
            second_name.split()
        )

        if (
            not first_tokens
            or not second_tokens
        ):
            return 0.0

        first_token = first_tokens[0]
        second_token = second_tokens[0]

        if first_token == second_token:
            return 1.0

        return SequenceMatcher(
            None,
            first_token,
            second_token,
        ).ratio()
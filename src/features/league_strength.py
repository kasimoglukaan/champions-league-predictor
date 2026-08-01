from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LeagueInformation:
    canonical_name: str
    strength: float
    is_domestic: bool


class LeagueStrengthResolver:
    """
    Resolves competition codes and names into a stable league-strength
    value.

    Strength values are relative modelling inputs, not official rankings.
    They must be validated through model experiments before production use.
    """

    DEFAULT_STRENGTH = 0.75

    LEAGUES = {
        "PREMIER_LEAGUE": LeagueInformation(
            canonical_name="PREMIER_LEAGUE",
            strength=1.00,
            is_domestic=True,
        ),
        "LA_LIGA": LeagueInformation(
            canonical_name="LA_LIGA",
            strength=0.96,
            is_domestic=True,
        ),
        "BUNDESLIGA": LeagueInformation(
            canonical_name="BUNDESLIGA",
            strength=0.94,
            is_domestic=True,
        ),
        "SERIE_A": LeagueInformation(
            canonical_name="SERIE_A",
            strength=0.93,
            is_domestic=True,
        ),
        "LIGUE_1": LeagueInformation(
            canonical_name="LIGUE_1",
            strength=0.88,
            is_domestic=True,
        ),
        "PRIMEIRA_LIGA": LeagueInformation(
            canonical_name="PRIMEIRA_LIGA",
            strength=0.82,
            is_domestic=True,
        ),
        "EREDIVISIE": LeagueInformation(
            canonical_name="EREDIVISIE",
            strength=0.81,
            is_domestic=True,
        ),
        "EFL_CHAMPIONSHIP": LeagueInformation(
            canonical_name="EFL_CHAMPIONSHIP",
            strength=0.77,
            is_domestic=True,
        ),
        "BELGIAN_FIRST_DIVISION": LeagueInformation(
            canonical_name="BELGIAN_FIRST_DIVISION",
            strength=0.75,
            is_domestic=True,
        ),
        "TURKISH_SUPER_LEAGUE": LeagueInformation(
            canonical_name="TURKISH_SUPER_LEAGUE",
            strength=0.74,
            is_domestic=True,
        ),
        "SCOTTISH_PREMIERSHIP": LeagueInformation(
            canonical_name="SCOTTISH_PREMIERSHIP",
            strength=0.70,
            is_domestic=True,
        ),
        "CHAMPIONS_LEAGUE": LeagueInformation(
            canonical_name="CHAMPIONS_LEAGUE",
            strength=1.00,
            is_domestic=False,
        ),
        "EUROPA_LEAGUE": LeagueInformation(
            canonical_name="EUROPA_LEAGUE",
            strength=0.90,
            is_domestic=False,
        ),
        "CONFERENCE_LEAGUE": LeagueInformation(
            canonical_name="CONFERENCE_LEAGUE",
            strength=0.80,
            is_domestic=False,
        ),
    }

    ALIASES = {
        "E0": "PREMIER_LEAGUE",
        "EPL": "PREMIER_LEAGUE",
        "ENGLISH_PREMIER_LEAGUE": "PREMIER_LEAGUE",
        "PREMIER_LEAGUE": "PREMIER_LEAGUE",

        "SP1": "LA_LIGA",
        "LA_LIGA": "LA_LIGA",
        "SPANISH_LA_LIGA": "LA_LIGA",

        "D1": "BUNDESLIGA",
        "BUNDESLIGA": "BUNDESLIGA",
        "GERMAN_BUNDESLIGA": "BUNDESLIGA",

        "I1": "SERIE_A",
        "SERIE_A": "SERIE_A",
        "ITALIAN_SERIE_A": "SERIE_A",

        "F1": "LIGUE_1",
        "LIGUE_1": "LIGUE_1",
        "FRENCH_LIGUE_1": "LIGUE_1",

        "P1": "PRIMEIRA_LIGA",
        "PRIMEIRA_LIGA": "PRIMEIRA_LIGA",
        "PORTUGAL_PRIMEIRA_LIGA": "PRIMEIRA_LIGA",

        "N1": "EREDIVISIE",
        "EREDIVISIE": "EREDIVISIE",
        "NETHERLANDS_EREDIVISIE": "EREDIVISIE",

        "E1": "EFL_CHAMPIONSHIP",
        "EFL_CHAMP": "EFL_CHAMPIONSHIP",
        "EFL_CHAMPIONSHIP": "EFL_CHAMPIONSHIP",
        "CHAMPIONSHIP": "EFL_CHAMPIONSHIP",

        "B1": "BELGIAN_FIRST_DIVISION",
        "BELGIUM_1": "BELGIAN_FIRST_DIVISION",
        "BELGIAN_FIRST_DIVISION": "BELGIAN_FIRST_DIVISION",

        "T1": "TURKISH_SUPER_LEAGUE",
        "SUPER_LIG": "TURKISH_SUPER_LEAGUE",
        "TURKISH_SUPER_LEAGUE": "TURKISH_SUPER_LEAGUE",

        "SC0": "SCOTTISH_PREMIERSHIP",
        "SCOTTISH_PREMIERSHIP": "SCOTTISH_PREMIERSHIP",

        "CL": "CHAMPIONS_LEAGUE",
        "UCL": "CHAMPIONS_LEAGUE",
        "CHAMPIONS_LEAGUE": "CHAMPIONS_LEAGUE",
        "UEFA_CHAMPIONS_LEAGUE": "CHAMPIONS_LEAGUE",

        "EL": "EUROPA_LEAGUE",
        "UEL": "EUROPA_LEAGUE",
        "EUROPA_LEAGUE": "EUROPA_LEAGUE",
        "UEFA_EUROPA_LEAGUE": "EUROPA_LEAGUE",

        "ECL": "CONFERENCE_LEAGUE",
        "UECL": "CONFERENCE_LEAGUE",
        "CONFERENCE_LEAGUE": "CONFERENCE_LEAGUE",
        "UEFA_CONFERENCE_LEAGUE": "CONFERENCE_LEAGUE",
        "UEFA_EUROPA_CONFERENCE_LEAGUE": "CONFERENCE_LEAGUE",
    }

    @classmethod
    def normalize_competition(
        cls,
        competition: object,
    ) -> str:
        normalized = str(
            competition
        ).strip().upper()

        normalized = re.sub(
            r"[^A-Z0-9]+",
            "_",
            normalized,
        )

        normalized = re.sub(
            r"_+",
            "_",
            normalized,
        ).strip("_")

        return normalized

    @classmethod
    def canonical_competition(
        cls,
        competition: object,
    ) -> str:
        normalized = cls.normalize_competition(
            competition
        )

        return cls.ALIASES.get(
            normalized,
            normalized,
        )

    @classmethod
    def get_information(
        cls,
        competition: object,
    ) -> LeagueInformation:
        canonical = cls.canonical_competition(
            competition
        )

        return cls.LEAGUES.get(
            canonical,
            LeagueInformation(
                canonical_name=canonical or "UNKNOWN",
                strength=cls.DEFAULT_STRENGTH,
                is_domestic=False,
            ),
        )

    @classmethod
    def get_strength(
        cls,
        competition: object,
    ) -> float:
        return cls.get_information(
            competition
        ).strength

    @classmethod
    def is_domestic(
        cls,
        competition: object,
    ) -> bool:
        return cls.get_information(
            competition
        ).is_domestic

    @classmethod
    def is_champions_league(
        cls,
        competition: object,
    ) -> bool:
        return (
            cls.canonical_competition(
                competition
            )
            == "CHAMPIONS_LEAGUE"
        )

    @classmethod
    def resolve_team_strength(
        cls,
        team_league: Optional[str],
        current_competition: object,
    ) -> float:
        if team_league:
            return cls.get_strength(
                team_league
            )

        current_information = (
            cls.get_information(
                current_competition
            )
        )

        if current_information.is_domestic:
            return current_information.strength

        return cls.DEFAULT_STRENGTH
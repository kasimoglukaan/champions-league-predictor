from datetime import datetime

from src.models.elo_model import EloModel
from src.models.match import Match
from src.models.poisson_model import PoissonModel
from src.models.team_statistics import TeamStatistics


def create_model() -> PoissonModel:
    matches = [
        Match(
            date=datetime(2025, 1, 1),
            home_team="Real Madrid",
            away_team="Bayern Munich",
            home_goals=2,
            away_goals=1,
        ),
        Match(
            date=datetime(2025, 1, 2),
            home_team="Bayern Munich",
            away_team="Real Madrid",
            home_goals=1,
            away_goals=1,
        ),
        Match(
            date=datetime(2025, 1, 3),
            home_team="Real Madrid",
            away_team="Arsenal",
            home_goals=3,
            away_goals=1,
        ),
        Match(
            date=datetime(2025, 1, 4),
            home_team="Arsenal",
            away_team="Bayern Munich",
            home_goals=0,
            away_goals=2,
        ),
    ]

    elo_model = EloModel()
    elo_model.train(matches)

    statistics = TeamStatistics()
    statistics.calculate(matches)

    return PoissonModel(
        team_statistics=statistics,
        elo_model=elo_model,
    )


def test_probabilities_total_one() -> None:
    model = create_model()

    prediction = model.predict(
        "Real Madrid",
        "Bayern Munich",
    )

    total_probability = (
        prediction.home_win_probability
        + prediction.draw_probability
        + prediction.away_win_probability
    )

    assert abs(
        total_probability - 1.0
    ) < 0.0001


def test_expected_goals_are_positive() -> None:
    model = create_model()

    prediction = model.predict(
        "Real Madrid",
        "Bayern Munich",
    )

    assert (
        prediction.expected_home_goals
        > 0
    )

    assert (
        prediction.expected_away_goals
        > 0
    )


def test_same_team_is_not_allowed() -> None:
    model = create_model()

    try:
        model.predict(
            "Real Madrid",
            "Real Madrid",
        )

        assert False

    except ValueError:
        assert True
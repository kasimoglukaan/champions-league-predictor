from datetime import datetime

from src.models.elo_model import EloModel
from src.models.match import Match


def test_winner_gains_elo_points() -> None:
    model = EloModel(
        initial_rating=1500,
        k_factor=30,
        home_advantage=0,
    )

    match = Match(
        date=datetime(2025, 1, 1),
        home_team="Real Madrid",
        away_team="Bayern Munich",
        home_goals=2,
        away_goals=1,
    )

    model.update_ratings(match)

    assert (
        model.get_rating("Real Madrid")
        > 1500
    )

    assert (
        model.get_rating("Bayern Munich")
        < 1500
    )


def test_draw_changes_equal_ratings_equally() -> None:
    model = EloModel(
        initial_rating=1500,
        k_factor=30,
        home_advantage=0,
    )

    match = Match(
        date=datetime(2025, 1, 1),
        home_team="Arsenal",
        away_team="Barcelona",
        home_goals=1,
        away_goals=1,
    )

    model.update_ratings(match)

    assert (
        model.get_rating("Arsenal")
        == 1500
    )

    assert (
        model.get_rating("Barcelona")
        == 1500
    )


def test_rating_difference_contains_home_advantage() -> None:
    model = EloModel(
        initial_rating=1500,
        home_advantage=80,
    )

    difference = model.get_rating_difference(
        "Inter Milan",
        "PSG",
    )

    assert difference == 80
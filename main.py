from src.services.predictor_service import (
    PredictorService,
)


def print_prediction(
    service: PredictorService,
    home_team: str,
    away_team: str,
) -> None:
    prediction = service.predict_match(
        home_team,
        away_team,
    )

    simulation = service.simulate_match(
        home_team,
        away_team,
    )

    print()
    print("=" * 50)
    print(f"{home_team} vs {away_team}")
    print("=" * 50)

    print(
        f"Expected goals: "
        f"{prediction.expected_home_goals:.2f} - "
        f"{prediction.expected_away_goals:.2f}"
    )

    print(
        f"Most likely score: "
        f"{prediction.most_likely_score}"
    )

    print()
    print("Poisson probabilities:")

    print(
        f"{home_team} win: "
        f"{prediction.home_win_probability:.1%}"
    )

    print(
        f"Draw: "
        f"{prediction.draw_probability:.1%}"
    )

    print(
        f"{away_team} win: "
        f"{prediction.away_win_probability:.1%}"
    )

    print()
    print("Monte Carlo probabilities:")

    print(
        f"{home_team} win: "
        f"{simulation.home_win_probability:.1%}"
    )

    print(
        f"Draw: "
        f"{simulation.draw_probability:.1%}"
    )

    print(
        f"{away_team} win: "
        f"{simulation.away_win_probability:.1%}"
    )


def main() -> None:
    service = PredictorService(
        csv_path="data/raw/matches.csv"
    )

    service.train()

    teams = service.get_teams()

    print("Available teams:")

    for team in teams:
        print(f"- {team}")

    print_prediction(
        service=service,
        home_team="Real Madrid",
        away_team="Bayern Munich",
    )


if __name__ == "__main__":
    main()
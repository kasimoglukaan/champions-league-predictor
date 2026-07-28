import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.services.predictor_service import (
    PredictorService,
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "matches.csv"
)


@st.cache_resource
def load_service() -> PredictorService:
    service = PredictorService(
        csv_path=str(DATA_PATH)
    )

    service.train()

    return service


def probability_dataframe(
    home_team: str,
    away_team: str,
    home_probability: float,
    draw_probability: float,
    away_probability: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Result": [
                f"{home_team} win",
                "Draw",
                f"{away_team} win",
            ],
            "Probability": [
                home_probability,
                draw_probability,
                away_probability,
            ],
        }
    )


def main() -> None:
    st.set_page_config(
        page_title=(
            "Champions League Match Predictor"
        ),
        page_icon="⚽",
        layout="wide",
    )

    st.title(
        "⚽ Champions League Match Predictor"
    )

    st.write(
        "Select two teams to calculate match "
        "probabilities using Elo and Poisson models."
    )

    try:
        service = load_service()

    except Exception as error:
        st.error(
            f"Application could not start: {error}"
        )
        st.stop()

    teams = service.get_teams()

    first_column, second_column = st.columns(2)

    with first_column:
        home_team = st.selectbox(
            "Home team",
            options=teams,
            index=(
                teams.index("Real Madrid")
                if "Real Madrid" in teams
                else 0
            ),
        )

    with second_column:
        default_away_index = 1

        if "Bayern Munich" in teams:
            default_away_index = teams.index(
                "Bayern Munich"
            )

        away_team = st.selectbox(
            "Away team",
            options=teams,
            index=default_away_index,
        )

    if home_team == away_team:
        st.warning(
            "Please select two different teams."
        )
        st.stop()

    if st.button(
        "Predict Match",
        type="primary",
        use_container_width=True,
    ):
        prediction = service.predict_match(
            home_team,
            away_team,
        )

        simulation = service.simulate_match(
            home_team,
            away_team,
        )

        st.subheader(
            f"{home_team} vs {away_team}"
        )

        metric_one, metric_two, metric_three = (
            st.columns(3)
        )

        with metric_one:
            st.metric(
                label=f"{home_team} win",
                value=(
                    f"{prediction.home_win_probability:.1%}"
                ),
            )

        with metric_two:
            st.metric(
                label="Draw",
                value=(
                    f"{prediction.draw_probability:.1%}"
                ),
            )

        with metric_three:
            st.metric(
                label=f"{away_team} win",
                value=(
                    f"{prediction.away_win_probability:.1%}"
                ),
            )

        score_column, goals_column = st.columns(2)

        with score_column:
            st.metric(
                label="Most likely score",
                value=(
                    prediction.most_likely_score
                ),
            )

        with goals_column:
            st.metric(
                label="Expected goals",
                value=(
                    f"{prediction.expected_home_goals:.2f}"
                    f" - "
                    f"{prediction.expected_away_goals:.2f}"
                ),
            )

        probabilities = probability_dataframe(
            home_team=home_team,
            away_team=away_team,
            home_probability=(
                prediction.home_win_probability
            ),
            draw_probability=(
                prediction.draw_probability
            ),
            away_probability=(
                prediction.away_win_probability
            ),
        )

        st.subheader(
            "Poisson model probabilities"
        )

        st.bar_chart(
            probabilities,
            x="Result",
            y="Probability",
        )

        st.subheader(
            "Monte Carlo simulation"
        )

        simulation_columns = st.columns(3)

        with simulation_columns[0]:
            st.metric(
                f"{home_team} simulated wins",
                f"{simulation.home_win_probability:.1%}",
            )

        with simulation_columns[1]:
            st.metric(
                "Simulated draws",
                f"{simulation.draw_probability:.1%}",
            )

        with simulation_columns[2]:
            st.metric(
                f"{away_team} simulated wins",
                f"{simulation.away_win_probability:.1%}",
            )

    st.divider()

    st.subheader("Elo rankings")

    ranking_dataframe = pd.DataFrame(
        service.get_rankings()
    )

    st.dataframe(
        ranking_dataframe,
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
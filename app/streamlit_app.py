import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.services.ml_prediction_service import (
    MLPredictionService,
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_matches.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "match_model.joblib"
)


@st.cache_resource
def load_prediction_service(
) -> MLPredictionService:
    service = MLPredictionService(
        data_path=str(DATA_PATH),
        model_path=str(MODEL_PATH),
    )

    service.load()

    return service


def create_probability_table(
    prediction,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Result": [
                (
                    f"{prediction.home_team} "
                    "win"
                ),
                "Draw",
                (
                    f"{prediction.away_team} "
                    "win"
                ),
            ],
            "Probability": [
                prediction
                .home_win_probability,

                prediction
                .draw_probability,

                prediction
                .away_win_probability,
            ],
        }
    )


def create_simulation_probability_table(
    prediction,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Result": [
                (
                    f"{prediction.home_team} "
                    "win"
                ),
                "Draw",
                (
                    f"{prediction.away_team} "
                    "win"
                ),
            ],
            "Probability": [
                prediction
                .poisson_home_probability,

                prediction
                .poisson_draw_probability,

                prediction
                .poisson_away_probability,
            ],
        }
    )


def create_model_comparison_table(
    prediction,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Result": [
                (
                    f"{prediction.home_team} "
                    "win"
                ),
                "Draw",
                (
                    f"{prediction.away_team} "
                    "win"
                ),
            ],
            "Machine Learning": [
                prediction
                .home_win_probability,

                prediction
                .draw_probability,

                prediction
                .away_win_probability,
            ],
            "Poisson Simulation": [
                prediction
                .poisson_home_probability,

                prediction
                .poisson_draw_probability,

                prediction
                .poisson_away_probability,
            ],
        }
    )


def display_prediction_factors(
    prediction,
) -> None:
    st.subheader(
        "Why did the model choose "
        "this result?"
    )

    st.caption(
        "These explanations summarize "
        "important match features used by "
        "the prediction system. They are "
        "not guarantees or exact causal "
        "model explanations."
    )

    positive_column, negative_column = (
        st.columns(2)
    )

    with positive_column:
        st.markdown(
            "### Factors supporting "
            "the prediction"
        )

        if (
            prediction
            .positive_factors
        ):
            for factor in (
                prediction
                .positive_factors
            ):
                st.success(
                    f"**{factor.title}**\n\n"
                    f"{factor.explanation}"
                )

        else:
            st.info(
                "No strong supporting "
                "factor was identified."
            )

    with negative_column:
        st.markdown(
            "### Factors creating "
            "uncertainty"
        )

        if (
            prediction
            .negative_factors
        ):
            for factor in (
                prediction
                .negative_factors
            ):
                st.warning(
                    f"**{factor.title}**\n\n"
                    f"{factor.explanation}"
                )

        else:
            st.info(
                "No major opposing factor "
                "was identified. The result "
                "should still be interpreted "
                "as probabilistic."
            )


def display_machine_learning_result(
    prediction,
) -> None:
    st.subheader(
        "Machine-learning prediction"
    )

    first_metric, second_metric, third_metric = (
        st.columns(3)
    )

    with first_metric:
        st.metric(
            f"{prediction.home_team} win",
            (
                f"{prediction.home_win_probability:.1%}"
            ),
        )

    with second_metric:
        st.metric(
            "Draw",
            (
                f"{prediction.draw_probability:.1%}"
            ),
        )

    with third_metric:
        st.metric(
            f"{prediction.away_team} win",
            (
                f"{prediction.away_win_probability:.1%}"
            ),
        )

    result_column, confidence_column = (
        st.columns(2)
    )

    with result_column:
        st.metric(
            "Model prediction",
            prediction
            .predicted_result_text,
        )

    with confidence_column:
        st.metric(
            "Confidence",
            (
                f"{prediction.confidence_label} "
                f"({prediction.confidence:.1%})"
            ),
        )

    if (
        prediction.confidence_label
        == "LOW"
    ):
        st.warning(
            "The machine-learning "
            "probabilities are close. "
            "This should be treated as a "
            "low-confidence prediction."
        )

    probability_table = (
        create_probability_table(
            prediction
        )
    )

    st.bar_chart(
        probability_table,
        x="Result",
        y="Probability",
    )


def display_poisson_simulation(
    prediction,
) -> None:
    st.subheader(
        "Poisson and Monte Carlo simulation"
    )

    st.caption(
        "The simulation generates 20,000 "
        "possible matches using estimated "
        "goal rates based on recent attack, "
        "defence and Elo strength."
    )

    home_metric, draw_metric, away_metric = (
        st.columns(3)
    )

    with home_metric:
        st.metric(
            (
                f"{prediction.home_team} "
                "simulation win"
            ),
            (
                f"{prediction.poisson_home_probability:.1%}"
            ),
        )

    with draw_metric:
        st.metric(
            "Simulation draw",
            (
                f"{prediction.poisson_draw_probability:.1%}"
            ),
        )

    with away_metric:
        st.metric(
            (
                f"{prediction.away_team} "
                "simulation win"
            ),
            (
                f"{prediction.poisson_away_probability:.1%}"
            ),
        )

    expected_goals_column, score_column = (
        st.columns(2)
    )

    with expected_goals_column:
        st.metric(
            "Expected goals",
            (
                f"{prediction.expected_home_goals:.2f}"
                " – "
                f"{prediction.expected_away_goals:.2f}"
            ),
        )

        st.caption(
            (
                f"{prediction.home_team}: "
                f"{prediction.expected_home_goals:.2f} xG proxy"
            )
            + " | "
            + (
                f"{prediction.away_team}: "
                f"{prediction.expected_away_goals:.2f} xG proxy"
            )
        )

    with score_column:
        st.metric(
            "Most likely score",
            (
                f"{prediction.most_likely_home_goals}"
                "-"
                f"{prediction.most_likely_away_goals}"
            ),
        )

        st.caption(
            prediction
            .most_likely_score_text
        )

    simulation_table = (
        create_simulation_probability_table(
            prediction
        )
    )

    st.bar_chart(
        simulation_table,
        x="Result",
        y="Probability",
    )


def display_model_comparison(
    prediction,
) -> None:
    st.subheader(
        "Model comparison"
    )

    comparison_table = (
        create_model_comparison_table(
            prediction
        )
    )

    st.dataframe(
        comparison_table.style.format(
            {
                "Machine Learning": (
                    "{:.1%}"
                ),
                "Poisson Simulation": (
                    "{:.1%}"
                ),
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.bar_chart(
        comparison_table,
        x="Result",
        y=[
            "Machine Learning",
            "Poisson Simulation",
        ],
    )

    ml_probabilities = {
        "H": (
            prediction
            .home_win_probability
        ),
        "D": (
            prediction
            .draw_probability
        ),
        "A": (
            prediction
            .away_win_probability
        ),
    }

    simulation_probabilities = {
        "H": (
            prediction
            .poisson_home_probability
        ),
        "D": (
            prediction
            .poisson_draw_probability
        ),
        "A": (
            prediction
            .poisson_away_probability
        ),
    }

    ml_result = max(
        ml_probabilities,
        key=ml_probabilities.get,
    )

    simulation_result = max(
        simulation_probabilities,
        key=simulation_probabilities.get,
    )

    if ml_result == simulation_result:
        st.success(
            "The machine-learning model and "
            "the Poisson simulation agree on "
            "the most likely result."
        )

    else:
        st.warning(
            "The machine-learning model and "
            "the Poisson simulation disagree. "
            "This match should be treated with "
            "additional uncertainty."
        )


def display_team_information(
    service,
    prediction,
) -> None:
    st.subheader(
        "Current team information"
    )

    home_summary = (
        service.get_team_summary(
            prediction.home_team
        )
    )

    away_summary = (
        service.get_team_summary(
            prediction.away_team
        )
    )

    summary_dataframe = pd.DataFrame(
        [
            {
                "Team": (
                    prediction.home_team
                ),
                "Elo": (
                    home_summary["elo"]
                ),
                "Recent points per match": (
                    home_summary[
                        "form_points"
                    ]
                ),
                "Recent goals scored": (
                    home_summary[
                        "goals_scored"
                    ]
                ),
                "Recent goals conceded": (
                    home_summary[
                        "goals_conceded"
                    ]
                ),
                "Recent win rate": (
                    home_summary[
                        "win_rate"
                    ]
                ),
            },
            {
                "Team": (
                    prediction.away_team
                ),
                "Elo": (
                    away_summary["elo"]
                ),
                "Recent points per match": (
                    away_summary[
                        "form_points"
                    ]
                ),
                "Recent goals scored": (
                    away_summary[
                        "goals_scored"
                    ]
                ),
                "Recent goals conceded": (
                    away_summary[
                        "goals_conceded"
                    ]
                ),
                "Recent win rate": (
                    away_summary[
                        "win_rate"
                    ]
                ),
            },
        ]
    )

    st.dataframe(
        summary_dataframe.style.format(
            {
                "Elo": "{:.2f}",
                "Recent points per match": (
                    "{:.2f}"
                ),
                "Recent goals scored": (
                    "{:.2f}"
                ),
                "Recent goals conceded": (
                    "{:.2f}"
                ),
                "Recent win rate": (
                    "{:.1%}"
                ),
            }
        ),
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(
        page_title=(
            "Champions League "
            "ML Predictor"
        ),
        page_icon="🏆",
        layout="wide",
    )

    st.title(
        "🏆 Champions League "
        "Hybrid Predictor"
    )

    st.write(
        "Match predictions combine a "
        "machine-learning classifier with "
        "a Poisson-based Monte Carlo goal "
        "simulation."
    )

    st.caption(
        "The production machine-learning "
        "model was trained using 7,592 "
        "historical matches."
    )

    try:
        service = (
            load_prediction_service()
        )

    except Exception as error:
        st.error(
            "The prediction system "
            f"could not start: {error}"
        )

        st.stop()

    teams = service.get_teams()

    if len(teams) < 2:
        st.error(
            "At least two teams are "
            "required for prediction."
        )

        st.stop()

    home_column, away_column = (
        st.columns(2)
    )

    with home_column:
        default_home_index = 0

        if "FC Barcelona" in teams:
            default_home_index = (
                teams.index(
                    "FC Barcelona"
                )
            )

        home_team = st.selectbox(
            "Home team",
            options=teams,
            index=default_home_index,
        )

    with away_column:
        default_away_index = min(
            1,
            len(teams) - 1,
        )

        if "Celtic FC" in teams:
            default_away_index = (
                teams.index(
                    "Celtic FC"
                )
            )

        away_team = st.selectbox(
            "Away team",
            options=teams,
            index=default_away_index,
        )

    if home_team == away_team:
        st.warning(
            "Please select two "
            "different teams."
        )

        st.stop()

    predict_button = st.button(
        "Predict Match",
        type="primary",
        width="stretch",
    )

    if not predict_button:
        return

    try:
        prediction = service.predict(
            home_team=home_team,
            away_team=away_team,
        )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )

        st.stop()

    st.header(
        f"{home_team} vs {away_team}"
    )

    display_machine_learning_result(
        prediction
    )

    st.divider()

    display_poisson_simulation(
        prediction
    )

    st.divider()

    display_model_comparison(
        prediction
    )

    st.divider()

    display_prediction_factors(
        prediction
    )

    st.divider()

    display_team_information(
        service=service,
        prediction=prediction,
    )

    st.caption(
        "Production ML test accuracy: "
        "58.20% on 378 unseen Champions "
        "League matches. The Poisson values "
        "are simulation estimates and have "
        "not yet been independently validated "
        "as a production model. Predictions "
        "are probabilities, not guarantees."
    )


if __name__ == "__main__":
    main()
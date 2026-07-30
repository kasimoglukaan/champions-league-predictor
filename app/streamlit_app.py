import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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


from src.models.bookmaker_odds import (
    MatchOdds,
    OddsEvent,
)
from src.services.betting_insight_service import (
    BettingInsightService,
)
from src.services.ml_prediction_service import (
    MLPredictionService,
)
from src.services.odds_api_service import (
    OddsAPIError,
    OddsAPIService,
)
from src.services.value_bet_service import (
    ValueBetService,
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


@st.cache_resource
def load_betting_insight_service(
) -> BettingInsightService:
    return BettingInsightService(
        strong_threshold=0.72,
        moderate_threshold=0.62,
    )


@st.cache_resource
def load_value_bet_service(
) -> ValueBetService:
    return ValueBetService(
        minimum_model_probability=0.45,
        minimum_edge=0.03,
        minimum_expected_value=0.03,
        maximum_decimal_odds=8.00,
    )


def get_api_key() -> Optional[str]:
    try:
        api_key = str(
            st.secrets["ODDS_API_KEY"]
        ).strip()

    except (
        KeyError,
        FileNotFoundError,
    ):
        return None

    if not api_key:
        return None

    return api_key


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def get_active_leagues(
    api_key: str,
) -> dict:
    service = OddsAPIService(
        api_key=api_key,
        region="eu",
    )

    return (
        service.get_active_soccer_leagues()
    )


@st.cache_data(
    ttl=600,
    show_spinner=False,
)
def get_league_events(
    api_key: str,
    sport_key: str,
) -> list:
    service = OddsAPIService(
        api_key=api_key,
        region="eu",
    )

    return service.get_league_events(
        sport_key=sport_key,
    )


@st.cache_data(
    ttl=120,
    show_spinner=False,
)
def get_live_event_odds(
    api_key: str,
    event_id: str,
    sport_key: str,
    sport_title: str,
    commence_time: str,
    home_team: str,
    away_team: str,
) -> MatchOdds:
    service = OddsAPIService(
        api_key=api_key,
        region="eu",
    )

    event = OddsEvent(
        event_id=event_id,
        sport_key=sport_key,
        sport_title=sport_title,
        commence_time=commence_time,
        home_team=home_team,
        away_team=away_team,
    )

    return service.get_event_odds(
        event=event,
    )


def find_model_team(
    api_team_name: str,
    model_teams: list,
) -> tuple:
    best_team = None
    best_score = 0.0

    for model_team in model_teams:
        score = (
            OddsAPIService
            ._team_similarity(
                api_team_name,
                model_team,
            )
        )

        if score > best_score:
            best_score = score
            best_team = model_team

    return best_team, best_score


def format_event_time(
    commence_time: str,
) -> str:
    if not commence_time:
        return "Time unavailable"

    try:
        parsed_time = (
            datetime
            .fromisoformat(
                commence_time.replace(
                    "Z",
                    "+00:00",
                )
            )
        )

        return parsed_time.strftime(
            "%d %b %Y, %H:%M UTC"
        )

    except ValueError:
        return commence_time


def create_match_result_table(
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
                (
                    prediction
                    .poisson_home_probability
                ),
                (
                    prediction
                    .poisson_draw_probability
                ),
                (
                    prediction
                    .poisson_away_probability
                ),
            ],
        }
    )


def create_goal_market_table(
    prediction,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Market": [
                (
                    "Both teams to score "
                    "— Yes"
                ),
                (
                    "Both teams to score "
                    "— No"
                ),
                "Over 1.5 goals",
                "Over 2.5 goals",
                "Under 2.5 goals",
                "Under 3.5 goals",
            ],
            "Probability": [
                (
                    prediction
                    .btts_yes_probability
                ),
                (
                    prediction
                    .btts_no_probability
                ),
                (
                    prediction
                    .over_1_5_probability
                ),
                (
                    prediction
                    .over_2_5_probability
                ),
                (
                    prediction
                    .under_2_5_probability
                ),
                (
                    prediction
                    .under_3_5_probability
                ),
            ],
        }
    )


def create_best_h2h_odds_table(
    match_odds: MatchOdds,
) -> pd.DataFrame:
    selections = [
        match_odds.home_team,
        "Draw",
        match_odds.away_team,
    ]

    rows = []

    for selection in selections:
        best_price = (
            match_odds.get_best_price(
                market_key="h2h",
                selection=selection,
            )
        )

        if best_price is None:
            continue

        rows.append(
            {
                "Selection": selection,
                "Best odds": (
                    best_price.odds
                ),
                "Bookmaker": (
                    best_price
                    .bookmaker_title
                ),
                "Implied probability": (
                    best_price
                    .implied_probability
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def create_best_totals_odds_table(
    match_odds: MatchOdds,
) -> pd.DataFrame:
    requested_markets = [
        (
            "Over",
            1.5,
            "Over 1.5 goals",
        ),
        (
            "Under",
            1.5,
            "Under 1.5 goals",
        ),
        (
            "Over",
            2.5,
            "Over 2.5 goals",
        ),
        (
            "Under",
            2.5,
            "Under 2.5 goals",
        ),
        (
            "Over",
            3.5,
            "Over 3.5 goals",
        ),
        (
            "Under",
            3.5,
            "Under 3.5 goals",
        ),
    ]

    rows = []

    for (
        api_selection,
        point,
        display_selection,
    ) in requested_markets:
        best_price = (
            match_odds.get_best_price(
                market_key="totals",
                selection=api_selection,
                point=point,
            )
        )

        if best_price is None:
            continue

        rows.append(
            {
                "Selection": (
                    display_selection
                ),
                "Best odds": (
                    best_price.odds
                ),
                "Bookmaker": (
                    best_price
                    .bookmaker_title
                ),
                "Implied probability": (
                    best_price
                    .implied_probability
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def display_match_prediction(
    prediction,
    api_event: OddsEvent,
) -> None:
    st.subheader(
        "Match prediction"
    )

    home_column, draw_column, away_column = (
        st.columns(3)
    )

    with home_column:
        st.metric(
            (
                f"{api_event.home_team} "
                "win"
            ),
            (
                f"{prediction.poisson_home_probability:.1%}"
            ),
        )

    with draw_column:
        st.metric(
            "Draw",
            (
                f"{prediction.poisson_draw_probability:.1%}"
            ),
        )

    with away_column:
        st.metric(
            (
                f"{api_event.away_team} "
                "win"
            ),
            (
                f"{prediction.poisson_away_probability:.1%}"
            ),
        )

    score_column, goals_column = (
        st.columns(2)
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

    with goals_column:
        st.metric(
            "Expected goals",
            (
                f"{prediction.expected_home_goals:.2f}"
                " – "
                f"{prediction.expected_away_goals:.2f}"
            ),
        )

    result_table = (
        create_match_result_table(
            prediction
        )
    )

    result_table["Result"] = [
        f"{api_event.home_team} win",
        "Draw",
        f"{api_event.away_team} win",
    ]

    st.bar_chart(
        result_table,
        x="Result",
        y="Probability",
    )


def display_goal_markets(
    prediction,
) -> None:
    st.subheader(
        "Goal markets"
    )

    btts_yes_column, btts_no_column = (
        st.columns(2)
    )

    with btts_yes_column:
        st.metric(
            (
                "Both teams to score "
                "— Yes"
            ),
            (
                f"{prediction.btts_yes_probability:.1%}"
            ),
        )

    with btts_no_column:
        st.metric(
            (
                "Both teams to score "
                "— No"
            ),
            (
                f"{prediction.btts_no_probability:.1%}"
            ),
        )

    (
        over_1_5_column,
        over_2_5_column,
        under_3_5_column,
    ) = st.columns(3)

    with over_1_5_column:
        st.metric(
            "Over 1.5 goals",
            (
                f"{prediction.over_1_5_probability:.1%}"
            ),
        )

    with over_2_5_column:
        st.metric(
            "Over 2.5 goals",
            (
                f"{prediction.over_2_5_probability:.1%}"
            ),
        )

    with under_3_5_column:
        st.metric(
            "Under 3.5 goals",
            (
                f"{prediction.under_3_5_probability:.1%}"
            ),
        )

    goal_market_table = (
        create_goal_market_table(
            prediction
        )
    )

    st.dataframe(
        goal_market_table.style.format(
            {
                "Probability": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def display_double_chance(
    prediction,
    api_event: OddsEvent,
) -> None:
    st.subheader(
        "Double chance"
    )

    home_or_draw = (
        prediction
        .poisson_home_probability
        + prediction
        .poisson_draw_probability
    )

    away_or_draw = (
        prediction
        .poisson_away_probability
        + prediction
        .poisson_draw_probability
    )

    home_column, away_column = (
        st.columns(2)
    )

    with home_column:
        st.metric(
            (
                f"{api_event.home_team} "
                "or draw (1X)"
            ),
            f"{home_or_draw:.1%}",
        )

    with away_column:
        st.metric(
            (
                f"{api_event.away_team} "
                "or draw (X2)"
            ),
            f"{away_or_draw:.1%}",
        )


def display_model_signal(
    prediction,
    betting_service,
) -> None:
    st.subheader(
        "Strongest model signal"
    )

    report = betting_service.generate(
        home_team=prediction.home_team,
        away_team=prediction.away_team,
        home_win_probability=(
            prediction
            .poisson_home_probability
        ),
        draw_probability=(
            prediction
            .poisson_draw_probability
        ),
        away_win_probability=(
            prediction
            .poisson_away_probability
        ),
        btts_probability=(
            prediction
            .btts_probability
        ),
        over_1_5_probability=(
            prediction
            .over_1_5_probability
        ),
        over_2_5_probability=(
            prediction
            .over_2_5_probability
        ),
        under_2_5_probability=(
            prediction
            .under_2_5_probability
        ),
        under_3_5_probability=(
            prediction
            .under_3_5_probability
        ),
    )

    if (
        report.no_strong_signal
        or report.best_insight is None
    ):
        st.warning(
            "No sufficiently strong model "
            "signal was identified."
        )

        return

    best_insight = report.best_insight

    st.success(
        f"### {best_insight.selection}\n\n"
        f"**Market:** "
        f"{best_insight.market}\n\n"
        f"**Probability:** "
        f"{best_insight.probability:.1%}\n\n"
        f"**Signal:** "
        f"{best_insight.confidence_label}"
    )


def display_live_odds(
    match_odds: MatchOdds,
) -> None:
    st.subheader(
        "Live bookmaker odds"
    )

    st.caption(
        "The tables show the highest available "
        "decimal price returned by the connected "
        "odds provider for each selection."
    )

    st.markdown(
        "### Match result odds"
    )

    h2h_table = (
        create_best_h2h_odds_table(
            match_odds
        )
    )

    if h2h_table.empty:
        st.info(
            "No match-result odds were returned."
        )

    else:
        st.dataframe(
            h2h_table.style.format(
                {
                    "Best odds": "{:.2f}",
                    (
                        "Implied probability"
                    ): "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        "### Total-goal odds"
    )

    totals_table = (
        create_best_totals_odds_table(
            match_odds
        )
    )

    if totals_table.empty:
        st.info(
            "No total-goal odds were returned."
        )

    else:
        st.dataframe(
            totals_table.style.format(
                {
                    "Best odds": "{:.2f}",
                    (
                        "Implied probability"
                    ): "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    credit_columns = st.columns(3)

    with credit_columns[0]:
        st.metric(
            "API request cost",
            (
                match_odds.request_cost
                if (
                    match_odds
                    .request_cost
                    is not None
                )
                else "Unknown"
            ),
        )

    with credit_columns[1]:
        st.metric(
            "Credits used",
            (
                match_odds.requests_used
                if (
                    match_odds
                    .requests_used
                    is not None
                )
                else "Unknown"
            ),
        )

    with credit_columns[2]:
        st.metric(
            "Credits remaining",
            (
                match_odds
                .requests_remaining
                if (
                    match_odds
                    .requests_remaining
                    is not None
                )
                else "Unknown"
            ),
        )


def display_value_analysis(
    prediction,
    match_odds: MatchOdds,
    value_service: ValueBetService,
) -> None:
    st.subheader(
        "Value analysis"
    )

    st.caption(
        "Model probabilities are compared with "
        "the implied probability of the best "
        "available bookmaker price."
    )

    report = value_service.analyse(
        prediction=prediction,
        match_odds=match_odds,
    )

    all_candidates = (
        report.opportunities
        + report.rejected_markets
    )

    if not all_candidates:
        st.info(
            "There are not enough matching odds "
            "to perform a value analysis."
        )

        return

    analysis_rows = []

    for value_bet in all_candidates:
        is_value = (
            value_bet
            in report.opportunities
        )

        analysis_rows.append(
            {
                "Market": (
                    value_bet.market_name
                ),
                "Selection": (
                    value_bet.selection
                ),
                "Bookmaker": (
                    value_bet.bookmaker
                ),
                "Odds": (
                    value_bet.decimal_odds
                ),
                "Model probability": (
                    value_bet
                    .model_probability
                ),
                "Market probability": (
                    value_bet
                    .implied_probability
                ),
                "Edge": (
                    value_bet.edge
                ),
                "Expected value": (
                    value_bet
                    .expected_value
                ),
                "Assessment": (
                    value_bet
                    .confidence_label
                    if is_value
                    else "AVOID"
                ),
            }
        )

    analysis_dataframe = pd.DataFrame(
        analysis_rows
    )

    analysis_dataframe = (
        analysis_dataframe.sort_values(
            by="Expected value",
            ascending=False,
        )
    )

    st.dataframe(
        analysis_dataframe.style.format(
            {
                "Odds": "{:.2f}",
                (
                    "Model probability"
                ): "{:.1%}",
                (
                    "Market probability"
                ): "{:.1%}",
                "Edge": "{:+.1%}",
                (
                    "Expected value"
                ): "{:+.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "## Suggested coupon"
    )

    if (
        not report.has_value_bet
        or report.best_opportunity is None
    ):
        st.warning(
            "No positive-value selection passed "
            "the current safety filters. The model "
            "does not recommend forcing a coupon "
            "for this match."
        )

        return

    best_bet = report.best_opportunity

    st.success(
        f"### ✅ {best_bet.selection}\n\n"
        f"**Market:** "
        f"{best_bet.market_name}\n\n"
        f"**Bookmaker:** "
        f"{best_bet.bookmaker}\n\n"
        f"**Decimal odds:** "
        f"{best_bet.decimal_odds:.2f}\n\n"
        f"**Model probability:** "
        f"{best_bet.model_probability:.1%}\n\n"
        f"**Market probability:** "
        f"{best_bet.implied_probability:.1%}\n\n"
        f"**Model edge:** "
        f"{best_bet.edge:+.1%}\n\n"
        f"**Estimated expected value:** "
        f"{best_bet.expected_value:+.1%}\n\n"
        f"**Signal:** "
        f"{best_bet.confidence_label}"
    )

    st.info(
        "The suggested coupon contains one "
        "selection. Combining correlated markets "
        "from the same match would require a "
        "joint-probability model; therefore the "
        "application does not multiply these "
        "probabilities as though they were "
        "independent."
    )

    alternatives = (
        report.opportunities[1:4]
    )

    if alternatives:
        st.markdown(
            "### Alternative value selections"
        )

        alternative_rows = []

        for opportunity in alternatives:
            alternative_rows.append(
                {
                    "Selection": (
                        opportunity.selection
                    ),
                    "Odds": (
                        opportunity
                        .decimal_odds
                    ),
                    "Bookmaker": (
                        opportunity
                        .bookmaker
                    ),
                    "Edge": (
                        opportunity.edge
                    ),
                    "Expected value": (
                        opportunity
                        .expected_value
                    ),
                }
            )

        alternative_dataframe = (
            pd.DataFrame(
                alternative_rows
            )
        )

        st.dataframe(
            (
                alternative_dataframe
                .style.format(
                    {
                        "Odds": "{:.2f}",
                        "Edge": "{:+.1%}",
                        (
                            "Expected value"
                        ): "{:+.1%}",
                    }
                )
            ),
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(
        page_title=(
            "Football Value Predictor"
        ),
        page_icon="⚽",
        layout="wide",
    )

    st.title(
        "⚽ Football Value Predictor"
    )

    st.write(
        "Select an active league and an upcoming "
        "match to compare model probabilities "
        "with live bookmaker prices."
    )

    api_key = get_api_key()

    if api_key is None:
        st.error(
            "ODDS_API_KEY could not be found. "
            "Add it to "
            ".streamlit/secrets.toml."
        )

        st.stop()

    try:
        prediction_service = (
            load_prediction_service()
        )

        betting_service = (
            load_betting_insight_service()
        )

        value_service = (
            load_value_bet_service()
        )

        with st.spinner(
            "Loading active football leagues..."
        ):
            active_leagues = (
                get_active_leagues(
                    api_key
                )
            )

    except Exception as error:
        st.error(
            "The application could not start: "
            f"{error}"
        )

        st.stop()

    if not active_leagues:
        st.warning(
            "No active football leagues were "
            "returned by the odds provider."
        )

        st.stop()

    league_keys = list(
        active_leagues.keys()
    )

    selected_sport_key = st.selectbox(
        "League",
        options=league_keys,
        format_func=lambda key: (
            active_leagues[key]
        ),
    )

    try:
        with st.spinner(
            "Loading upcoming matches..."
        ):
            league_events = (
                get_league_events(
                    api_key=api_key,
                    sport_key=(
                        selected_sport_key
                    ),
                )
            )

    except OddsAPIError as error:
        st.error(
            f"Could not load matches: {error}"
        )

        st.stop()

    if not league_events:
        st.info(
            "There are currently no upcoming "
            "matches listed for this league. "
            "Select another league."
        )

        st.stop()

    event_indexes = list(
        range(
            len(league_events)
        )
    )

    selected_event_index = st.selectbox(
        "Upcoming match",
        options=event_indexes,
        format_func=lambda index: (
            f"{league_events[index].home_team} "
            f"vs "
            f"{league_events[index].away_team} "
            f"— "
            f"{format_event_time(league_events[index].commence_time)}"
        ),
    )

    selected_event = (
        league_events[
            selected_event_index
        ]
    )

    st.caption(
        f"Competition: "
        f"{active_leagues[selected_sport_key]} "
        f"| Kick-off: "
        f"{format_event_time(selected_event.commence_time)}"
    )

    analyse_button = st.button(
        "Analyse Match",
        type="primary",
        width="stretch",
    )

    if not analyse_button:
        return

    model_teams = (
        prediction_service.get_teams()
    )

    model_home_team, home_score = (
        find_model_team(
            api_team_name=(
                selected_event.home_team
            ),
            model_teams=model_teams,
        )
    )

    model_away_team, away_score = (
        find_model_team(
            api_team_name=(
                selected_event.away_team
            ),
            model_teams=model_teams,
        )
    )

    if (
        model_home_team is None
        or model_away_team is None
        or home_score < 0.60
        or away_score < 0.60
    ):
        st.error(
            "The selected API teams could not "
            "be matched reliably with teams in "
            "the model dataset."
        )

        st.write(
            {
                "API home team": (
                    selected_event.home_team
                ),
                "Best model home match": (
                    model_home_team
                ),
                "Home similarity": (
                    round(
                        home_score,
                        3,
                    )
                ),
                "API away team": (
                    selected_event.away_team
                ),
                "Best model away match": (
                    model_away_team
                ),
                "Away similarity": (
                    round(
                        away_score,
                        3,
                    )
                ),
            }
        )

        st.stop()

    if model_home_team == model_away_team:
        st.error(
            "Both API teams were matched to the "
            "same model team. The prediction was "
            "cancelled."
        )

        st.stop()

    try:
        with st.spinner(
            "Running model and loading live odds..."
        ):
            prediction = (
                prediction_service.predict(
                    home_team=(
                        model_home_team
                    ),
                    away_team=(
                        model_away_team
                    ),
                )
            )

            match_odds = (
                get_live_event_odds(
                    api_key=api_key,
                    event_id=(
                        selected_event.event_id
                    ),
                    sport_key=(
                        selected_event.sport_key
                    ),
                    sport_title=(
                        selected_event.sport_title
                    ),
                    commence_time=(
                        selected_event
                        .commence_time
                    ),
                    home_team=(
                        selected_event.home_team
                    ),
                    away_team=(
                        selected_event.away_team
                    ),
                )
            )

    except Exception as error:
        st.error(
            f"Match analysis failed: {error}"
        )

        st.stop()

    st.header(
        f"{selected_event.home_team} "
        f"vs "
        f"{selected_event.away_team}"
    )

    with st.expander(
        "Team-name matching details"
    ):
        st.write(
            {
                (
                    selected_event.home_team
                ): {
                    "Model team": (
                        model_home_team
                    ),
                    "Similarity": (
                        round(
                            home_score,
                            3,
                        )
                    ),
                },
                (
                    selected_event.away_team
                ): {
                    "Model team": (
                        model_away_team
                    ),
                    "Similarity": (
                        round(
                            away_score,
                            3,
                        )
                    ),
                },
            }
        )

    display_match_prediction(
        prediction=prediction,
        api_event=selected_event,
    )

    st.divider()

    display_goal_markets(
        prediction
    )

    st.divider()

    display_double_chance(
        prediction=prediction,
        api_event=selected_event,
    )

    st.divider()

    display_model_signal(
        prediction=prediction,
        betting_service=(
            betting_service
        ),
    )

    st.divider()

    display_live_odds(
        match_odds
    )

    st.divider()

    display_value_analysis(
        prediction=prediction,
        match_odds=match_odds,
        value_service=value_service,
    )

    st.divider()

    st.warning(
        "This application is an experimental "
        "data-science project. Model estimates "
        "can be wrong, bookmaker prices can "
        "change, and positive expected value is "
        "not a guarantee of profit."
    )


if __name__ == "__main__":
    main()
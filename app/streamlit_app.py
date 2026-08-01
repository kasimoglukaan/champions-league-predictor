from __future__ import annotations

import html
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


from src.config import (
    PREDICTION_HISTORY_DATABASE,
    PRODUCTION_DATASET,
    PRODUCTION_MODEL,
)
from src.models.bookmaker_odds import (
    MatchOdds,
    OddsEvent,
)
from src.repositories.prediction_history_repository import (
    PredictionHistoryRepository,
)
from src.services.ml_prediction_service import (
    MLPredictionService,
)
from src.services.prediction_history_service import (
    PredictionHistoryService,
)
from src.services.odds_api_service import (
    OddsAPIError,
    OddsAPIService,
)
from src.services.team_matching_service import (
    TeamMatchResult,
    TeamMatchingService,
)
from src.services.value_bet_service import (
    ValueBetService,
)


DATA_PATH = PRODUCTION_DATASET
MODEL_PATH = PRODUCTION_MODEL


PAGE_CSS = """
<style>
    :root {
        --background: #07110e;
        --surface: #0d1c17;
        --surface-light: #12251e;
        --border: rgba(255, 255, 255, 0.08);
        --primary: #38df8f;
        --primary-light: #71efb2;
        --text: #f5f7f6;
        --muted: #91a69e;
        --danger: #f47c7c;
        --warning: #f6d365;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(31, 125, 88, 0.14),
                transparent 34%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(16, 79, 56, 0.12),
                transparent 28%
            ),
            var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 14, 0.88);
        backdrop-filter: blur(12px);
    }

    [data-testid="stSidebar"] {
        background: #081711;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
    }

    .brand {
        margin-bottom: 2rem;
    }

    .brand-name {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .brand-accent {
        color: var(--primary);
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.45;
        margin-top: 0.3rem;
    }

    .sidebar-section {
        color: #ffffff;
        font-size: 0.95rem;
        font-weight: 800;
        margin-top: 0.7rem;
        margin-bottom: 0.8rem;
    }

    .match-hero {
        background:
            linear-gradient(
                120deg,
                rgba(16, 70, 51, 0.98),
                rgba(10, 37, 29, 0.98)
            );
        border: 1px solid rgba(56, 223, 143, 0.22);
        border-radius: 22px;
        padding: 27px 30px;
        margin-bottom: 20px;
        box-shadow: 0 22px 55px rgba(0, 0, 0, 0.25);
    }

    .competition-row {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 14px;
    }

    .competition-name {
        color: var(--primary);
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .upcoming-pill {
        color: var(--primary-light);
        background: rgba(56, 223, 143, 0.10);
        border: 1px solid rgba(56, 223, 143, 0.27);
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.68rem;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .match-title {
        color: #ffffff;
        font-size: clamp(1.8rem, 3vw, 2.65rem);
        font-weight: 900;
        letter-spacing: -0.045em;
        line-height: 1.1;
    }

    .versus {
        color: #7f968d;
        font-weight: 600;
        padding: 0 8px;
    }

    .match-meta {
        color: #afc0ba;
        font-size: 0.87rem;
        margin-top: 13px;
    }

    .empty-state {
        background: rgba(13, 28, 23, 0.96);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 38px 30px;
        text-align: center;
        margin-top: 16px;
    }

    .empty-icon {
        font-size: 2rem;
        margin-bottom: 12px;
    }

    .empty-title {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 850;
    }

    .empty-text {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: 7px;
    }

    .section-title {
        color: #ffffff;
        font-size: 1.22rem;
        font-weight: 850;
        letter-spacing: -0.02em;
        margin-top: 10px;
        margin-bottom: 14px;
    }

    .section-kicker {
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 850;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-top: 22px;
        margin-bottom: 9px;
    }

    .section-description {
        color: #8ea39b;
        font-size: 0.82rem;
        line-height: 1.5;
        margin-top: -4px;
        margin-bottom: 15px;
    }

    .best-bet-card {
        background:
            linear-gradient(
                135deg,
                rgba(25, 100, 72, 0.98),
                rgba(13, 52, 38, 0.98)
            );
        border: 1px solid rgba(56, 223, 143, 0.28);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.23);
    }

    .card-label {
        color: #9bb0a8;
        font-size: 0.69rem;
        font-weight: 850;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    .bet-selection {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 900;
        letter-spacing: -0.035em;
        margin-top: 9px;
        line-height: 1.12;
    }

    .bet-market {
        color: #b7c8c1;
        font-size: 0.82rem;
        margin-top: 6px;
    }

    .value-explanation {
        background: rgba(3, 27, 18, 0.24);
        border-left: 3px solid var(--primary);
        color: #b7cac2;
        font-size: 0.79rem;
        line-height: 1.45;
        padding: 10px 12px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }

    .odds-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 18px;
        margin-bottom: 21px;
    }

    .odds-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 72px;
        background: var(--primary);
        color: #052116;
        font-size: 1.3rem;
        font-weight: 950;
        padding: 8px 15px;
        border-radius: 11px;
    }

    .confidence-badge {
        display: inline-flex;
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 850;
        letter-spacing: 0.07em;
    }

    .confidence-strong {
        color: #7af2b7;
        background: rgba(56, 223, 143, 0.11);
        border: 1px solid rgba(56, 223, 143, 0.28);
    }

    .confidence-moderate {
        color: #f6dc77;
        background: rgba(246, 211, 101, 0.10);
        border: 1px solid rgba(246, 211, 101, 0.23);
    }

    .confidence-small {
        color: #9dc7ff;
        background: rgba(99, 164, 255, 0.10);
        border: 1px solid rgba(99, 164, 255, 0.23);
    }

    .bet-stat-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
    }

    .bet-stat {
        background: rgba(4, 28, 19, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 13px;
        padding: 12px;
    }

    .bet-stat-value {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 850;
        margin-top: 4px;
    }

    .positive {
        color: var(--primary-light);
    }

    .bookmaker-note {
        color: #aac0b7;
        font-size: 0.79rem;
        margin-top: 16px;
    }

    .no-value-card {
        background: rgba(63, 28, 27, 0.34);
        border: 1px solid rgba(244, 124, 124, 0.18);
        border-radius: 20px;
        padding: 25px;
    }

    .no-value-title {
        color: #ffd0d0;
        font-size: 1.25rem;
        font-weight: 850;
    }

    .no-value-text {
        color: #d9aaaa;
        font-size: 0.84rem;
        margin-top: 8px;
    }

    .probability-card {
        background: rgba(13, 28, 23, 0.97);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 20px;
        min-height: 130px;
    }

    .probability-value {
        color: #ffffff;
        font-size: 1.72rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin-top: 7px;
    }

    .probability-team {
        color: #a5b7b0;
        font-size: 0.82rem;
        margin-top: 5px;
    }

    .alternative-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 18px;
        background: rgba(13, 28, 23, 0.75);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 9px;
    }

    .alternative-selection {
        color: #ffffff;
        font-weight: 800;
    }

    .alternative-meta {
        color: var(--muted);
        font-size: 0.76rem;
        margin-top: 3px;
    }

    .alternative-odds {
        color: #ffffff;
        font-size: 1.08rem;
        font-weight: 900;
        text-align: right;
    }

    .alternative-edge {
        color: var(--primary-light);
        font-size: 0.76rem;
        font-weight: 800;
        text-align: right;
        margin-top: 2px;
    }

    .matching-alert {
        background: rgba(74, 35, 31, 0.50);
        border: 1px solid rgba(244, 124, 124, 0.22);
        border-radius: 18px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 18px;
    }

    .matching-alert-title {
        color: #ffd1d1;
        font-size: 1.15rem;
        font-weight: 850;
    }

    .matching-alert-text {
        color: #d5aaaa;
        font-size: 0.84rem;
        line-height: 1.5;
        margin-top: 8px;
    }

    .matching-success {
        color: var(--primary-light);
        font-weight: 800;
    }

    .matching-rejected {
        color: #ff9b9b;
        font-weight: 800;
    }

    div[data-testid="stMetric"] {
        background: rgba(13, 28, 23, 0.97);
        border: 1px solid var(--border);
        padding: 17px;
        border-radius: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 850;
    }

    div[data-baseweb="select"] > div {
        background: #10231c;
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
    }

    .stButton > button {
        background:
            linear-gradient(
                135deg,
                #40e597,
                #29c878
            );
        color: #052116;
        border: none;
        border-radius: 12px;
        font-weight: 900;
        min-height: 47px;
        box-shadow:
            0 11px 26px
            rgba(35, 201, 117, 0.17);
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #6aefad,
                #39d98a
            );
        color: #041c12;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 15px;
        overflow: hidden;
    }

    button[data-baseweb="tab"] {
        color: #8da199;
        font-weight: 750;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary);
    }

    hr {
        border-color: var(--border);
    }

    .sidebar-note {
        color: #70867d;
        font-size: 0.72rem;
        line-height: 1.5;
        text-align: center;
        margin-top: 24px;
    }

    .footer-note {
        color: #70857d;
        font-size: 0.74rem;
        line-height: 1.5;
        text-align: center;
        margin-top: 32px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 800px) {
        .bet-stat-grid {
            grid-template-columns: 1fr;
        }

        .match-title {
            font-size: 1.75rem;
        }
    }
</style>
"""


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
def load_value_service(
) -> ValueBetService:
    return ValueBetService(
        minimum_model_probability=0.45,
        minimum_edge=0.03,
        minimum_expected_value=0.03,
        maximum_decimal_odds=8.00,
        maximum_bookmaker_margin=0.15,
    )


@st.cache_resource
def load_team_matching_service(
) -> TeamMatchingService:
    return TeamMatchingService(
        minimum_score=0.84,
        minimum_margin=0.12,
        candidate_limit=5,
    )


@st.cache_resource
def load_prediction_history_service(
) -> PredictionHistoryService:
    repository = PredictionHistoryRepository(
        PREDICTION_HISTORY_DATABASE
    )

    return PredictionHistoryService(
        repository=repository
    )


def hybrid_result_probabilities(
    prediction,
) -> tuple[float, float, float]:
    home_probability = (
        float(prediction.home_win_probability)
        + float(prediction.poisson_home_probability)
    ) / 2.0

    draw_probability = (
        float(prediction.draw_probability)
        + float(prediction.poisson_draw_probability)
    ) / 2.0

    away_probability = (
        float(prediction.away_win_probability)
        + float(prediction.poisson_away_probability)
    ) / 2.0

    total = (
        home_probability
        + draw_probability
        + away_probability
    )

    if total <= 0:
        return (
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        )

    return (
        home_probability / total,
        draw_probability / total,
        away_probability / total,
    )


def get_api_key(
) -> Optional[str]:
    try:
        api_key = str(
            st.secrets[
                "ODDS_API_KEY"
            ]
        ).strip()

    except (
        KeyError,
        FileNotFoundError,
    ):
        return None

    return api_key or None


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def get_active_leagues(
    api_key: str,
) -> dict[str, str]:
    service = OddsAPIService(
        api_key=api_key,
        region="eu",
    )

    return (
        service
        .get_active_soccer_leagues()
    )


@st.cache_data(
    ttl=600,
    show_spinner=False,
)
def get_league_events(
    api_key: str,
    sport_key: str,
) -> list[OddsEvent]:
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


def safe_text(
    value: object,
) -> str:
    return html.escape(
        str(value)
    )


def format_event_time(
    commence_time: str,
) -> str:
    if not commence_time:
        return "Time unavailable"

    try:
        parsed_time = (
            datetime.fromisoformat(
                commence_time.replace(
                    "Z",
                    "+00:00",
                )
            )
        )

        return parsed_time.strftime(
            "%d %B %Y · %H:%M UTC"
        )

    except ValueError:
        return commence_time


def confidence_class(
    confidence_label: str,
) -> str:
    normalized = (
        confidence_label.upper()
    )

    if normalized == "STRONG":
        return "confidence-strong"

    if normalized == "MODERATE":
        return "confidence-moderate"

    return "confidence-small"


def render_brand(
) -> None:
    st.sidebar.markdown(
        (
            '<div class="brand">'
            '<div class="brand-name">'
            'Edge'
            '<span class="brand-accent">'
            'XI'
            '</span>'
            '</div>'
            '<div class="brand-subtitle">'
            'Football analytics and '
            'market intelligence'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_match_hero(
    event: OddsEvent,
    league_name: str,
) -> None:
    home_team = safe_text(
        event.home_team
    )

    away_team = safe_text(
        event.away_team
    )

    competition = safe_text(
        league_name
    )

    kick_off = safe_text(
        format_event_time(
            event.commence_time
        )
    )

    markup = (
        '<div class="match-hero">'
        '<div class="competition-row">'
        '<div class="competition-name">'
        f'{competition}'
        '</div>'
        '<div class="upcoming-pill">'
        'Upcoming'
        '</div>'
        '</div>'
        '<div class="match-title">'
        f'{home_team}'
        '<span class="versus">'
        'vs'
        '</span>'
        f'{away_team}'
        '</div>'
        '<div class="match-meta">'
        f'Kick-off: {kick_off}'
        ' &nbsp;·&nbsp; '
        'Live bookmaker market comparison'
        '</div>'
        '</div>'
    )

    st.markdown(
        markup,
        unsafe_allow_html=True,
    )


def render_empty_state(
) -> None:
    st.markdown(
        (
            '<div class="empty-state">'
            '<div class="empty-icon">'
            '📊'
            '</div>'
            '<div class="empty-title">'
            'Match analysis is ready'
            '</div>'
            '<div class="empty-text">'
            'Select a fixture and press '
            'Run Match Analysis to view '
            'model probabilities, live odds '
            'and value selections.'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_best_bet(
    report,
) -> None:
    if (
        not report.has_value_bet
        or report.best_opportunity
        is None
    ):
        st.markdown(
            (
                '<div class="no-value-card">'
                '<div class="no-value-title">'
                'No qualifying live-odds '
                'value selection'
                '</div>'
                '<div class="no-value-text">'
                'The available bookmaker prices '
                'do not currently pass the model '
                'probability, edge and expected-value '
                'requirements. Avoid forcing a selection.'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        return

    bet = report.best_opportunity

    selection = safe_text(
        bet.selection
    )

    market = safe_text(
        bet.market_name
    )

    bookmaker = safe_text(
        bet.bookmaker
    )

    confidence = safe_text(
        bet.confidence_label
    )

    badge_class = confidence_class(
        bet.confidence_label
    )

    markup = (
        '<div class="best-bet-card">'
        '<div class="card-label">'
        'Live odds value pick'
        '</div>'
        '<div class="bet-selection">'
        f'{selection}'
        '</div>'
        '<div class="bet-market">'
        f'{market}'
        '</div>'
        '<div class="value-explanation">'
        'This selection combines the model '
        'probability with current bookmaker '
        'odds. It is not the model prediction '
        'alone.'
        '</div>'
        '<div class="odds-row">'
        '<div class="odds-badge">'
        f'{bet.decimal_odds:.2f}'
        '</div>'
        '<div class="confidence-badge '
        f'{badge_class}">'
        f'{confidence}'
        '</div>'
        '</div>'
        '<div class="bet-stat-grid">'
        '<div class="bet-stat">'
        '<div class="card-label">'
        'Model probability'
        '</div>'
        '<div class="bet-stat-value">'
        f'{bet.model_probability:.1%}'
        '</div>'
        '</div>'
        '<div class="bet-stat">'
        '<div class="card-label">'
        'Model edge'
        '</div>'
        '<div class="bet-stat-value positive">'
        f'{bet.edge:+.1%}'
        '</div>'
        '</div>'
        '<div class="bet-stat">'
        '<div class="card-label">'
        'Expected value'
        '</div>'
        '<div class="bet-stat-value positive">'
        f'{bet.expected_value:+.1%}'
        '</div>'
        '</div>'
        '</div>'
        '<div class="bookmaker-note">'
        'Best available price: '
        f'<strong>{bookmaker}</strong>'
        '</div>'
        '</div>'
    )

    st.markdown(
        markup,
        unsafe_allow_html=True,
    )


def render_probability_cards(
    prediction,
    event: OddsEvent,
) -> None:
    (
        home_probability,
        draw_probability,
        away_probability,
    ) = hybrid_result_probabilities(
        prediction
    )

    cards = [
        (
            "Home win",
            event.home_team,
            home_probability,
        ),
        (
            "Draw",
            "Match draw",
            draw_probability,
        ),
        (
            "Away win",
            event.away_team,
            away_probability,
        ),
    ]

    columns = st.columns(3)

    for (
        column,
        (
            label,
            team,
            probability,
        ),
    ) in zip(
        columns,
        cards,
    ):
        with column:
            st.markdown(
                (
                    '<div class="probability-card">'
                    '<div class="card-label">'
                    f'{safe_text(label)}'
                    '</div>'
                    '<div class="probability-value">'
                    f'{probability:.1%}'
                    '</div>'
                    '<div class="probability-team">'
                    f'{safe_text(team)}'
                    '</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )


def render_alternatives(
    report,
) -> None:
    alternatives = (
        report.opportunities[1:5]
    )

    if not alternatives:
        st.info(
            "No additional live-odds value "
            "selections passed the filters."
        )

        return

    st.markdown(
        (
            '<div class="section-title">'
            'Alternative live-odds value picks'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    for bet in alternatives:
        st.markdown(
            (
                '<div class="alternative-row">'
                '<div>'
                '<div class="alternative-selection">'
                f'{safe_text(bet.selection)}'
                '</div>'
                '<div class="alternative-meta">'
                f'{safe_text(bet.market_name)}'
                ' · '
                f'{safe_text(bet.bookmaker)}'
                '</div>'
                '</div>'
                '<div>'
                '<div class="alternative-odds">'
                f'{bet.decimal_odds:.2f}'
                '</div>'
                '<div class="alternative-edge">'
                f'Edge {bet.edge:+.1%}'
                '</div>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


def create_market_dataframe(
    prediction,
    event: OddsEvent,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Market": "Match result",
                "Selection": event.home_team,
                "Probability": (
                    hybrid_result_probabilities(
                        prediction
                    )[0]
                ),
            },
            {
                "Market": "Match result",
                "Selection": "Draw",
                "Probability": (
                    hybrid_result_probabilities(
                        prediction
                    )[1]
                ),
            },
            {
                "Market": "Match result",
                "Selection": event.away_team,
                "Probability": (
                    hybrid_result_probabilities(
                        prediction
                    )[2]
                ),
            },
            {
                "Market": (
                    "Both teams to score"
                ),
                "Selection": "Yes",
                "Probability": (
                    prediction
                    .btts_yes_probability
                ),
            },
            {
                "Market": (
                    "Both teams to score"
                ),
                "Selection": "No",
                "Probability": (
                    prediction
                    .btts_no_probability
                ),
            },
            {
                "Market": "Total goals",
                "Selection": "Over 1.5",
                "Probability": (
                    prediction
                    .over_1_5_probability
                ),
            },
            {
                "Market": "Total goals",
                "Selection": "Over 2.5",
                "Probability": (
                    prediction
                    .over_2_5_probability
                ),
            },
            {
                "Market": "Total goals",
                "Selection": "Under 2.5",
                "Probability": (
                    prediction
                    .under_2_5_probability
                ),
            },
            {
                "Market": "Total goals",
                "Selection": "Under 3.5",
                "Probability": (
                    prediction
                    .under_3_5_probability
                ),
            },
        ]
    )


def create_best_odds_dataframe(
    match_odds: MatchOdds,
) -> pd.DataFrame:
    rows = []

    result_selections = [
        match_odds.home_team,
        "Draw",
        match_odds.away_team,
    ]

    for selection in result_selections:
        price = match_odds.get_best_price(
            market_key="h2h",
            selection=selection,
        )

        if price is None:
            continue

        rows.append(
            {
                "Market": "Match result",
                "Selection": selection,
                "Best odds": price.odds,
                "Bookmaker": (
                    price.bookmaker_title
                ),
                "Implied probability": (
                    price
                    .implied_probability
                ),
            }
        )

    totals = [
        (
            "Over",
            1.5,
            "Over 1.5",
        ),
        (
            "Under",
            1.5,
            "Under 1.5",
        ),
        (
            "Over",
            2.5,
            "Over 2.5",
        ),
        (
            "Under",
            2.5,
            "Under 2.5",
        ),
        (
            "Over",
            3.5,
            "Over 3.5",
        ),
        (
            "Under",
            3.5,
            "Under 3.5",
        ),
    ]

    for (
        selection,
        point,
        display_name,
    ) in totals:
        price = match_odds.get_best_price(
            market_key="totals",
            selection=selection,
            point=point,
        )

        if price is None:
            continue

        rows.append(
            {
                "Market": "Total goals",
                "Selection": display_name,
                "Best odds": price.odds,
                "Bookmaker": (
                    price.bookmaker_title
                ),
                "Implied probability": (
                    price
                    .implied_probability
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def create_value_dataframe(
    report,
) -> pd.DataFrame:
    rows = []

    candidates = (
        report.opportunities
        + report.rejected_markets
    )

    for bet in candidates:
        rating = (
            bet.confidence_label
            if bet in report.opportunities
            else "NO VALUE"
        )

        rows.append(
            {
                "Market": (
                    bet.market_name
                ),
                "Selection": (
                    bet.selection
                ),
                "Odds": (
                    bet.decimal_odds
                ),
                "Bookmaker": (
                    bet.bookmaker
                ),
                "Model probability": (
                    bet.model_probability
                ),
                "Market probability": (
                    bet
                    .fair_market_probability
                ),
                "Edge": (
                    bet.edge
                ),
                "Expected value": (
                    bet.expected_value
                ),
                "Rating": (
                    rating
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            by="Expected value",
            ascending=False,
        )
    )


def create_matching_candidate_dataframe(
    result: TeamMatchResult,
) -> pd.DataFrame:
    rows = []

    for candidate in result.candidates:
        rows.append(
            {
                "Candidate": (
                    candidate.team_name
                ),
                "Score": (
                    candidate.score
                ),
                "Distinctive overlap": (
                    ", ".join(
                        candidate
                        .distinctive_overlap
                    )
                    or "None"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def render_rejected_team_matching(
    home_match: TeamMatchResult,
    away_match: TeamMatchResult,
    model_teams: list[str],
    event_id: str,
) -> Optional[tuple[str, str]]:
    st.markdown(
        (
            '<div class="matching-alert">'
            '<div class="matching-alert-title">'
            'Unsafe team matching prevented'
            '</div>'
            '<div class="matching-alert-text">'
            'The automatic matcher could not confirm one or both '
            'teams safely. Review the candidates below and select '
            'the correct model teams manually to continue.'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    home_column, away_column = (
        st.columns(2)
    )

    for (
        column,
        title,
        match_result,
    ) in [
        (
            home_column,
            "Home-team matching",
            home_match,
        ),
        (
            away_column,
            "Away-team matching",
            away_match,
        ),
    ]:
        with column:
            st.markdown(
                f"### {title}"
            )

            if match_result.accepted:
                st.success(
                    "Safe match found: "
                    f"{match_result.matched_team_name}"
                )

            else:
                st.error(
                    match_result.reason
                )

            summary_frame = pd.DataFrame(
                [
                    {
                        "API team": (
                            match_result
                            .api_team_name
                        ),
                        "Selected model team": (
                            match_result
                            .matched_team_name
                            or "Rejected"
                        ),
                        "Best score": (
                            match_result.score
                        ),
                        "Second-best score": (
                            match_result
                            .second_best_score
                        ),
                        "Score margin": (
                            match_result
                            .score_margin
                        ),
                    }
                ]
            )

            st.dataframe(
                summary_frame.style.format(
                    {
                        "Best score": (
                            "{:.1%}"
                        ),
                        "Second-best score": (
                            "{:.1%}"
                        ),
                        "Score margin": (
                            "{:.1%}"
                        ),
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            candidate_frame = (
                create_matching_candidate_dataframe(
                    match_result
                )
            )

            if not candidate_frame.empty:
                st.caption(
                    "Highest-scoring candidates"
                )

                st.dataframe(
                    candidate_frame.style.format(
                        {
                            "Score": "{:.1%}",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

    st.markdown(
        (
            '<div class="section-title">'
            'Manual team selection'
            '</div>'
            '<div class="section-description">'
            'Only use this override when you are certain which model '
            'team corresponds to the bookmaker team. The selected '
            'teams will be used for this analysis only.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    def selection_options(
        match_result: TeamMatchResult,
    ) -> list[str]:
        preferred = []

        if (
            match_result.accepted
            and match_result.matched_team_name
        ):
            preferred.append(
                match_result.matched_team_name
            )

        preferred.extend(
            candidate.team_name
            for candidate in match_result.candidates
        )

        ordered = []
        seen = set()

        for team_name in [
            *preferred,
            *model_teams,
        ]:
            cleaned = str(team_name).strip()

            if (
                cleaned
                and cleaned not in seen
            ):
                ordered.append(cleaned)
                seen.add(cleaned)

        return ordered

    home_options = selection_options(
        home_match
    )

    away_options = selection_options(
        away_match
    )

    if not home_options or not away_options:
        st.error(
            "The model team list is empty, so a manual "
            "selection cannot be made."
        )

        return None

    selection_columns = st.columns(2)

    with selection_columns[0]:
        manual_home_team = st.selectbox(
            f"Model team for {home_match.api_team_name}",
            options=home_options,
            key=(
                "manual_home_team_"
                f"{event_id}"
            ),
            help=(
                "The highest-scoring candidates appear first. "
                "You can also search the full model team list."
            ),
        )

    with selection_columns[1]:
        manual_away_team = st.selectbox(
            f"Model team for {away_match.api_team_name}",
            options=away_options,
            key=(
                "manual_away_team_"
                f"{event_id}"
            ),
            help=(
                "The highest-scoring candidates appear first. "
                "You can also search the full model team list."
            ),
        )

    confirmation = st.checkbox(
        (
            "I have checked both teams and confirm that "
            "these manual selections are correct."
        ),
        key=(
            "manual_team_confirmation_"
            f"{event_id}"
        ),
    )

    continue_button = st.button(
        "Use Selected Teams and Continue Analysis",
        type="primary",
        width="stretch",
        disabled=not confirmation,
        key=(
            "manual_team_continue_"
            f"{event_id}"
        ),
    )

    if not continue_button:
        return None

    if manual_home_team == manual_away_team:
        st.error(
            "Home and away teams cannot use the same model team."
        )

        return None

    return (
        manual_home_team,
        manual_away_team,
    )


def render_model_markets(
    prediction,
    event: OddsEvent,
) -> None:
    st.markdown(
        (
            '<div class="section-title">'
            'Model match outlook'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="section-description">'
            'The 1X2 probabilities combine '
            '50% machine learning and 50% Poisson. '
            'Goal markets come from Poisson simulation. '
            'Bookmaker odds are not used here.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    (
        hybrid_home_probability,
        hybrid_draw_probability,
        hybrid_away_probability,
    ) = hybrid_result_probabilities(
        prediction
    )

    result_frame = pd.DataFrame(
        {
            "Selection": [
                event.home_team,
                "Draw",
                event.away_team,
            ],
            "Probability": [
                hybrid_home_probability,
                hybrid_draw_probability,
                hybrid_away_probability,
            ],
        }
    )

    st.bar_chart(
        result_frame,
        x="Selection",
        y="Probability",
    )

    home_or_draw = (
        hybrid_home_probability
        + hybrid_draw_probability
    )

    away_or_draw = (
        hybrid_away_probability
        + hybrid_draw_probability
    )

    metric_columns = (
        st.columns(4)
    )

    metric_columns[0].metric(
        "Likely score",
        (
            f"{prediction.most_likely_home_goals}"
            "-"
            f"{prediction.most_likely_away_goals}"
        ),
    )

    metric_columns[1].metric(
        "Expected goals",
        (
            f"{prediction.expected_home_goals:.2f}"
            " – "
            f"{prediction.expected_away_goals:.2f}"
        ),
    )

    metric_columns[2].metric(
        "1X model probability",
        f"{home_or_draw:.1%}",
    )

    metric_columns[3].metric(
        "X2 model probability",
        f"{away_or_draw:.1%}",
    )

    st.markdown(
        (
            '<div class="section-title">'
            'All model market probabilities'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    market_frame = (
        create_market_dataframe(
            prediction,
            event,
        )
    )

    st.dataframe(
        market_frame.style.format(
            {
                "Probability": (
                    "{:.1%}"
                ),
            }
        ),
        width="stretch",
        hide_index=True,
    )


def render_live_odds_analysis(
    match_odds: MatchOdds,
    report,
) -> None:
    st.markdown(
        (
            '<div class="section-title">'
            'Best live bookmaker prices'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="section-description">'
            'These prices come from the connected '
            'odds provider and are not generated '
            'by the prediction model.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    odds_frame = (
        create_best_odds_dataframe(
            match_odds
        )
    )

    if odds_frame.empty:
        st.info(
            "No supported bookmaker "
            "odds were returned."
        )

    else:
        st.dataframe(
            odds_frame.style.format(
                {
                    "Best odds": (
                        "{:.2f}"
                    ),
                    (
                        "Implied probability"
                    ): "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        (
            '<div class="section-title">'
            'Model and bookmaker comparison'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="section-description">'
            'This table compares model probabilities '
            'with bookmaker-market probabilities to '
            'calculate edge and expected value.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    value_frame = (
        create_value_dataframe(
            report
        )
    )

    if value_frame.empty:
        st.info(
            "There is not enough data "
            "for a value comparison."
        )

    else:
        st.dataframe(
            value_frame.style.format(
                {
                    "Odds": (
                        "{:.2f}"
                    ),
                    (
                        "Model probability"
                    ): "{:.1%}",
                    (
                        "Market probability"
                    ): "{:.1%}",
                    "Edge": (
                        "{:+.1%}"
                    ),
                    (
                        "Expected value"
                    ): "{:+.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )


def render_model_details(
    prediction,
    selected_event: OddsEvent,
    home_match: TeamMatchResult,
    away_match: TeamMatchResult,
    match_odds: MatchOdds,
) -> None:
    st.markdown(
        (
            '<div class="section-title">'
            'Prediction diagnostics'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    diagnostic_columns = (
        st.columns(4)
    )

    diagnostic_columns[0].metric(
        "Home xG estimate",
        (
            f"{prediction.expected_home_goals:.2f}"
        ),
    )

    diagnostic_columns[1].metric(
        "Away xG estimate",
        (
            f"{prediction.expected_away_goals:.2f}"
        ),
    )

    diagnostic_columns[2].metric(
        "Home team match",
        f"{home_match.score:.1%}",
    )

    diagnostic_columns[3].metric(
        "Away team match",
        f"{away_match.score:.1%}",
    )

    matching_frame = pd.DataFrame(
        [
            {
                "API team": (
                    selected_event
                    .home_team
                ),
                "Model team": (
                    home_match
                    .matched_team_name
                ),
                "Similarity": (
                    home_match.score
                ),
                "Margin over second": (
                    home_match
                    .score_margin
                ),
                "Matching method": (
                    home_match.reason
                ),
            },
            {
                "API team": (
                    selected_event
                    .away_team
                ),
                "Model team": (
                    away_match
                    .matched_team_name
                ),
                "Similarity": (
                    away_match.score
                ),
                "Margin over second": (
                    away_match
                    .score_margin
                ),
                "Matching method": (
                    away_match.reason
                ),
            },
        ]
    )

    st.dataframe(
        matching_frame.style.format(
            {
                "Similarity": (
                    "{:.1%}"
                ),
                "Margin over second": (
                    "{:.1%}"
                ),
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        (
            '<div class="section-title">'
            'Odds API information'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    api_columns = st.columns(3)

    api_columns[0].metric(
        "Request cost",
        (
            match_odds.request_cost
            if match_odds.request_cost
            is not None
            else "Unknown"
        ),
    )

    api_columns[1].metric(
        "Credits used",
        (
            match_odds.requests_used
            if match_odds.requests_used
            is not None
            else "Unknown"
        ),
    )

    api_columns[2].metric(
        "Credits remaining",
        (
            match_odds
            .requests_remaining
            if match_odds
            .requests_remaining
            is not None
            else "Unknown"
        ),
    )



def render_prediction_history(
    history_service: PredictionHistoryService,
) -> None:
    st.markdown(
        (
            '<div class="section-kicker">'
            'Performance tracking'
            '</div>'
            '<div class="section-title">'
            'Prediction History'
            '</div>'
            '<div class="section-description">'
            'Saved analyses use the validated '
            '50% ML + 50% Poisson 1X2 probabilities. '
            'Results can be settled manually after '
            'the final score is known.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    summary = history_service.summary()

    metric_columns = st.columns(6)

    metric_columns[0].metric(
        "Saved",
        summary["total_predictions"],
    )

    metric_columns[1].metric(
        "Pending",
        summary["pending_predictions"],
    )

    metric_columns[2].metric(
        "Settled",
        summary["settled_predictions"],
    )

    metric_columns[3].metric(
        "1X2 accuracy",
        f"{summary['prediction_accuracy']:.1%}",
    )

    metric_columns[4].metric(
        "Value hit rate",
        f"{summary['bet_hit_rate']:.1%}",
    )

    metric_columns[5].metric(
        "ROI",
        f"{summary['roi']:+.1%}",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    filter_column_1, filter_column_2 = st.columns(2)

    with filter_column_1:
        selected_status = st.selectbox(
            "Status",
            options=[
                "ALL",
                "PENDING",
                "SETTLED",
            ],
        )

    competitions = [
        "ALL",
        *history_service.list_competitions(),
    ]

    with filter_column_2:
        selected_competition = st.selectbox(
            "Competition",
            options=competitions,
            key="history_competition",
        )

    history_frame = history_service.list_predictions(
        status=selected_status,
        competition=selected_competition,
        limit=1000,
    )

    if history_frame.empty:
        st.info(
            "No prediction-history records "
            "match the selected filters."
        )

        return

    display_columns = [
        "prediction_id",
        "created_at",
        "competition",
        "api_home_team",
        "api_away_team",
        "predicted_result_text",
        "confidence",
        "recommended_selection",
        "recommended_odds",
        "status",
        "actual_home_goals",
        "actual_away_goals",
        "prediction_correct",
        "bet_won",
        "profit_loss",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in history_frame.columns
    ]

    display_frame = history_frame[
        available_columns
    ].copy()

    display_frame = display_frame.rename(
        columns={
            "prediction_id": "ID",
            "created_at": "Saved at",
            "competition": "Competition",
            "api_home_team": "Home",
            "api_away_team": "Away",
            "predicted_result_text": "Prediction",
            "confidence": "Confidence",
            "recommended_selection": "Value pick",
            "recommended_odds": "Odds",
            "status": "Status",
            "actual_home_goals": "Home goals",
            "actual_away_goals": "Away goals",
            "prediction_correct": "Correct",
            "bet_won": "Bet won",
            "profit_loss": "P/L",
        }
    )

    formatters = {}

    if "Confidence" in display_frame.columns:
        formatters["Confidence"] = "{:.1%}"

    if "Odds" in display_frame.columns:
        formatters["Odds"] = "{:.2f}"

    if "P/L" in display_frame.columns:
        formatters["P/L"] = "{:+.2f}"

    st.dataframe(
        display_frame.style.format(
            formatters,
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        (
            '<div class="section-title">'
            'Delete a saved prediction'
            '</div>'
            '<div class="section-description">'
            'Use this only for analyses saved by mistake. '
            'Deletion is permanent and immediately updates '
            'all history metrics.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    deletion_ids = (
        history_frame[
            "prediction_id"
        ]
        .astype(int)
        .tolist()
    )

    deletion_lookup = {
        int(row.prediction_id): (
            f"#{int(row.prediction_id)} · "
            f"{row.api_home_team} vs "
            f"{row.api_away_team} · "
            f"{row.status}"
        )
        for row in history_frame.itertuples(
            index=False
        )
    }

    deletion_prediction_id = st.selectbox(
        "Saved prediction",
        options=deletion_ids,
        format_func=lambda value: (
            deletion_lookup[value]
        ),
        key="delete_prediction_id",
    )

    confirm_deletion = st.checkbox(
        "I understand that this record will be permanently deleted.",
        key="confirm_prediction_deletion",
    )

    delete_submitted = st.button(
        "Delete Selected Prediction",
        type="secondary",
        width="stretch",
        disabled=(
            not confirm_deletion
        ),
        key="delete_selected_prediction_button",
    )

    if delete_submitted:
        history_service.delete_prediction(
            prediction_id=int(
                deletion_prediction_id
            )
        )

        st.success(
            "The selected prediction was deleted."
        )

        st.rerun()

    pending_frame = history_frame[
        history_frame["status"]
        == "PENDING"
    ]

    if pending_frame.empty:
        return

    st.markdown(
        (
            '<div class="section-title">'
            'Settle a prediction'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    pending_ids = (
        pending_frame[
            "prediction_id"
        ]
        .astype(int)
        .tolist()
    )

    pending_lookup = {
        int(row.prediction_id): (
            f"#{int(row.prediction_id)} · "
            f"{row.api_home_team} vs "
            f"{row.api_away_team}"
        )
        for row in pending_frame.itertuples(
            index=False
        )
    }

    with st.form(
        "settle_prediction_form"
    ):
        selected_prediction_id = st.selectbox(
            "Prediction",
            options=pending_ids,
            format_func=lambda value: (
                pending_lookup[value]
            ),
        )

        score_columns = st.columns(2)

        with score_columns[0]:
            actual_home_goals = st.number_input(
                "Home goals",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
            )

        with score_columns[1]:
            actual_away_goals = st.number_input(
                "Away goals",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
            )

        settle_submitted = st.form_submit_button(
            "Save Final Result",
            type="primary",
            width="stretch",
        )

    if settle_submitted:
        history_service.settle_prediction(
            prediction_id=int(
                selected_prediction_id
            ),
            actual_home_goals=int(
                actual_home_goals
            ),
            actual_away_goals=int(
                actual_away_goals
            ),
        )

        st.success(
            "The final result was saved."
        )

        st.rerun()



def main(
) -> None:
    st.set_page_config(
        page_title=(
            "EdgeXI Football Analytics"
        ),
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state=(
            "expanded"
        ),
    )

    st.markdown(
        PAGE_CSS,
        unsafe_allow_html=True,
    )

    render_brand()

    try:
        history_service = (
            load_prediction_history_service()
        )

    except Exception as error:
        st.error(
            "Prediction history could not "
            f"be initialized: {error}"
        )

        st.stop()

    selected_page = st.sidebar.radio(
        "Workspace",
        options=[
            "Match Analysis",
            "Prediction History",
        ],
    )

    if selected_page == "Prediction History":
        render_prediction_history(
            history_service
        )

        return

    api_key = get_api_key()

    if api_key is None:
        st.error(
            "ODDS_API_KEY was not found "
            "in .streamlit/secrets.toml."
        )

        st.stop()

    try:
        prediction_service = (
            load_prediction_service()
        )

        value_service = (
            load_value_service()
        )

        team_matching_service = (
            load_team_matching_service()
        )

        active_leagues = (
            get_active_leagues(
                api_key
            )
        )

    except Exception as error:
        st.error(
            "Application startup "
            f"failed: {error}"
        )

        st.stop()

    if not active_leagues:
        st.warning(
            "No active football "
            "competitions were returned."
        )

        st.stop()

    st.sidebar.markdown(
        (
            '<div class="sidebar-section">'
            'Match centre'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    league_keys = list(
        active_leagues.keys()
    )

    selected_sport_key = (
        st.sidebar.selectbox(
            "Competition",
            options=league_keys,
            format_func=lambda key: (
                active_leagues[key]
            ),
        )
    )

    try:
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
            "Could not load fixtures: "
            f"{error}"
        )

        st.stop()

    if not league_events:
        st.info(
            "There are currently no "
            "upcoming fixtures for this "
            "competition."
        )

        st.stop()

    event_indexes = list(
        range(
            len(league_events)
        )
    )

    selected_event_index = (
        st.sidebar.selectbox(
            "Fixture",
            options=event_indexes,
            format_func=lambda index: (
                f"{league_events[index].home_team} "
                "vs "
                f"{league_events[index].away_team}"
            ),
        )
    )

    selected_event = (
        league_events[
            selected_event_index
        ]
    )

    st.sidebar.caption(
        format_event_time(
            selected_event
            .commence_time
        )
    )

    st.sidebar.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    analyse_button = (
        st.sidebar.button(
            "Run Match Analysis",
            type="primary",
            width="stretch",
        )
    )

    selected_event_id = str(
        selected_event.event_id
    )

    if analyse_button:
        st.session_state[
            "analysis_requested_event_id"
        ] = selected_event_id

    analysis_requested = (
        st.session_state.get(
            "analysis_requested_event_id"
        )
        == selected_event_id
    )

    st.sidebar.markdown(
        (
            '<div class="sidebar-note">'
            'Odds data is cached briefly '
            'to reduce API usage and '
            'unnecessary requests.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    render_match_hero(
        selected_event,
        active_leagues[
            selected_sport_key
        ],
    )

    if not analysis_requested:
        render_empty_state()

        return

    model_teams = (
        prediction_service.get_teams()
    )

    home_match = (
        team_matching_service.match(
            api_team_name=(
                selected_event.home_team
            ),
            model_teams=model_teams,
        )
    )

    away_match = (
        team_matching_service.match(
            api_team_name=(
                selected_event.away_team
            ),
            model_teams=model_teams,
        )
    )

    if (
        not home_match.accepted
        or not away_match.accepted
    ):
        manual_selection = (
            render_rejected_team_matching(
                home_match=home_match,
                away_match=away_match,
                model_teams=model_teams,
                event_id=selected_event_id,
            )
        )

        if manual_selection is None:
            return

        (
            model_home_team,
            model_away_team,
        ) = manual_selection

    else:
        model_home_team = (
            home_match.matched_team_name
        )

        model_away_team = (
            away_match.matched_team_name
        )

    if (
        model_home_team is None
        or model_away_team is None
    ):
        st.error(
            "Safe team matching did not "
            "return two valid model teams."
        )

        st.stop()

    if (
        model_home_team
        == model_away_team
    ):
        st.error(
            "Both API teams matched the same "
            "model team. The analysis was cancelled."
        )

        st.stop()

    try:
        with st.spinner(
            "Running the model and "
            "scanning live prices..."
        ):
            prediction = (
                prediction_service
                .predict(
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
                        selected_event
                        .event_id
                    ),
                    sport_key=(
                        selected_event
                        .sport_key
                    ),
                    sport_title=(
                        selected_event
                        .sport_title
                    ),
                    commence_time=(
                        selected_event
                        .commence_time
                    ),
                    home_team=(
                        selected_event
                        .home_team
                    ),
                    away_team=(
                        selected_event
                        .away_team
                    ),
                )
            )

            value_report = (
                value_service.analyse(
                    prediction=prediction,
                    match_odds=(
                        match_odds
                    ),
                )
            )

            history_saved = False

            if not (
                history_service
                .repository
                .event_exists(
                    str(
                        selected_event
                        .event_id
                    )
                )
            ):
                history_service.save_analysis(
                    event=selected_event,
                    competition=(
                        active_leagues[
                            selected_sport_key
                        ]
                    ),
                    prediction=prediction,
                    value_report=value_report,
                    model_home_team=(
                        model_home_team
                    ),
                    model_away_team=(
                        model_away_team
                    ),
                )

                history_saved = True

    except Exception as error:
        st.error(
            "Match analysis failed: "
            f"{error}"
        )

        st.stop()

    st.session_state.pop(
        "analysis_requested_event_id",
        None,
    )

    if history_saved:
        st.success(
            "This analysis was saved to "
            "Prediction History."
        )

    else:
        st.info(
            "This fixture already exists in "
            "Prediction History, so a duplicate "
            "record was not created."
        )

    top_left, top_right = (
        st.columns(
            [
                1.55,
                1,
            ]
        )
    )

    with top_left:
        render_best_bet(
            value_report
        )

    with top_right:
        st.markdown(
            (
                '<div class="section-kicker">'
                'Model goal forecast'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            (
                '<div class="section-description">'
                'These values come only from '
                'the model and score simulation. '
                'Bookmaker odds are not used.'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        metric_columns = (
            st.columns(2)
        )

        metric_columns[0].metric(
            "Most likely score",
            (
                f"{prediction.most_likely_home_goals}"
                "-"
                f"{prediction.most_likely_away_goals}"
            ),
        )

        metric_columns[1].metric(
            "Expected total goals",
            (
                f"{prediction.expected_home_goals + prediction.expected_away_goals:.2f}"
            ),
        )

        second_metric_columns = (
            st.columns(2)
        )

        second_metric_columns[0].metric(
            "Model BTTS Yes",
            (
                f"{prediction.btts_yes_probability:.1%}"
            ),
        )

        second_metric_columns[1].metric(
            "Model Over 2.5",
            (
                f"{prediction.over_2_5_probability:.1%}"
            ),
        )

    st.markdown(
        (
            '<div class="section-kicker">'
            'Model match-result prediction'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="section-description">'
            'The 1X2 probabilities below combine '
            '50% machine-learning probability and '
            '50% Poisson probability. Bookmaker '
            'prices are not used.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    render_probability_cards(
        prediction,
        selected_event,
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    (
        value_picks_tab,
        model_markets_tab,
        live_odds_tab,
        model_details_tab,
    ) = st.tabs(
        [
            "⭐ Live Odds Value Picks",
            "📊 Model Predictions",
            "💰 Live Odds Analysis",
            "🧠 Model Details",
        ]
    )

    with value_picks_tab:
        st.caption(
            "Selections in this section "
            "combine model probabilities "
            "with current bookmaker prices."
        )

        render_best_bet(
            value_report
        )

        render_alternatives(
            value_report
        )

    with model_markets_tab:
        st.caption(
            "This section contains model "
            "and simulation probabilities "
            "only. It does not use odds."
        )

        render_model_markets(
            prediction,
            selected_event,
        )

    with live_odds_tab:
        st.caption(
            "This section displays bookmaker "
            "prices and compares them with "
            "the model probabilities."
        )

        render_live_odds_analysis(
            match_odds,
            value_report,
        )

    with model_details_tab:
        render_model_details(
            prediction=prediction,
            selected_event=(
                selected_event
            ),
            home_match=home_match,
            away_match=away_match,
            match_odds=match_odds,
        )

    st.markdown(
        (
            '<div class="footer-note">'
            'EdgeXI is an experimental '
            'football analytics project. '
            'Model estimates and positive '
            'expected value do not guarantee '
            'profitable results.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
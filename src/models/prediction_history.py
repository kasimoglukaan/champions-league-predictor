from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PredictionHistoryRecord:
    prediction_id: Optional[int]

    event_id: str
    sport_key: str

    created_at: str
    competition: str
    kickoff_time: str

    api_home_team: str
    api_away_team: str

    model_home_team: str
    model_away_team: str

    ml_home_probability: float
    ml_draw_probability: float
    ml_away_probability: float

    poisson_home_probability: float
    poisson_draw_probability: float
    poisson_away_probability: float

    hybrid_home_probability: float
    hybrid_draw_probability: float
    hybrid_away_probability: float

    predicted_result: str
    predicted_result_text: str
    confidence: float
    confidence_label: str

    expected_home_goals: float
    expected_away_goals: float

    most_likely_home_goals: int
    most_likely_away_goals: int

    recommended_market: Optional[str]
    recommended_selection: Optional[str]
    recommended_odds: Optional[float]
    bookmaker: Optional[str]

    value_model_probability: Optional[float]
    market_probability: Optional[float]
    edge: Optional[float]
    expected_value: Optional[float]

    actual_home_goals: Optional[int]
    actual_away_goals: Optional[int]
    actual_result: Optional[str]

    prediction_correct: Optional[bool]
    bet_won: Optional[bool]
    profit_loss: Optional[float]

    status: str
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.prediction_history import (
    PredictionHistoryRecord,
)


class PredictionHistoryRepository:
    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    def _connect(
        self,
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = (
            sqlite3.Row
        )

        return connection

    def _initialize_database(
        self,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                prediction_history (
                    prediction_id INTEGER
                        PRIMARY KEY AUTOINCREMENT,

                    event_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    competition TEXT NOT NULL,
                    kickoff_time TEXT NOT NULL,

                    api_home_team TEXT NOT NULL,
                    api_away_team TEXT NOT NULL,

                    model_home_team TEXT NOT NULL,
                    model_away_team TEXT NOT NULL,

                    ml_home_probability REAL NOT NULL,
                    ml_draw_probability REAL NOT NULL,
                    ml_away_probability REAL NOT NULL,

                    poisson_home_probability REAL NOT NULL,
                    poisson_draw_probability REAL NOT NULL,
                    poisson_away_probability REAL NOT NULL,

                    hybrid_home_probability REAL NOT NULL,
                    hybrid_draw_probability REAL NOT NULL,
                    hybrid_away_probability REAL NOT NULL,

                    predicted_result TEXT NOT NULL,
                    predicted_result_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    confidence_label TEXT NOT NULL,

                    expected_home_goals REAL NOT NULL,
                    expected_away_goals REAL NOT NULL,

                    most_likely_home_goals INTEGER NOT NULL,
                    most_likely_away_goals INTEGER NOT NULL,

                    recommended_market TEXT,
                    recommended_selection TEXT,
                    recommended_odds REAL,
                    bookmaker TEXT,

                    value_model_probability REAL,
                    market_probability REAL,
                    edge REAL,
                    expected_value REAL,

                    actual_home_goals INTEGER,
                    actual_away_goals INTEGER,
                    actual_result TEXT,

                    prediction_correct INTEGER,
                    bet_won INTEGER,
                    profit_loss REAL,

                    status TEXT NOT NULL
                        DEFAULT 'PENDING',

                    UNIQUE (
                        event_id,
                        created_at
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_prediction_history_event_id
                ON prediction_history (
                    event_id
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_prediction_history_status
                ON prediction_history (
                    status
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_prediction_history_created_at
                ON prediction_history (
                    created_at
                )
                """
            )

            connection.commit()

    def insert(
        self,
        record: PredictionHistoryRecord,
    ) -> int:
        values = {
            "event_id": record.event_id,
            "created_at": record.created_at,
            "competition": record.competition,
            "kickoff_time": record.kickoff_time,

            "api_home_team": (
                record.api_home_team
            ),
            "api_away_team": (
                record.api_away_team
            ),

            "model_home_team": (
                record.model_home_team
            ),
            "model_away_team": (
                record.model_away_team
            ),

            "ml_home_probability": (
                record.ml_home_probability
            ),
            "ml_draw_probability": (
                record.ml_draw_probability
            ),
            "ml_away_probability": (
                record.ml_away_probability
            ),

            "poisson_home_probability": (
                record.poisson_home_probability
            ),
            "poisson_draw_probability": (
                record.poisson_draw_probability
            ),
            "poisson_away_probability": (
                record.poisson_away_probability
            ),

            "hybrid_home_probability": (
                record.hybrid_home_probability
            ),
            "hybrid_draw_probability": (
                record.hybrid_draw_probability
            ),
            "hybrid_away_probability": (
                record.hybrid_away_probability
            ),

            "predicted_result": (
                record.predicted_result
            ),
            "predicted_result_text": (
                record.predicted_result_text
            ),
            "confidence": record.confidence,
            "confidence_label": (
                record.confidence_label
            ),

            "expected_home_goals": (
                record.expected_home_goals
            ),
            "expected_away_goals": (
                record.expected_away_goals
            ),

            "most_likely_home_goals": (
                record
                .most_likely_home_goals
            ),
            "most_likely_away_goals": (
                record
                .most_likely_away_goals
            ),

            "recommended_market": (
                record.recommended_market
            ),
            "recommended_selection": (
                record.recommended_selection
            ),
            "recommended_odds": (
                record.recommended_odds
            ),
            "bookmaker": record.bookmaker,

            "value_model_probability": (
                record
                .value_model_probability
            ),
            "market_probability": (
                record.market_probability
            ),
            "edge": record.edge,
            "expected_value": (
                record.expected_value
            ),

            "actual_home_goals": (
                record.actual_home_goals
            ),
            "actual_away_goals": (
                record.actual_away_goals
            ),
            "actual_result": (
                record.actual_result
            ),

            "prediction_correct": (
                None
                if record.prediction_correct
                is None
                else int(
                    record.prediction_correct
                )
            ),
            "bet_won": (
                None
                if record.bet_won is None
                else int(
                    record.bet_won
                )
            ),
            "profit_loss": (
                record.profit_loss
            ),

            "status": record.status,
        }

        columns = ", ".join(
            values.keys()
        )

        placeholders = ", ".join(
            f":{column}"
            for column in values
        )

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                INSERT INTO prediction_history (
                    {columns}
                )
                VALUES (
                    {placeholders}
                )
                """,
                values,
            )

            connection.commit()

            return int(
                cursor.lastrowid
            )

    def event_exists(
        self,
        event_id: str,
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT prediction_id
                FROM prediction_history
                WHERE event_id = ?
                LIMIT 1
                """,
                (
                    event_id,
                ),
            ).fetchone()

        return row is not None

    def get_latest_for_event(
        self,
        event_id: str,
    ) -> Optional[dict]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM prediction_history
                WHERE event_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (
                    event_id,
                ),
            ).fetchone()

        if row is None:
            return None

        return dict(
            row
        )

    def list_predictions(
        self,
        status: Optional[str] = None,
        competition: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        query = """
            SELECT *
            FROM prediction_history
            WHERE 1 = 1
        """

        parameters: list[
            object
        ] = []

        if (
            status
            and status != "ALL"
        ):
            query += """
                AND status = ?
            """

            parameters.append(
                status
            )

        if (
            competition
            and competition != "ALL"
        ):
            query += """
                AND competition = ?
            """

            parameters.append(
                competition
            )

        query += """
            ORDER BY created_at DESC
            LIMIT ?
        """

        parameters.append(
            int(limit)
        )

        with self._connect() as connection:
            dataframe = pd.read_sql_query(
                query,
                connection,
                params=parameters,
            )

        return dataframe

    def list_competitions(
        self,
    ) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT competition
                FROM prediction_history
                WHERE competition IS NOT NULL
                  AND competition != ''
                ORDER BY competition
                """
            ).fetchall()

        return [
            str(
                row["competition"]
            )
            for row in rows
        ]

    def update_result(
        self,
        prediction_id: int,
        actual_home_goals: int,
        actual_away_goals: int,
        actual_result: str,
        prediction_correct: bool,
        bet_won: Optional[bool],
        profit_loss: Optional[float],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE prediction_history
                SET
                    actual_home_goals = ?,
                    actual_away_goals = ?,
                    actual_result = ?,
                    prediction_correct = ?,
                    bet_won = ?,
                    profit_loss = ?,
                    status = 'SETTLED'
                WHERE prediction_id = ?
                """,
                (
                    int(
                        actual_home_goals
                    ),
                    int(
                        actual_away_goals
                    ),
                    actual_result,
                    int(
                        prediction_correct
                    ),
                    (
                        None
                        if bet_won is None
                        else int(
                            bet_won
                        )
                    ),
                    profit_loss,
                    int(
                        prediction_id
                    ),
                ),
            )

            connection.commit()

    def delete(
        self,
        prediction_id: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM prediction_history
                WHERE prediction_id = ?
                """,
                (
                    int(
                        prediction_id
                    ),
                ),
            )

            connection.commit()

    def summary(
        self,
    ) -> dict[str, float | int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_predictions,

                    SUM(
                        CASE
                            WHEN status = 'SETTLED'
                            THEN 1
                            ELSE 0
                        END
                    ) AS settled_predictions,

                    SUM(
                        CASE
                            WHEN prediction_correct = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS correct_predictions,

                    SUM(
                        CASE
                            WHEN bet_won = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS won_bets,

                    SUM(
                        CASE
                            WHEN bet_won = 0
                            THEN 1
                            ELSE 0
                        END
                    ) AS lost_bets,

                    SUM(
                        CASE
                            WHEN profit_loss IS NOT NULL
                            THEN profit_loss
                            ELSE 0
                        END
                    ) AS total_profit_loss,

                    SUM(
                        CASE
                            WHEN status = 'SETTLED'
                             AND recommended_odds
                                 IS NOT NULL
                            THEN 1
                            ELSE 0
                        END
                    ) AS settled_bets

                FROM prediction_history
                """
            ).fetchone()

        total_predictions = int(
            row[
                "total_predictions"
            ]
            or 0
        )

        settled_predictions = int(
            row[
                "settled_predictions"
            ]
            or 0
        )

        correct_predictions = int(
            row[
                "correct_predictions"
            ]
            or 0
        )

        won_bets = int(
            row[
                "won_bets"
            ]
            or 0
        )

        lost_bets = int(
            row[
                "lost_bets"
            ]
            or 0
        )

        total_profit_loss = float(
            row[
                "total_profit_loss"
            ]
            or 0.0
        )

        settled_bets = int(
            row[
                "settled_bets"
            ]
            or 0
        )

        prediction_accuracy = (
            correct_predictions
            / settled_predictions
            if settled_predictions > 0
            else 0.0
        )

        bet_hit_rate = (
            won_bets
            / (
                won_bets
                + lost_bets
            )
            if (
                won_bets
                + lost_bets
            ) > 0
            else 0.0
        )

        roi = (
            total_profit_loss
            / settled_bets
            if settled_bets > 0
            else 0.0
        )

        return {
            "total_predictions": (
                total_predictions
            ),
            "settled_predictions": (
                settled_predictions
            ),
            "pending_predictions": (
                total_predictions
                - settled_predictions
            ),
            "correct_predictions": (
                correct_predictions
            ),
            "prediction_accuracy": (
                prediction_accuracy
            ),
            "won_bets": won_bets,
            "lost_bets": lost_bets,
            "bet_hit_rate": (
                bet_hit_rate
            ),
            "total_profit_loss": (
                total_profit_loss
            ),
            "roi": roi,
        }
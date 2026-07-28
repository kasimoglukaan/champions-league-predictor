from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


class ModelEvaluator:
    LABEL_ORDER = ["A", "D", "H"]

    def evaluate_named_model(
        self,
        model_manager,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        predictions = (
            model_manager.predict_with_model(
                model_name=model_name,
                X=X_test,
            )
        )

        probabilities = (
            model_manager.predict_proba_with_model(
                model_name=model_name,
                X=X_test,
            )
        )

        return {
            "accuracy": float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),
            "log_loss": float(
                log_loss(
                    y_test,
                    probabilities,
                    labels=self.LABEL_ORDER,
                )
            ),
            "multiclass_brier": (
                self._multiclass_brier(
                    y_test=y_test,
                    probabilities=probabilities,
                )
            ),
        }

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, object]:
        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )

        return {
            "accuracy": float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),
            "log_loss": float(
                log_loss(
                    y_test,
                    probabilities,
                    labels=self.LABEL_ORDER,
                )
            ),
            "multiclass_brier": (
                self._multiclass_brier(
                    y_test=y_test,
                    probabilities=probabilities,
                )
            ),
            "classification_report": (
                classification_report(
                    y_test,
                    predictions,
                    labels=self.LABEL_ORDER,
                    zero_division=0,
                )
            ),
            "confusion_matrix": (
                confusion_matrix(
                    y_test,
                    predictions,
                    labels=self.LABEL_ORDER,
                )
            ),
        }

    def _multiclass_brier(
        self,
        y_test: pd.Series,
        probabilities: np.ndarray,
    ) -> float:
        encoded_targets = np.zeros_like(
            probabilities,
            dtype=float,
        )

        label_to_index = {
            label: index
            for index, label in enumerate(
                self.LABEL_ORDER
            )
        }

        for row_index, label in enumerate(
            y_test.to_numpy()
        ):
            encoded_targets[
                row_index,
                label_to_index[label],
            ] = 1.0

        squared_errors = (
            probabilities
            - encoded_targets
        ) ** 2

        return float(
            np.mean(
                np.sum(
                    squared_errors,
                    axis=1,
                )
            )
        )
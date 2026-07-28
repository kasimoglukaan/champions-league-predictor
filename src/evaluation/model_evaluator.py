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

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, object]:
        predictions = model.predict(X_test)
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

    def evaluate_single_model(
        self,
        sklearn_model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        predictions = sklearn_model.predict(
            X_test
        )

        probabilities = (
            sklearn_model.predict_proba(
                X_test
            )
        )

        classes = list(
            sklearn_model.classes_
        )

        ordered_probabilities = np.zeros(
            (
                len(X_test),
                len(self.LABEL_ORDER),
            )
        )

        for output_index, label in enumerate(
            self.LABEL_ORDER
        ):
            class_index = classes.index(label)

            ordered_probabilities[
                :,
                output_index,
            ] = probabilities[
                :,
                class_index,
            ]

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
                    ordered_probabilities,
                    labels=self.LABEL_ORDER,
                )
            ),
            "multiclass_brier": (
                self._multiclass_brier(
                    y_test,
                    ordered_probabilities,
                )
            ),
        }

    def _multiclass_brier(
        self,
        y_test: pd.Series,
        probabilities: np.ndarray,
    ) -> float:
        encoded = np.zeros_like(
            probabilities
        )

        label_to_index = {
            label: index
            for index, label in enumerate(
                self.LABEL_ORDER
            )
        }

        for row_index, label in enumerate(
            y_test
        ):
            encoded[
                row_index,
                label_to_index[label],
            ] = 1.0

        return float(
            np.mean(
                np.sum(
                    (
                        probabilities
                        - encoded
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
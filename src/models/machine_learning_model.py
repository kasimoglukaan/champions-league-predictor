from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MachineLearningModel:
    LABEL_ORDER = ["A", "D", "H"]

    def __init__(self) -> None:
        self.models = {
            "logistic_regression": Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            multi_class="auto",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "gradient_boosting": Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            learning_rate=0.04,
                            max_iter=350,
                            max_leaf_nodes=15,
                            min_samples_leaf=25,
                            l2_regularization=1.0,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        }

        self.best_model_name = None
        self.best_model = None

    def fit_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        for model in self.models.values():
            model.fit(X_train, y_train)

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        self._ensure_best_model()

        return self.best_model.predict(X)

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        self._ensure_best_model()

        raw_probabilities = (
            self.best_model.predict_proba(X)
        )

        classes = list(
            self.best_model.classes_
        )

        ordered = np.zeros(
            (
                len(X),
                len(self.LABEL_ORDER),
            )
        )

        for output_index, label in enumerate(
            self.LABEL_ORDER
        ):
            class_index = classes.index(label)

            ordered[:, output_index] = (
                raw_probabilities[
                    :,
                    class_index,
                ]
            )

        return ordered

    def select_best_model(
        self,
        results: Dict[str, Dict[str, float]],
    ) -> str:
        best_name = min(
            results,
            key=lambda name: (
                results[name]["log_loss"],
                -results[name]["accuracy"],
            ),
        )

        self.best_model_name = best_name
        self.best_model = self.models[best_name]

        return best_name

    def save(
        self,
        file_path: str,
    ) -> None:
        self._ensure_best_model()

        path = Path(file_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "model_name": (
                    self.best_model_name
                ),
                "model": self.best_model,
                "label_order": self.LABEL_ORDER,
            },
            path,
        )

    def load(
        self,
        file_path: str,
    ) -> None:
        saved = joblib.load(file_path)

        self.best_model_name = saved[
            "model_name"
        ]

        self.best_model = saved["model"]

    def _ensure_best_model(self) -> None:
        if self.best_model is None:
            raise RuntimeError(
                "A trained best model "
                "must be selected first."
            )
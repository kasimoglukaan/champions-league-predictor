from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "hist_gradient_boosting": Pipeline(
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
                            learning_rate=0.03,
                            max_iter=300,
                            max_leaf_nodes=15,
                            min_samples_leaf=30,
                            l2_regularization=2.0,
                            class_weight="balanced",
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
            model.fit(
                X_train,
                y_train,
            )

    def predict_with_model(
        self,
        model_name: str,
        X: pd.DataFrame,
    ) -> np.ndarray:
        if model_name not in self.models:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        model = self.models[model_name]

        return model.predict(X)

    def predict_proba_with_model(
        self,
        model_name: str,
        X: pd.DataFrame,
    ) -> np.ndarray:
        if model_name not in self.models:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        model = self.models[model_name]

        raw_probabilities = (
            model.predict_proba(X)
        )

        classes = list(model.classes_)

        ordered_probabilities = np.zeros(
            (
                len(X),
                len(self.LABEL_ORDER),
            )
        )

        for output_index, label in enumerate(
            self.LABEL_ORDER
        ):
            if label not in classes:
                raise ValueError(
                    f"Model does not contain "
                    f"the expected class: {label}"
                )

            class_index = classes.index(label)

            ordered_probabilities[
                :,
                output_index,
            ] = raw_probabilities[
                :,
                class_index,
            ]

        return ordered_probabilities

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

        ordered_probabilities = np.zeros(
            (
                len(X),
                len(self.LABEL_ORDER),
            )
        )

        for output_index, label in enumerate(
            self.LABEL_ORDER
        ):
            if label not in classes:
                raise ValueError(
                    f"Model does not contain "
                    f"the expected class: {label}"
                )

            class_index = classes.index(label)

            ordered_probabilities[
                :,
                output_index,
            ] = raw_probabilities[
                :,
                class_index,
            ]

        return ordered_probabilities

    def select_best_model(
        self,
        results: Dict[
            str,
            Dict[str, float],
        ],
    ) -> str:
        if not results:
            raise ValueError(
                "Model results cannot be empty."
            )

        best_name = min(
            results,
            key=lambda name: (
                results[name]["log_loss"],
                -results[name]["accuracy"],
            ),
        )

        if best_name not in self.models:
            raise ValueError(
                f"Selected model does not exist: "
                f"{best_name}"
            )

        self.best_model_name = best_name
        self.best_model = self.models[
            best_name
        ]

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
                "label_order": (
                    self.LABEL_ORDER
                ),
            },
            path,
        )

    def load(
        self,
        file_path: str,
    ) -> None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Saved model was not found: "
                f"{path}"
            )

        saved = joblib.load(path)

        required_keys = {
            "model_name",
            "model",
            "label_order",
        }

        missing_keys = (
            required_keys
            - set(saved.keys())
        )

        if missing_keys:
            raise ValueError(
                "Saved model file is missing: "
                + ", ".join(
                    sorted(missing_keys)
                )
            )

        self.best_model_name = saved[
            "model_name"
        ]

        self.best_model = saved[
            "model"
        ]

    def _ensure_best_model(self) -> None:
        if self.best_model is None:
            raise RuntimeError(
                "A trained or loaded model "
                "must be selected first."
            )
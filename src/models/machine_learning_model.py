from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
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
                            strategy="median",
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
                            solver="lbfgs",
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
                            strategy="median",
                        ),
                    ),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            learning_rate=0.04,
                            max_iter=300,
                            max_leaf_nodes=15,
                            min_samples_leaf=25,
                            l2_regularization=2.0,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),

            "random_forest": Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                        ),
                    ),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=500,
                            max_depth=10,
                            min_samples_split=10,
                            min_samples_leaf=5,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),

            "extra_trees": Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                        ),
                    ),
                    (
                        "classifier",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            max_depth=12,
                            min_samples_split=8,
                            min_samples_leaf=4,
                            max_features="sqrt",
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        }

        self.best_model_name: Optional[str] = None
        self.best_model = None

    def fit_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, str]:
        training_status: Dict[str, str] = {}

        for model_name, model in self.models.items():
            print(
                f"Training {model_name}..."
            )

            try:
                model.fit(
                    X_train,
                    y_train,
                )

                training_status[model_name] = (
                    "success"
                )

            except Exception as error:
                training_status[model_name] = (
                    f"failed: {error}"
                )

                print(
                    f"{model_name} failed: "
                    f"{error}"
                )

        return training_status

    def predict_with_model(
        self,
        model_name: str,
        X: pd.DataFrame,
    ) -> np.ndarray:
        model = self._get_model(
            model_name
        )

        return model.predict(X)

    def predict_proba_with_model(
        self,
        model_name: str,
        X: pd.DataFrame,
    ) -> np.ndarray:
        model = self._get_model(
            model_name
        )

        raw_probabilities = (
            model.predict_proba(X)
        )

        classes = list(
            model.classes_
        )

        ordered_probabilities = np.zeros(
            (
                len(X),
                len(self.LABEL_ORDER),
            ),
            dtype=float,
        )

        for output_index, label in enumerate(
            self.LABEL_ORDER
        ):
            if label not in classes:
                raise ValueError(
                    f"{model_name} does not "
                    f"contain class {label}."
                )

            class_index = classes.index(
                label
            )

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
                "No successful model results "
                "were provided."
            )

        best_name = min(
            results,
            key=lambda name: (
                results[name]["log_loss"],
                -results[name]["accuracy"],
                results[name][
                    "multiclass_brier"
                ],
            ),
        )

        self.best_model_name = best_name
        self.best_model = self.models[
            best_name
        ]

        return best_name

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
            ),
            dtype=float,
        )

        for output_index, label in enumerate(
            self.LABEL_ORDER
        ):
            class_index = classes.index(
                label
            )

            ordered_probabilities[
                :,
                output_index,
            ] = raw_probabilities[
                :,
                class_index,
            ]

        return ordered_probabilities

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
                f"Model file not found: {path}"
            )

        saved = joblib.load(path)

        self.best_model_name = saved[
            "model_name"
        ]

        self.best_model = saved[
            "model"
        ]

    def _get_model(
        self,
        model_name: str,
    ):
        if model_name not in self.models:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        return self.models[model_name]

    def _ensure_best_model(self) -> None:
        if self.best_model is None:
            raise RuntimeError(
                "A trained or loaded model "
                "must be selected first."
            )
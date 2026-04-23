import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def encode_labels(string_labels: List[str]) -> Tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(string_labels)
    logger.info(f"Classes encoded: {list(encoder.classes_)}")
    return y_encoded, encoder


def split_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        stratify=y_data,
        random_state=random_state,
    )
    logger.info(f"Data split - Train: {len(x_train)} | Test: {len(x_test)}")
    return x_train, x_test, y_train, y_test


def build_model(hyperparams: Dict[str, Any]) -> RandomForestClassifier:
    model = RandomForestClassifier(**hyperparams)
    logger.info(f"Model instantiated with params: {hyperparams}")
    return model


def tune_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_dist: Dict[str, Any],
    n_iter: int = 30,
    cv_folds: int = 5,
    random_state: int = 42,
) -> RandomForestClassifier:
    logger.info(
        f"Starting RandomizedSearchCV - {n_iter} iterations x {cv_folds}-fold CV"
    )
    logger.info("This step can take several minutes depending on dataset size.")

    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring="accuracy",
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    search.fit(x_train, y_train)

    logger.info(f"Best CV Accuracy: {search.best_score_ * 100:.2f}%")
    logger.info(f"Best Params: {search.best_params_}")

    return search.best_estimator_


def train_model(
    model: RandomForestClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    logger.info("Training Random Forest classifier")
    model.fit(x_train, y_train)
    logger.info("Training complete")
    return model


def evaluate_model(
    model: RandomForestClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
) -> Dict[str, Any]:
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)

    logger.info("\n" + "=" * 60)
    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info("=" * 60)

    return {"accuracy": accuracy, "report": report}

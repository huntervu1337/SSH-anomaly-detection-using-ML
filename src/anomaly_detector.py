"""Anomaly Detector — Layer 2 of the SSH hybrid detection system.

Trains an Isolation Forest on Normal sessions (class=0) to detect
unknown/novel attack patterns that the rule-based labeling didn't anticipate.

Unlike the supervised RandomForest (Layer 1), this model learns the
statistical distribution of legitimate SSH activity and flags deviations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_anomaly_detector(
    X: pd.DataFrame,
    y: pd.Series,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForest:
    """Train Isolation Forest on Normal sessions only (class=0).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (all sessions).
    y : pd.Series
        Target labels (0-4).
    contamination : float
        Expected proportion of outliers in normal data.
    n_estimators : int
        Number of isolation trees.
    random_state : int
        Random seed.

    Returns
    -------
    IsolationForest
        Fitted model.
    """
    X_normal = X[y == 0]

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
    )
    iso.fit(X_normal)
    return iso


def score_session(
    iso: IsolationForest, x: pd.DataFrame
) -> Dict[str, object]:
    """Score a single session (or batch) with the anomaly detector.

    Parameters
    ----------
    iso : IsolationForest
        Fitted anomaly detector.
    x : pd.DataFrame
        Feature row(s) to score.

    Returns
    -------
    dict
        anomaly_score : float — lower = more anomalous
        is_anomaly    : int   — 1 = anomaly, 0 = normal
    """
    return {
        "anomaly_score": float(iso.decision_function(x.iloc[[0]])[0]),
        "is_anomaly": int(iso.predict(x.iloc[[0]])[0] == -1),
    }


def score_batch(
    iso: IsolationForest, X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Score a batch of sessions.

    Returns
    -------
    scores : np.ndarray
        Anomaly scores (lower = more anomalous).
    predictions : np.ndarray
        1 = anomaly, 0 = normal.
    """
    scores = iso.decision_function(X)
    predictions = (iso.predict(X) == -1).astype(int)
    return scores, predictions


def tune_contamination(
    X_train_normal: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    contamination_values: List[float] = None,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Tune contamination parameter by evaluating false alarm and detection rates.

    Parameters
    ----------
    X_train_normal : pd.DataFrame
        Normal sessions for training.
    X_test : pd.DataFrame
        Test set features.
    y_test : pd.Series
        Test set labels.
    contamination_values : list of float
        Values to try.

    Returns
    -------
    pd.DataFrame
        Results table with false_alarm_rate and detection_rate per contamination.
    """
    if contamination_values is None:
        contamination_values = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    results = []
    for c in contamination_values:
        iso = IsolationForest(
            n_estimators=n_estimators,
            contamination=c,
            max_samples="auto",
            random_state=random_state,
        )
        iso.fit(X_train_normal)

        # Normal test samples
        X_test_normal = X_test[y_test == 0]
        if len(X_test_normal) > 0:
            false_alarm = (iso.predict(X_test_normal) == -1).mean()
        else:
            false_alarm = float("nan")

        # Attack test samples
        X_test_attack = X_test[y_test > 0]
        if len(X_test_attack) > 0:
            detection = (iso.predict(X_test_attack) == -1).mean()
        else:
            detection = float("nan")

        # Per-class detection
        per_class = {}
        for cls in sorted(y_test.unique()):
            X_cls = X_test[y_test == cls]
            if len(X_cls) > 0:
                rate = (iso.predict(X_cls) == -1).mean()
                per_class[f"detect_class_{cls}"] = rate

        results.append({
            "contamination": c,
            "false_alarm_rate": false_alarm,
            "detection_rate": detection,
            **per_class,
        })

    return pd.DataFrame(results)

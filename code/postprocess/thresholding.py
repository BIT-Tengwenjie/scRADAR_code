"""Threshold selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ThresholdResult:
    threshold: float
    diagnostics: Dict[str, object]


def _ensure_candidates(candidate_thresholds: Optional[Iterable[float]]) -> np.ndarray:
    if candidate_thresholds is None:
        return np.linspace(0.1, 0.9, 17)
    return np.asarray(list(candidate_thresholds), dtype=float)


def _precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def _fbeta_score(precision: float, recall: float, beta: float) -> float:
    beta_sq = float(beta) ** 2
    if precision == 0.0 and recall == 0.0:
        return 0.0
    denominator = beta_sq * precision + recall
    if denominator == 0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denominator


def _enforce_min_recall(
    history: Sequence[Dict[str, float]],
    min_recall: Optional[float],
    default_threshold: float,
) -> tuple[float, Optional[Dict[str, float]]]:
    if not min_recall or min_recall <= 0.0:
        return default_threshold, None
    eligible = [entry for entry in history if entry.get("recall", 0.0) >= min_recall]
    if not eligible:
        return default_threshold, None
    best_entry = min(eligible, key=lambda x: x["threshold"])
    return float(best_entry["threshold"]), best_entry


def learn_thresholds(
    strategy: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    candidate_thresholds: Optional[Iterable[float]] = None,
    min_recall: Optional[float] = None,
    beta: float = 1.0,
) -> ThresholdResult:
    """Learn a decision threshold according to ``strategy``."""
    strategy = strategy or "f1_grid"
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    candidates = _ensure_candidates(candidate_thresholds)

    if strategy == "fixed_0.5":
        return ThresholdResult(0.5, {"strategy": strategy})

    if strategy == "youden":
        best_threshold = 0.5
        best_score = -np.inf
        history = []
        for threshold in candidates:
            preds = (y_prob >= threshold).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            tn = np.sum((preds == 0) & (y_true == 0))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = sensitivity + specificity - 1.0
            history.append(
                {
                    "threshold": float(threshold),
                    "youden_j": score,
                    "recall": sensitivity,
                    "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                }
            )
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        adjusted_threshold, recall_entry = _enforce_min_recall(history, min_recall, best_threshold)
        diagnostics = {
            "strategy": strategy,
            "history": history,
            "best_score": float(best_score),
            "min_recall_enforced": recall_entry,
        }
        return ThresholdResult(adjusted_threshold, diagnostics)

    best_threshold = 0.5
    best_f = -np.inf
    history = []
    for threshold in candidates:
        preds = (y_prob >= threshold).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        precision, recall = _precision_recall(int(tp), int(fp), int(fn))
        f_beta = _fbeta_score(precision, recall, beta)
        history.append(
            {
                "threshold": float(threshold),
                "f_beta": f_beta,
                "beta": float(beta),
                "precision": precision,
                "recall": recall,
            }
        )
        if f_beta > best_f:
            best_f = f_beta
            best_threshold = float(threshold)
    adjusted_threshold, recall_entry = _enforce_min_recall(history, min_recall, best_threshold)
    diagnostics = {
        "strategy": "f1_grid",
        "history": history,
        "beta": float(beta),
        "best_f_beta": float(best_f),
        "min_recall_enforced": recall_entry,
    }
    return ThresholdResult(adjusted_threshold, diagnostics)

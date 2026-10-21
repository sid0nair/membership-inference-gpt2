"""Low-FPR evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(y, s, max_fpr: float = 0.01) -> float:
    """Highest TPR among ROC points with FPR strictly below ``max_fpr``."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr[fpr < max_fpr]))


def strict_tpr_fast(s, y, max_fpr: float = 0.01) -> float:
    """Same quantity as ``tpr_at_fpr`` without building the ROC curve (for bootstraps)."""
    s, y = np.asarray(s), np.asarray(y)
    neg, pos = s[y == 0], s[y == 1]
    thr = np.sort(neg)[::-1][int(np.ceil(max_fpr * neg.size)) - 1]
    return float(np.mean(pos > thr))


def stratified_bootstrap(y, n: int = 2000, seed: int = 0) -> list[np.ndarray]:
    """Resample members and non-members separately (keeps the class balance fixed)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return [np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)]) for _ in range(n)]


def paired_delta(a, b, y, boot) -> dict:
    """Strict-TPR difference a - b with a paired bootstrap interval."""
    y = np.asarray(y)
    d = np.array([strict_tpr_fast(a[ix], y[ix]) - strict_tpr_fast(b[ix], y[ix]) for ix in boot])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": tpr_at_fpr(y, a) - tpr_at_fpr(y, b), "ci": (float(lo), float(hi)),
            "p_positive": float(np.mean(d > 0))}


def summary(y, s) -> dict:
    return {"auc": float(roc_auc_score(y, s)), "tpr@1%": tpr_at_fpr(y, s), "tpr@5%": tpr_at_fpr(y, s, 0.05)}

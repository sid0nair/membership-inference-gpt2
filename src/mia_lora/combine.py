"""Tail-focused linear combination of membership scores.

Pairwise logistic ranking loss between all members and the currently highest-scoring
``frac`` of non-members, averaged with equal weight over mock targets, initialized from the
first feature alone and L2-regularized toward that initialization. The hard-negative set is
refreshed from the current weights until it stops changing.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def tail_rank_fit(Zs, ys, lam: float, w0, frac: float = 0.05, rounds: int = 10):
    w, prev = np.asarray(w0, float).copy(), None
    w0 = np.asarray(w0, float)
    for r in range(rounds):
        hard = []
        for Z, y in zip(Zs, ys):
            neg = np.flatnonzero(y == 0)
            k = max(1, int(np.ceil(frac * neg.size)))
            hard.append(np.sort(neg[np.argsort(-(Z[neg] @ w))[:k]]))
        key = tuple(map(tuple, hard))
        if key == prev:
            break
        prev = key

        def obj(v):
            L, G = 0.0, np.zeros_like(v)
            for Z, y, h in zip(Zs, ys, hard):
                Pz, Nz = Z[y == 1], Z[h]
                D = (Pz @ v)[:, None] - (Nz @ v)[None, :]
                L += np.logaddexp(0.0, -D).mean()
                g = -expit(-D) / D.size
                G += g.sum(1) @ Pz - g.sum(0) @ Nz
            L /= len(Zs)
            G /= len(Zs)
            return L + lam * ((v - w0) ** 2).sum(), G + 2 * lam * (v - w0)

        w = minimize(obj, w, jac=True, method="L-BFGS-B").x
    return w, r + 1


def initial_scale(Zs, ys) -> float:
    """Scale of the first feature alone under the same tail loss (sets the margin of w0)."""
    a, _ = tail_rank_fit([Z[:, :1] for Z in Zs], ys, 0.0, np.array([1.0]))
    return float(a[0])

"""Reference-model membership scores. Higher score = more likely a member.

Notation (per candidate text, vectorized over texts):
    t   : target model's mean token loss                       shape (N,)
    L   : number of scored tokens                               shape (N,)
    P   : reference models' mean token losses                   shape (K, N); NaN = reference unusable
    Pin : True where the reference model trained on the text    shape (K, N)
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _masks(P: np.ndarray, Pin: np.ndarray):
    fin = np.isfinite(P)
    return fin, Pin & fin, ~Pin & fin


def gaussian_lr(t, L, P, Pin, kappa: float = 8.0) -> np.ndarray:
    """Per-text Gaussian likelihood ratio on the sqrt(T)-scaled mean loss (main attack).

    For every text, IN and OUT references give a mean and a variance of u = sqrt(T) * mean loss.
    Each per-text variance is shrunk toward the pooled IN (or OUT) variance with strength kappa
    (an inverse-gamma prior worth kappa pseudo-observations), multiplied by (1 + 1/n) to get a
    predictive variance, and the score is log N(u; IN) - log N(u; OUT).
    """
    sq = np.sqrt(L)
    U, u = P * sq, t * sq
    fin, m_in, m_out = _masks(P, Pin)
    n_in, n_out = m_in.sum(0), m_out.sum(0)
    if not ((n_in > 0).all() and (n_out > 0).all()):
        raise ValueError("every text needs at least one IN and one OUT reference")
    Uz = np.where(fin, U, 0.0)
    mu_in = np.where(m_in, Uz, 0.0).sum(0) / n_in
    mu_out = np.where(m_out, Uz, 0.0).sum(0) / n_out
    rss_in = np.where(m_in, (Uz - mu_in) ** 2, 0.0).sum(0)
    rss_out = np.where(m_out, (Uz - mu_out) ** 2, 0.0).sum(0)
    df_in, df_out = np.maximum(n_in - 1, 0), np.maximum(n_out - 1, 0)
    pool_out = rss_out.sum() / df_out.sum()
    pool_in = rss_in.sum() / df_in.sum() if df_in.sum() > 0 else pool_out
    s_in = (rss_in + kappa * pool_in) / (df_in + kappa) * (1 + 1 / n_in)
    s_out = (rss_out + kappa * pool_out) / (df_out + kappa) * (1 + 1 / n_out)
    return ((-(u - mu_in) ** 2 / (2 * s_in) - 0.5 * np.log(s_in))
            - (-(u - mu_out) ** 2 / (2 * s_out) - 0.5 * np.log(s_out)))


def mean_gap(t, L, P, Pin) -> np.ndarray:
    """Mean OUT-reference loss minus target loss, scaled by sqrt(T) (offline reference attack)."""
    _, _, m_out = _masks(P, Pin)
    n_out = m_out.sum(0)
    if not (n_out > 0).all():
        raise ValueError("every text needs at least one OUT reference")
    return (np.where(m_out, P, 0.0).sum(0) / n_out - t) * np.sqrt(L)


def shared_variance_lr(t, R, Rin, pool_idx=None, pool=None, kappa: float = 8.0):
    """Gaussian LR with one shared within-group variance per text, in a numerically stable form.

    score = (mu_in - mu_out) * (t - (mu_in + mu_out) / 2) / v. The pooled variance is estimated
    on ``pool_idx`` texts (e.g. a training fold) unless given explicitly. Returns (score, pool).
    """
    Rout = ~Rin
    n_in, n_out = Rin.sum(0), Rout.sum(0)
    if not ((n_in > 0).all() and (n_out > 0).all()):
        raise ValueError("every text needs at least one IN and one OUT reference")
    mu_in = np.where(Rin, R, 0.0).sum(0) / n_in
    mu_out = np.where(Rout, R, 0.0).sum(0) / n_out
    rss = np.where(Rin, (R - mu_in) ** 2, 0.0).sum(0) + np.where(Rout, (R - mu_out) ** 2, 0.0).sum(0)
    df = np.maximum(n_in - 1, 0) + np.maximum(n_out - 1, 0)
    if pool is None:
        idx = slice(None) if pool_idx is None else pool_idx
        pool = rss[idx].sum() / df[idx].sum()
    v = (rss + kappa * pool) / (df + kappa)
    return (mu_in - mu_out) * (t - 0.5 * (mu_in + mu_out)) / v, float(pool)


def log_trimmed_mean_exp(logp: np.ndarray, proportion: float = 0.2) -> np.ndarray:
    """log of scipy.stats.trim_mean(exp(logp), proportion, axis=0), computed in log space."""
    K = logp.shape[0]
    c = int(proportion * K)
    S = np.sort(logp, axis=0)[c:K - c]
    return logsumexp(S, axis=0) - np.log(S.shape[0])


def rmia_log_ratio(t_logp, ref_logp, proportion: float = 0.2) -> np.ndarray:
    """Continuous RMIA statistic a(x) = log p_target(x) - log of the trimmed mean reference probability."""
    return t_logp - log_trimmed_mean_exp(ref_logp, proportion)


def rmia_fraction(a_x, a_population, gamma: float = 1.0) -> np.ndarray:
    """RMIA pairwise score: fraction of population texts z with LR(x, z) > gamma."""
    az = np.sort(a_population)
    return np.searchsorted(az, a_x - np.log(gamma), side="left") / az.size


def ez_error_zone_ratio(t_logp, r_logp, top1, target_ids, ignore_first: bool = True, min_tokens: int = 2) -> float:
    """EZ-MIA score for one text: on positions where the target's top-1 guess is wrong,
    sum of positive (target - reference) log-prob gaps divided by the sum of negative gaps."""
    valid = np.ones(t_logp.size, dtype=bool)
    if ignore_first and valid.size:
        valid[0] = False
    err = (top1 != target_ids) & valid
    if err.sum() < min_tokens:
        return 0.0
    d = (t_logp.astype(np.float32) - r_logp.astype(np.float32))[err]
    pos, neg = d[d > 0].sum(dtype=np.float32), np.abs(d[d < 0]).sum(dtype=np.float32)
    if neg == 0:
        return 0.0
    v = float(pos / (neg + np.float32(1e-12)))
    return 0.0 if (v != v or v < 0) else v

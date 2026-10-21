import zlib

import numpy as np
import pytest
from scipy.stats import trim_mean
from sklearn.metrics import roc_curve

from mia_lora.attacks import (gaussian_lr, log_trimmed_mean_exp, mean_gap, rmia_fraction,
                              shared_variance_lr)
from mia_lora.combine import tail_rank_fit
from mia_lora.data import CandidateIndex, assign_complementary
from mia_lora.metrics import strict_tpr_fast, tpr_at_fpr


def reference_lr(mt, L, P, Pin, kappa=8):
    """Literal copy of the notebook implementation the experiments were run with."""
    sq = np.sqrt(L); U, u = P * sq, mt * sq
    fin = np.isfinite(P); m_in, m_out = Pin & fin, ~Pin & fin
    n_in, n_out = m_in.sum(0), m_out.sum(0)
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
    return (-(u - mu_in) ** 2 / (2 * s_in) - 0.5 * np.log(s_in)) - (-(u - mu_out) ** 2 / (2 * s_out) - 0.5 * np.log(s_out))


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    K, N = 12, 400
    L = rng.integers(25, 512, N).astype(float)
    Pin = np.zeros((K, N), bool)
    Pin[:8] = rng.random((8, N)) < 0.5
    Pin[1] = ~Pin[0]                                # complementary pair: one IN and one OUT per text
    P = 3.0 + rng.normal(0, 0.3, (K, N)) - 0.4 * Pin
    P[10, :50] = np.nan                             # masked reference
    y = rng.random(N) < 0.5
    t = 3.0 + rng.normal(0, 0.3, N) - 0.4 * y
    return t, L, P, Pin, y.astype(int)


def test_gaussian_lr_matches_notebook(synthetic):
    t, L, P, Pin, _ = synthetic
    assert np.allclose(gaussian_lr(t, L, P, Pin), reference_lr(t, L, P, Pin), atol=1e-12)


def test_gaussian_lr_separates(synthetic):
    t, L, P, Pin, y = synthetic
    s = gaussian_lr(t, L, P, Pin)
    assert s[y == 1].mean() > s[y == 0].mean()


def test_mean_gap_uses_out_only(synthetic):
    t, L, P, Pin, _ = synthetic
    m = ~Pin & np.isfinite(P)
    expect = (np.nanmean(np.where(m, P, np.nan), 0) - t) * np.sqrt(L)
    assert np.allclose(mean_gap(t, L, P, Pin), expect)


def test_shared_variance_stable_form(synthetic):
    t, L, P, Pin, _ = synthetic
    R, Rin = P[:8], Pin[:8]
    s, pool = shared_variance_lr(t, R, Rin)
    n_in, n_out = Rin.sum(0), (~Rin).sum(0)
    mi = np.where(Rin, R, 0).sum(0) / n_in
    mo = np.where(~Rin, R, 0).sum(0) / n_out
    rss = np.where(Rin, (R - mi) ** 2, 0).sum(0) + np.where(~Rin, (R - mo) ** 2, 0).sum(0)
    df = np.maximum(n_in - 1, 0) + np.maximum(n_out - 1, 0)
    v = (rss + 8 * pool) / (df + 8)
    assert np.allclose(s, ((t - mo) ** 2 - (t - mi) ** 2) / (2 * v))


def test_log_trimmed_mean_exp():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, (8, 100))
    assert np.allclose(log_trimmed_mean_exp(np.log(p)), np.log(trim_mean(p, 0.2, axis=0)))


def test_rmia_fraction_bruteforce():
    rng = np.random.default_rng(2)
    ax, az = rng.normal(size=50), rng.normal(size=300)
    brute = np.array([(np.exp(a - az) > 2.0).mean() for a in ax])
    assert np.allclose(rmia_fraction(ax, az, 2.0), brute)


def test_strict_tpr_fast_matches_roc():
    rng = np.random.default_rng(3)
    for n_neg in (1000, 497, 510):
        y = np.r_[np.ones(800, int), np.zeros(n_neg, int)]
        s = rng.normal(size=y.size) + 2.0 * y
        assert np.isclose(strict_tpr_fast(s, y), tpr_at_fpr(y, s))


def test_tail_rank_strong_regularization_stays_at_init():
    rng = np.random.default_rng(4)
    Zs = [rng.normal(size=(300, 3)) for _ in range(3)]
    ys = [(Z[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(int) for Z in Zs]
    w0 = np.array([2.0, 0.0, 0.0])
    w, _ = tail_rank_fit(Zs, ys, lam=1e4, w0=w0)
    assert np.allclose(w, w0, atol=1e-2)


class WordTokenizer:
    """Minimal stand-in for a HF tokenizer: one id per whitespace token."""
    def __call__(self, texts, add_special_tokens=False, truncation=True, max_length=512, padding=False):
        return {"input_ids": [[zlib.crc32(w.encode()) % 50000 for w in t.split()][:max_length] for t in texts]}


def test_complementary_split_is_exclusive():
    tok = WordTokenizer()
    dev = [f"text number {i} alpha" for i in range(40)] + ["text number 3 alpha"]     # duplicate joins a group
    val = [f"other text {i}" for i in range(30)]
    idx = CandidateIndex(tok, {"dev": dev, "val": val}, max_length=16)
    groups = idx.groups()
    assert len(groups) == len(dev) + len(val) - 1
    a, b = assign_complementary(groups, seed=7)
    assert not set(a) & set(b) and len(a) + len(b) == len(dev) + len(val)
    sets, items = {"dev": dev, "val": val}, idx.items()
    text_of = lambda k: sets[items[k][0]][items[k][1]]
    mem_a, mem_b = idx.membership([text_of(k) for k in a]), idx.membership([text_of(k) for k in b])
    for k in ("dev", "val"):
        assert (mem_a[k] ^ mem_b[k]).all()

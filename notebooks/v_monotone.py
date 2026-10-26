# V-MONO — Monotone-constrained Gaussian LR (scoring hypothesis, not a bug fix).
# The unequal-variance Gaussian LR is quadratic in u; its non-monotonicity is valid under the Gaussian model. Here we add
# the extra assumption "lower loss never means weaker membership evidence" to guard against unreliable tail estimates:
#   u > μ_out : score = min of LR over the interval [μ_out, u]      (running minimum; endpoints + vertex)
#   u < μ_in  : score = max of LR over the interval [u, μ_in]       (running maximum; endpoints + vertex)
#   texts with μ_in > μ_out (reversed means) keep their original score.
# The lower-tail rule can raise scores of non-members with unusually low loss; this is not guaranteed to remove FPs.
# Run after V-BATCH (same runtime). Prints dev / fresh-mock context, then builds val ZIPs at κ=6 and κ=8.
for v in ("build", "MT_VAL", "L_VAL", "s8", "zip_scores", "OUTB", "NAMES4"):
    assert v in globals(), f"run V-BATCH first ({v} missing)"
from scipy.stats import rankdata

def lr_parts(mt, L, P, Pin, kappa):
    sq = np.sqrt(L); U, u = P * sq, mt * sq
    fin = np.isfinite(P); m_in, m_out = Pin & fin, ~Pin & fin
    n_in, n_out = m_in.sum(0), m_out.sum(0)
    assert (n_in > 0).all() and (n_out > 0).all(), "every text needs IN and OUT references"
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
    assert np.isfinite(s_in).all() and np.isfinite(s_out).all() and (s_in > 0).all() and (s_out > 0).all()
    f = lambda x: (-(x - mu_in) ** 2 / (2 * s_in) - 0.5 * np.log(s_in)) - (-(x - mu_out) ** 2 / (2 * s_out) - 0.5 * np.log(s_out))
    with np.errstate(divide="ignore", invalid="ignore"):
        vtx = (mu_in / s_in - mu_out / s_out) / (1 / s_in - 1 / s_out)     # stationary point (inf if s_in == s_out)
    return f, u, mu_in, mu_out, vtx

def lr_mono(mt, L, P, Pin, kappa):
    f, u, mi, mo, v = lr_parts(mt, L, P, Pin, kappa)
    raw = f(u); s = raw.copy()
    ok = mi <= mo                                                     # reversed means are left unchanged
    fv = np.where(np.isfinite(v), f(np.where(np.isfinite(v), v, 0.0)), np.nan)
    hi = ok & (u > mo)
    s = np.where(hi, np.minimum(raw, f(mo)), s)
    s = np.where(hi & np.isfinite(v) & (v > mo) & (v < u), np.minimum(s, fv), s)
    lo = ok & (u < mi)
    s = np.where(lo, np.maximum(s, f(mi)), s)
    s = np.where(lo & np.isfinite(v) & (v > u) & (v < mi), np.maximum(s, fv), s)
    info = {"reversed means": int((~ok).sum()), "eligible high": int(hi.sum()), "eligible low": int(lo.sum()),
            "decreased": int((s < raw - 1e-12).sum()), "increased": int((s > raw + 1e-12).sum())}
    return s, raw, info

def near_cutoff(s, raw):
    """Overlap and rank movement near the operating region (≈767 accepted), against the raw score at the same κ."""
    out = {f"top-{k} overlap": len(set(np.argsort(-s)[:k]) & set(np.argsort(-raw)[:k])) for k in (700, 750, 800)}
    rs, rr = rankdata(-s, method="ordinal"), rankdata(-raw, method="ordinal")
    band = (rr >= 650) & (rr <= 900)
    out["|rank Δ| in raw ranks 650–900: mean"] = float(np.abs(rs - rr)[band].mean())
    out["max"] = int(np.abs(rs - rr)[band].max())
    out["crossings of rank 767"] = int(((rr <= 767) != (rs <= 767)).sum())
    return out

# Gate: raw part equals the frozen LR on dev.
P_dev = np.vstack([np.stack([mean_loss(n) for n in NAMES4]), FIX4])
Pin_dev = np.vstack([np.stack([MEM[n]["train"] for n in NAMES4]), np.zeros(FIX4.shape, bool)])
_, raw8, _ = lr_mono(DEV_MT, DEV_L, P_dev, Pin_dev, 8.0)
assert np.allclose(raw8, LR4_DEV, rtol=0, atol=1e-6)

# Dev: actual changes for the top-20 non-members (by raw score), and strict TPR.
neg = np.flatnonzero(LABELS == 0)
for k in (6.0, 8.0):
    s, raw, info = lr_mono(DEV_MT, DEV_L, P_dev, Pin_dev, k)
    t20 = neg[np.argsort(-raw[neg])[:20]]
    print(f"dev κ={k:g}: {info} | top-20 non-members: decreased {int((s[t20] < raw[t20] - 1e-12).sum())}, "
          f"increased {int((s[t20] > raw[t20] + 1e-12).sum())} | strict TPR raw {tpr_strict(LABELS, raw):.4f} "
          f"→ mono {tpr_strict(LABELS, s):.4f} | AUC {roc_auc_score(LABELS, raw):.4f} → {roc_auc_score(LABELS, s):.4f}")

# Fresh mocks (own pair excluded): mono minus raw, per target.
ctx = []
for k in (6.0, 8.0):
    r = {"κ": k}
    for t in ("p4A", "p4B", "p5A", "p5B", "p6A", "p6B"):
        keep = [n for n in NAMES4 if n not in (t[:2] + "A", t[:2] + "B")]
        Pm = np.vstack([np.stack([mean_loss(n) for n in keep]), FIX4])
        Pmin = np.vstack([np.stack([MEM[n]["train"] for n in keep]), np.zeros(FIX4.shape, bool)])
        y = MEM[t]["train"].astype(int)
        s, raw, _ = lr_mono(mean_loss(t), DEV_L, Pm, Pmin, k)
        r[f"{t} Δ"] = tpr_strict(y, s) - tpr_strict(y, raw)
    ctx.append(r)
ctx = pd.DataFrame(ctx); ctx["mean Δ"] = ctx.filter(like=" Δ").mean(1)
print("Fresh mocks, mono minus raw strict TPR:"); print(ctx.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

# Val ZIPs (raw κ=6 stays the current best at 76.7; nothing here overwrites it).
rows = []
P, Pin = build()
for name, k in (("M6_mono_k6", 6.0), ("M8_mono_k8", 8.0)):
    s, raw_v, info = lr_mono(MT_VAL, L_VAL, P, Pin, k)
    if k == 8.0:
        assert np.abs(raw_v - s8).max() < 1e-12
    assert s.shape == (2000,) and np.isfinite(s).all()
    pred = ROOT / "data" / "val" / f"prediction_{name}.csv"
    zp = ROOT / f"project_phase2_val_submission_{name}.zip"
    np.savetxt(pred, s, fmt="%.17g")
    for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", "val",
                 "--prediction", str(pred), "--output", str(zp)],
                [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", "val"]):
        rr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); assert rr.returncode == 0, rr.stderr[-800:]
    assert np.allclose(zip_scores(zp), s, rtol=0, atol=1e-9)
    shutil.copy(zp, OUTB)
    rows.append({"candidate": name, "κ": k, **info, **near_cutoff(s, raw_v), "zip": zp.name})
print("Val, against raw LR at the same κ:"); print(pd.DataFrame(rows).T.to_string(header=False))

# V-MONO — Monotone LR: the unequal-variance Gaussian LR is quadratic in u, so it can rank a text whose loss is ABOVE
# the OUT mean as member-like (wide IN variance) or penalize very low loss (narrow IN variance). Fix, label-free:
#   u > μ_out : score = min(LR(u), LR(μ_out))      u < μ_in : score = max(LR(u), LR(μ_in))
# Run after V-BATCH (same runtime). Prints dev / fresh-mock context, then builds val ZIPs at κ=6 and κ=8.
for v in ("build", "MT_VAL", "L_VAL", "s8", "zip_scores", "OUTB", "NAMES4"):
    assert v in globals(), f"run V-BATCH first ({v} missing)"

def lr_parts(mt, L, P, Pin, kappa):
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
    f = lambda x: (-(x - mu_in) ** 2 / (2 * s_in) - 0.5 * np.log(s_in)) - (-(x - mu_out) ** 2 / (2 * s_out) - 0.5 * np.log(s_out))
    with np.errstate(divide="ignore", invalid="ignore"):
        vtx = (mu_in / s_in - mu_out / s_out) / (1 / s_in - 1 / s_out)     # stationary point of the quadratic
    return f, u, mu_in, mu_out, vtx

def lr_mono(mt, L, P, Pin, kappa):
    """Exact running min above μ_out and running max below μ_in (quadratic: endpoints + vertex suffice)."""
    f, u, mi, mo, v = lr_parts(mt, L, P, Pin, kappa)
    raw = f(u); s = raw.copy()
    fv = np.where(np.isfinite(v), f(np.where(np.isfinite(v), v, 0.0)), np.nan)
    hi = u > mo
    in_hi = hi & np.isfinite(v) & (v > mo) & (v < u)
    s = np.where(hi, np.minimum(raw, f(mo)), s); s = np.where(in_hi, np.minimum(s, fv), s)
    lo = u < mi
    in_lo = lo & np.isfinite(v) & (v > u) & (v < mi)
    s = np.where(lo, np.maximum(raw, f(mi)), s); s = np.where(in_lo, np.maximum(s, fv), s)
    return s, raw, u, mi, mo

# Gate: unclamped part equals the frozen LR.
P_dev = np.vstack([np.stack([mean_loss(n) for n in NAMES4]), FIX4])
Pin_dev = np.vstack([np.stack([MEM[n]["train"] for n in NAMES4]), np.zeros(FIX4.shape, bool)])
_, raw, u, mi, mo = lr_mono(DEV_MT, DEV_L, P_dev, Pin_dev, 8.0)
assert np.allclose(raw, LR4_DEV, rtol=0, atol=1e-6)

# Diagnostic on dev: where do the top-ranked non-members sit relative to the means?
neg = np.flatnonzero(LABELS == 0); topneg = neg[np.argsort(-raw[neg])[:20]]
print(f"dev: μ_in<μ_out on {np.mean(mi < mo):.3f} of texts | top-20 non-members: u>μ_out {int((u[topneg] > mo[topneg]).sum())}, "
      f"u<μ_in {int((u[topneg] < mi[topneg]).sum())} | all texts clamped: high {int((u > mo).sum())}, low {int((u < mi).sum())}")

ctx = []
for k in (6.0, 8.0):
    r = {"κ": k, "dev raw": tpr_strict(LABELS, lr_frozen(DEV_MT, DEV_L, P_dev, Pin_dev, kappa=k)),
         "dev mono": tpr_strict(LABELS, lr_mono(DEV_MT, DEV_L, P_dev, Pin_dev, k)[0])}
    for t in ("p4A", "p4B", "p5A", "p5B", "p6A", "p6B"):
        pr = (t[:2] + "A", t[:2] + "B")
        keep = [n for n in NAMES4 if n not in pr]
        Pm = np.vstack([np.stack([mean_loss(n) for n in keep]), FIX4])
        Pmin = np.vstack([np.stack([MEM[n]["train"] for n in keep]), np.zeros(FIX4.shape, bool)])
        y = MEM[t]["train"].astype(int)
        r[f"{t} Δ"] = tpr_strict(y, lr_mono(mean_loss(t), DEV_L, Pm, Pmin, k)[0]) - tpr_strict(y, lr_frozen(mean_loss(t), DEV_L, Pm, Pmin, kappa=k))
    ctx.append(r)
print("Context (mono minus raw on mocks; information only):")
print(pd.DataFrame(ctx).to_string(index=False, float_format=lambda x: f"{x:+.4f}" if abs(x) < 0.5 else f"{x:.4f}"))

top = lambda s, k: set(np.argsort(-s)[:k])
rows = []
P, Pin = build()
for name, k in (("M6_mono_k6", 6.0), ("M8_mono_k8", 8.0)):
    s, raw_v, uv, miv, mov = lr_mono(MT_VAL, L_VAL, P, Pin, k)
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
    rows.append({"candidate": name, "κ": k, "texts changed vs raw": int((np.abs(s - raw_v) > 1e-12).sum()),
                 "top-50 changed vs raw": 50 - len(top(s, 50) & top(raw_v, 50)),
                 "top-50 overlap vs LR4": len(top(s, 50) & top(s8, 50)), "zip": zp.name})
print(pd.DataFrame(rows).to_string(index=False))

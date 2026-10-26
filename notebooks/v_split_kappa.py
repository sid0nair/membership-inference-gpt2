# V-SPLIT — Separate shrinkage for IN and OUT variances (κ_in, κ_out). Run after V-BATCH and V-MONO (same runtime).
# Equal κ already shrinks IN more (typical text at κ=6: IN weight 6/(3+6)=67%, OUT 6/(7+6)=46%); separate κ values change
# the prior strength beyond that. Local tuning informed by validation feedback (κ=6 best of 4/6/8/16/32), not evidence
# about the cause of the remaining false positives. One change at a time, against raw κ=6 (76.7):
#   A: κ_in=8, κ_out=6 (stronger IN shrinkage only)     B: κ_in=6, κ_out=4 (weaker OUT shrinkage only)
for v in ("build", "MT_VAL", "L_VAL", "s8", "zip_scores", "OUTB", "NAMES4", "near_cutoff"):
    assert v in globals(), f"run V-BATCH and V-MONO first ({v} missing)"

def lr_split(mt, L, P, Pin, k_in, k_out):
    sq = np.sqrt(L); U, u = P * sq, mt * sq
    fin = np.isfinite(P); m_in, m_out = Pin & fin, ~Pin & fin
    n_in, n_out = m_in.sum(0), m_out.sum(0)
    assert (n_in > 0).all() and (n_out > 0).all()
    Uz = np.where(fin, U, 0.0)
    mu_in = np.where(m_in, Uz, 0.0).sum(0) / n_in
    mu_out = np.where(m_out, Uz, 0.0).sum(0) / n_out
    rss_in = np.where(m_in, (Uz - mu_in) ** 2, 0.0).sum(0)
    rss_out = np.where(m_out, (Uz - mu_out) ** 2, 0.0).sum(0)
    df_in, df_out = np.maximum(n_in - 1, 0), np.maximum(n_out - 1, 0)
    pool_out = rss_out.sum() / df_out.sum()
    pool_in = rss_in.sum() / df_in.sum() if df_in.sum() > 0 else pool_out
    s_in = (rss_in + k_in * pool_in) / (df_in + k_in) * (1 + 1 / n_in)
    s_out = (rss_out + k_out * pool_out) / (df_out + k_out) * (1 + 1 / n_out)
    assert (s_in > 0).all() and (s_out > 0).all() and np.isfinite(s_in).all() and np.isfinite(s_out).all()
    A = (-(u - mu_in) ** 2 / (2 * s_in) - 0.5 * np.log(s_in)) - (-(u - mu_out) ** 2 / (2 * s_out) - 0.5 * np.log(s_out))
    assert np.isfinite(A).all()
    return A

P, Pin = build()
s6 = lr_frozen(MT_VAL, L_VAL, P, Pin, kappa=6.0)
assert np.allclose(lr_split(MT_VAL, L_VAL, P, Pin, 6.0, 6.0), s6, rtol=0, atol=1e-12)
assert np.allclose(lr_split(MT_VAL, L_VAL, P, Pin, 8.0, 8.0), s8, rtol=0, atol=1e-12)
print("n_in per val text:", dict(zip(*np.unique((Pin & np.isfinite(P)).sum(0), return_counts=True))),
      "| n_out:", dict(zip(*np.unique((~Pin & np.isfinite(P)).sum(0), return_counts=True))))

GRID = [(8.0, 6.0), (6.0, 4.0)]          # (κ_in, κ_out); fixed before any validation score

def cutoff_band(s, base, lo=767, hi=776):
    """Top-k set changes for every k in the band where the 76.7% cutoff can sit (767 members + ≤9 non-members)."""
    rs, rb = rankdata(-s, method="ordinal"), rankdata(-base, method="ordinal")
    changed = [k for k in range(lo, hi + 1) if set(np.flatnonzero(rs <= k)) != set(np.flatnonzero(rb <= k))]
    kth = np.sort(s)[::-1]
    ties = int(sum((np.isclose(s, kth[k - 1], rtol=0, atol=1e-12)).sum() > 1 for k in range(lo, hi + 1)))
    return {"top-k sets changed for k in 767–776": f"{len(changed)}/10 {changed[:10]}", "ties at k in 767–776": ties,
            "texts entering/leaving top-776": int(((rs <= hi) != (rb <= hi)).sum())}
P_dev = np.vstack([np.stack([mean_loss(n) for n in NAMES4]), FIX4])
Pin_dev = np.vstack([np.stack([MEM[n]["train"] for n in NAMES4]), np.zeros(FIX4.shape, bool)])
dev6 = tpr_strict(LABELS, lr_frozen(DEV_MT, DEV_L, P_dev, Pin_dev, kappa=6.0))
rows = []
for ki, ko in GRID:
    name = f"S_in{ki:g}_out{ko:g}"
    r = {"candidate": name, "dev TPR": tpr_strict(LABELS, lr_split(DEV_MT, DEV_L, P_dev, Pin_dev, ki, ko)), "dev κ=6": dev6}
    dm = []
    for t in ("p4A", "p4B", "p5A", "p5B", "p6A", "p6B"):
        keep = [n for n in NAMES4 if n not in (t[:2] + "A", t[:2] + "B")]
        Pm = np.vstack([np.stack([mean_loss(n) for n in keep]), FIX4])
        Pmin = np.vstack([np.stack([MEM[n]["train"] for n in keep]), np.zeros(FIX4.shape, bool)])
        y = MEM[t]["train"].astype(int)
        dm.append(tpr_strict(y, lr_split(mean_loss(t), DEV_L, Pm, Pmin, ki, ko)) - tpr_strict(y, lr_frozen(mean_loss(t), DEV_L, Pm, Pmin, kappa=6.0)))
    r["mock Δ vs κ=6 (p4A..p6B)"] = " ".join(f"{d:+.3f}" for d in dm); r["mock mean Δ"] = float(np.mean(dm))
    s = lr_split(MT_VAL, L_VAL, P, Pin, ki, ko)
    r.update({f"val {k}": v for k, v in near_cutoff(s, s6).items()})
    r.update({f"val {k}": v for k, v in cutoff_band(s, s6).items()})
    pred = ROOT / "data" / "val" / f"prediction_{name}.csv"
    zp = ROOT / f"project_phase2_val_submission_{name}.zip"
    np.savetxt(pred, s, fmt="%.17g")
    for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", "val",
                 "--prediction", str(pred), "--output", str(zp)],
                [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", "val"]):
        rr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); assert rr.returncode == 0, rr.stderr[-800:]
    assert np.allclose(zip_scores(zp), s, rtol=0, atol=1e-9)
    shutil.copy(zp, OUTB); r["zip"] = zp.name
    rows.append(r)
print("Against raw κ=6 (val diagnostics are vs raw κ=6 ranks):")
print(pd.DataFrame(rows).T.to_string(header=False))

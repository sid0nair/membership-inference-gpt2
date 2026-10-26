# V-KLOW — LR4 with weaker shrinkage (κ=4, κ=6), plus C3 references at κ=4. Run right after V-BATCH (same runtime).
# Chosen from validation feedback (κ=8 76.6 > κ=16 73.9 > κ=32 72.9): leaderboard optimization, not independent evidence.
#   K6       LR4 refs, κ=6
#   K4       LR4 refs, κ=4
#   C3_K4    train target as IN on its 39 val texts (C3 tied LR4 at 76.6), κ=4
for v in ("build", "MT_VAL", "L_VAL", "s8", "zip_scores", "OUTB", "NAMES4"):
    assert v in globals(), f"run V-BATCH first ({v} missing)"
assert NAMES4 == ["inA", "inB", "p2A", "p2B", "p3A", "p3B", "p4A", "p4B"]
P, Pin = build()
assert np.abs(lr_frozen(MT_VAL, L_VAL, P, Pin, kappa=8.0) - s8).max() < 1e-12, "LR4 not reproduced; stop"

# Dev and fresh-mock context for the same κ values (four pairs; mocks exclude their own pair). Information only.
P_dev = np.vstack([np.stack([mean_loss(n) for n in NAMES4]), FIX4])
Pin_dev = np.vstack([np.stack([MEM[n]["train"] for n in NAMES4]), np.zeros(FIX4.shape, bool)])
ctx = []
for k in (4.0, 6.0, 8.0):
    row = {"κ": k, "dev TPR": tpr_strict(LABELS, lr_frozen(DEV_MT, DEV_L, P_dev, Pin_dev, kappa=k))}
    for t in ("p4A", "p4B"):
        keep = [n for n in NAMES4 if n[:2] != "p4"]
        Pm = np.vstack([np.stack([mean_loss(n) for n in keep]), FIX4])
        Pmin = np.vstack([np.stack([MEM[n]["train"] for n in keep]), np.zeros(FIX4.shape, bool)])
        row[t] = tpr_strict(MEM[t]["train"].astype(int), lr_frozen(mean_loss(t), DEV_L, Pm, Pmin, kappa=k))
    ctx.append(row)
print("Context (not used for selection):"); print(pd.DataFrame(ctx).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

CANDS_LOW = {"K6_LR4_k6": dict(kappa=6.0), "K4_LR4_k4": dict(kappa=4.0), "C3K4_trainIN_k4": dict(kappa=4.0, train_as_in=True)}
top = lambda s, k: set(np.argsort(-s)[:k])
rows = []
for name, cfg in CANDS_LOW.items():
    P, Pin = build(False, cfg.get("train_as_in", False))
    s = lr_frozen(MT_VAL, L_VAL, P, Pin, kappa=cfg["kappa"])
    assert s.shape == (2000,) and np.isfinite(s).all()
    pred = ROOT / "data" / "val" / f"prediction_{name}.csv"
    zp = ROOT / f"project_phase2_val_submission_{name}.zip"
    np.savetxt(pred, s, fmt="%.17g")
    for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", "val",
                 "--prediction", str(pred), "--output", str(zp)],
                [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", "val"]):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-800:]
    assert np.allclose(zip_scores(zp), s, rtol=0, atol=1e-9)
    shutil.copy(zp, OUTB)
    rows.append({"candidate": name, "κ": cfg["kappa"], "train as IN": cfg.get("train_as_in", False),
                 "corr vs LR4": np.corrcoef(s, s8)[0, 1], "top-50 overlap": len(top(s, 50) & top(s8, 50)),
                 "top-750 overlap": len(top(s, 750) & top(s8, 750)), "zip": zp.name})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

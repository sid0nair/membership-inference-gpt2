# V-TRIM — Trimmed-mean Gaussian LR: drop each model's highest-loss tokens per text (Min-K showed they add cross-model
# noise: MINK20 dev 27.7 vs 71.0). Per model and text independently: mean of the smallest T − floor(q·T) token losses,
# × √T with the original T. Same four pairs, references, masks, κ=6, (1+1/n); pooled variances re-estimated.
# Fixed candidates: TRIM10 (q=0.10), TRIM20 (q=0.20). q=0 must reproduce mean-loss LR. Run after V-MINK (same runtime).
for v in ("REF", "TGT", "DEV_FIX_TOK", "VT", "VR", "in_tf_val_", "PIN_V", "s6", "paired", "near_cutoff", "cutoff_band",
          "zip_scores", "OUTB", "NAMES4", "KAPPA"):
    assert v in globals(), f"run V-MINK first ({v} missing)"
assert NAMES4 == ["inA", "inB", "p2A", "p2B", "p3A", "p3B", "p4A", "p4B"] and KAPPA == 6.0

def trim(arrs, q):
    out = np.empty(len(arrs))
    for i, a in enumerate(arrs):
        drop = int(np.floor(q * a.size))
        out[i] = np.partition(a, a.size - drop - 1)[:a.size - drop].mean() if drop else a.mean()
    return out
_chk = [np.array([1., 5, 3, 2, 4, 9, 8, 7, 6, 10]), np.array([2.0, 7.0, 1.0])]
assert np.allclose(trim(_chk, 0.1), [5.0, 10 / 3]) and np.allclose(trim(_chk, 0.2), [4.5, 10 / 3])
assert np.allclose(trim(_chk, 0.0), [a.mean() for a in _chk])

PIN_DEV4 = np.vstack([np.stack([MEM[n]["train"] for n in NAMES4]), np.zeros((4, 2000), bool)])
def dev_score(tstat_fn, target_arrs, names, q):
    P = np.vstack([np.stack([tstat_fn(REF[n]["train"], q) for n in names]), np.stack([tstat_fn(a, q) for a in DEV_FIX_TOK])])
    Pin = np.vstack([np.stack([MEM[n]["train"] for n in names]), np.zeros((4, 2000), bool)])
    return lr_frozen(tstat_fn(target_arrs, q), DEV_L, P, Pin, kappa=KAPPA)

dm = dev_score(trim, TGT["train"], NAMES4, 0.0)
assert np.isclose(tpr_strict(LABELS, dm), 0.710, atol=5e-4), "q=0 must reproduce dev κ=6 (0.710)"
print("q=0 check passed (dev 0.710).")

def val_score(q):
    rows = [trim(REF[n]["val"], q) for n in NAMES4] + [trim(VR[k], q) for k in ("s1", "s2", "fin", "trn")]
    rows[-1] = np.where(in_tf_val_, np.nan, rows[-1])
    return lr_frozen(trim(VT, q), L_VAL, np.stack(rows), PIN_V, kappa=KAPPA)
assert np.allclose(val_score(0.0), s6, rtol=0, atol=1e-9), "q=0 must reproduce raw κ=6 on val"

out = []
for name, q in (("TRIM10", 0.10), ("TRIM20", 0.20)):
    dv = dev_score(trim, TGT["train"], NAMES4, q)
    r = {"candidate": name, "dev TPR": tpr_strict(LABELS, dv), "dev Δ [95% CI]": paired(dv, dm, LABELS, 20261017),
         "dev AUC": roc_auc_score(LABELS, dv), "dev AUC mean-loss": roc_auc_score(LABELS, dm)}
    md, mci = [], []
    for j, t in enumerate(("p4A", "p4B", "p5A", "p5B", "p6A", "p6B")):
        keep = [n for n in NAMES4 if n not in (t[:2] + "A", t[:2] + "B")]
        y = MEM[t]["train"].astype(int)
        v0, v1 = dev_score(trim, REF[t]["train"], keep, 0.0), dev_score(trim, REF[t]["train"], keep, q)
        md.append(tpr_strict(y, v1) - tpr_strict(y, v0)); mci.append(f"{t} {paired(v1, v0, y, 20261026 + j)}")
    r.update({"mock paired Δ [CI]": " | ".join(mci), "mock mean Δ": float(np.mean(md)),
              "mock wins/losses": f"{sum(d > 0 for d in md)}/{sum(d < 0 for d in md)}"})
    s = val_score(q)
    assert s.shape == (2000,) and np.isfinite(s).all()
    r["corr vs raw κ=6 (val)"] = float(np.corrcoef(s, s6)[0, 1])
    r.update({f"val {k}": v for k, v in near_cutoff(s, s6).items()})
    r.update({f"val {k}": v for k, v in cutoff_band(s, s6).items()})
    zname = f"TR_{name}_k6"
    pred = ROOT / "data" / "val" / f"prediction_{zname}.csv"; zp = ROOT / f"project_phase2_val_submission_{zname}.zip"
    np.savetxt(pred, s, fmt="%.17g")
    for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", "val",
                 "--prediction", str(pred), "--output", str(zp)],
                [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", "val"]):
        rr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); assert rr.returncode == 0, rr.stderr[-800:]
    assert np.allclose(zip_scores(zp), s, rtol=0, atol=1e-9)
    shutil.copy(zp, OUTB); r["zip"] = zp.name
    out.append(r)
print("Against raw κ=6 mean-loss LR (val 76.7):"); print(pd.DataFrame(out).T.to_string(header=False))

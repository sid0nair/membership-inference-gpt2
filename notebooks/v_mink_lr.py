# V-MINK — Min-K Gaussian LR. Per model and text independently: mean of the largest max(1, floor(0.2·T)) token losses,
# × √T with the original token count T. Same four pairs, fixed references, masks, κ=6 and (1+1/n); pooled variances are
# re-estimated for this statistic. Three fixed scores: mean-loss LR (incumbent, val 76.7), Min-K LR, and
# 0.5·mean-loss LR + 0.5·Min-K LR (heuristic; the two are correlated, so this is not a joint likelihood ratio).
# Run after V-BATCH, V-MONO, V-SPLIT (same runtime). Cached per-token losses only. Leaderboard tuning, fixed in advance.
for v in ("build", "MT_VAL", "L_VAL", "zip_scores", "OUTB", "NAMES4", "near_cutoff", "cutoff_band", "VT", "VR"):
    assert v in globals(), f"run V-BATCH, V-MONO and V-SPLIT first ({v} missing)"
assert NAMES4 == ["inA", "inB", "p2A", "p2B", "p3A", "p3B", "p4A", "p4B"]
KAPPA, FRAC = 6.0, 0.20

def mink(arrs, frac=FRAC):
    out = np.empty(len(arrs))
    for i, a in enumerate(arrs):
        k = max(1, int(np.floor(frac * a.size)))
        out[i] = np.partition(a, a.size - k)[a.size - k:].mean()
    return out

def stats(arrs):
    return np.array([a.mean() for a in arrs]), mink(arrs)
_chk = [np.array([1., 5, 3, 2, 4]), np.full(10, 0.5), np.arange(1, 101, dtype=float), np.array([2.0, 7.0, 1.0])]
assert np.allclose(mink(_chk), [5.0, 0.5, 90.5, 7.0])                       # floor(0.2·3)=0 → k=1
assert np.allclose(mink(_chk, 1.0), [a.mean() for a in _chk])               # K=100% is the mean

# ---------- validation statistics, built exactly like build() but for both statistics ----------
in_tf_val_ = np.array([norm(t) in {norm(txt(r)) for r in read_json(ROOT / "data" / "train" / "train_finetune.json")}
                       for t in TEXTS["val"]]) if "in_tf_val" not in globals() else in_tf_val
VAL_T = stats(VT)
VAL_REFS = [stats(REF[n]["val"]) for n in NAMES4] + [stats(VR["s1"]), stats(VR["s2"]), stats(VR["fin"]), stats(VR["trn"])]
def val_matrix(j):
    rows = [r[j] for r in VAL_REFS]
    rows[-1] = np.where(in_tf_val_, np.nan, rows[-1])              # train target masked on its 39 val texts
    return np.stack(rows)
PIN_V = np.vstack([np.stack([MEM[n]["val"] for n in NAMES4]), np.zeros((4, 2000), bool)])
P_mean_v, _ = build()
assert np.allclose(np.nan_to_num(val_matrix(0), nan=-1), np.nan_to_num(P_mean_v, nan=-1), rtol=0, atol=1e-12), "val refs differ from build()"
s6 = lr_frozen(MT_VAL, L_VAL, P_mean_v, PIN_V, kappa=KAPPA)

# ---------- dev and fresh mocks ----------
DEV_FIX_TOK = [split_npz(OLD / "train_shadow_1001_losses.npz", "shadow_losses"), split_npz(OLD / "train_shadow_1002_losses.npz"),
               split_npz(OLD / "train_ref_val_losses.npz"), split_npz(OLD / "train_ref_final_losses.npz")]
DEV_FIX = [stats(a) for a in DEV_FIX_TOK]
assert np.allclose(np.stack([d[0] for d in DEV_FIX]), FIX4, rtol=0, atol=1e-12)
REFSTAT = {n: stats(REF[n]["train"]) for n in ["inA", "inB", "p2A", "p2B", "p3A", "p3B", "p4A", "p4B", "p5A", "p5B", "p6A", "p6B"] if n in REF}
DEV_T = stats(TGT["train"])

def dev_like(target_stats, names, j):
    P = np.vstack([np.stack([REFSTAT[n][j] for n in names]), np.stack([d[j] for d in DEV_FIX])])
    Pin = np.vstack([np.stack([MEM[n]["train"] for n in names]), np.zeros((4, 2000), bool)])
    return lr_frozen(target_stats[j], DEV_L, P, Pin, kappa=KAPPA)

def cands(lr_mean, lr_k):
    return {"MINK20": lr_k, "AVG": 0.5 * lr_mean + 0.5 * lr_k}

def paired(a, b, y, seed, B=2000):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); g = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        ix = np.concatenate([g.choice(pos, pos.size), g.choice(neg, neg.size)])
        d.append(strict_fast(a[ix], y[ix]) - strict_fast(b[ix], y[ix]))
    lo, hi = np.percentile(d, [2.5, 97.5]); return f"{tpr_strict(y, a) - tpr_strict(y, b):+.3f} [{lo:+.3f}, {hi:+.3f}]"

rows = []
dm, dk = dev_like(DEV_T, NAMES4, 0), dev_like(DEV_T, NAMES4, 1)
assert np.isclose(tpr_strict(LABELS, dm), 0.710, atol=5e-4), "dev κ=6 raw should be 0.710"
# Implementation check: K=100% through the Min-K path must reproduce mean-loss LR.
_full = lambda arrs: mink(arrs, 1.0)
P100 = np.vstack([np.stack([_full(REF[n]["train"]) for n in NAMES4]), np.stack([_full(a) for a in DEV_FIX_TOK])])
assert np.allclose(lr_frozen(_full(TGT["train"]), DEV_L, P100,
                             np.vstack([np.stack([MEM[n]["train"] for n in NAMES4]), np.zeros((4, 2000), bool)]), kappa=KAPPA),
                   dm, rtol=0, atol=1e-9), "K=100% does not reproduce mean-loss LR"
print("K=100% check passed.")
dev_c = cands(dm, dk)
mock_d = {k: [] for k in dev_c}; mock_ci = {k: [] for k in dev_c}; mock_auc = {k: [] for k in dev_c}
for t in ("p4A", "p4B", "p5A", "p5B", "p6A", "p6B"):
    if t not in REFSTAT:
        continue
    keep = [n for n in NAMES4 if n not in (t[:2] + "A", t[:2] + "B")]
    y = MEM[t]["train"].astype(int)
    m0, m1 = dev_like(REFSTAT[t], keep, 0), dev_like(REFSTAT[t], keep, 1)
    for k, v in cands(m0, m1).items():
        mock_d[k].append(tpr_strict(y, v) - tpr_strict(y, m0))
        mock_ci[k].append(f"{t} {paired(v, m0, y, 20261026 + len(mock_ci[k]))}")
        mock_auc[k].append(roc_auc_score(y, v) - roc_auc_score(y, m0))
lr_mean_v = lr_frozen(VAL_T[0], L_VAL, val_matrix(0), PIN_V, kappa=KAPPA)
assert np.allclose(lr_mean_v, s6, rtol=0, atol=1e-12)
lr_k_v = lr_frozen(VAL_T[1], L_VAL, val_matrix(1), PIN_V, kappa=KAPPA)
for name, s in cands(lr_mean_v, lr_k_v).items():
    assert s.shape == (2000,) and np.isfinite(s).all()
    r = {"candidate": name, "dev TPR": tpr_strict(LABELS, dev_c[name]), "dev κ=6 mean-loss": 0.710,
         "dev Δ [95% CI]": paired(dev_c[name], dm, LABELS, 20261017),
         "dev AUC": roc_auc_score(LABELS, dev_c[name]), "dev AUC mean-loss": roc_auc_score(LABELS, dm),
         "mock paired Δ [CI]": " | ".join(mock_ci[name]), "mock mean AUC Δ": float(np.mean(mock_auc[name])),
         "mock Δ vs κ=6 (p4A..p6B)": " ".join(f"{d:+.3f}" for d in mock_d[name]), "mock mean Δ": float(np.mean(mock_d[name])),
         "corr(LR_mean, LR_mink) val": float(np.corrcoef(lr_mean_v, lr_k_v)[0, 1])}
    r.update({f"val {k}": v for k, v in near_cutoff(s, s6).items()})
    r.update({f"val {k}": v for k, v in cutoff_band(s, s6).items()})
    zname = f"MK_{name}_k6"
    pred = ROOT / "data" / "val" / f"prediction_{zname}.csv"; zp = ROOT / f"project_phase2_val_submission_{zname}.zip"
    np.savetxt(pred, s, fmt="%.17g")
    for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", "val",
                 "--prediction", str(pred), "--output", str(zp)],
                [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", "val"]):
        rr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); assert rr.returncode == 0, rr.stderr[-800:]
    assert np.allclose(zip_scores(zp), s, rtol=0, atol=1e-9)
    shutil.copy(zp, OUTB); r["zip"] = zp.name
    rows.append(r)
print("Against raw κ=6 mean-loss LR (val 76.7):"); print(pd.DataFrame(rows).T.to_string(header=False))

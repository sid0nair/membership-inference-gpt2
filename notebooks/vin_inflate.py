# VIN-1 — One exploratory candidate: LR4 references, κ=6, IN predictive variance × 1.2 (OUT variance and both means
# unchanged; full Gaussian log-density ratio recomputed, log-variance term included). Not a fitted correction.
# First: standardized member residuals z_IN = (u − μ_IN)/√v_IN with the scorer's actual predictive variances, real dev
# vs matched fresh mocks. Then dev / mock comparison against raw κ=6, val ZIP + cutoff diagnostics. CPU; after N-2..N-4.
import zipfile, io
from scipy.stats import rankdata
KAP, C_IN = 6.0, 1.2
PAIRS4 = [("inA", "inB"), ("p2A", "p2B"), ("p3A", "p3B"), ("p4A", "p4B")]
assert [tuple(p) for p in PAIRS] == PAIRS4
NAMES8 = [n for p in PAIRS4 for n in p]
MOCKS = ["p4A", "p4B", "p5A", "p5B", "p6A", "p6B"]
txt = lambda r: r if isinstance(r, str) else r["text"]
tmn = lambda arrs: np.array([a.mean() for a in arrs])
for n in MOCKS:
    if n not in REF:
        REF[n] = {ph: split_npz(OLD / f"{ph}_shadow_{n}_losses.npz") for ph in PHASES}
    if n not in MEM:
        MEM[n] = membership(shadow_texts(n))

def lr_vin(mt, L, P, Pin, kappa=KAP, c_in=1.0, parts=False):
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
    s_in = (rss_in + kappa * pool_in) / (df_in + kappa) * (1 + 1 / n_in) * c_in       # final IN predictive variance × c
    s_out = (rss_out + kappa * pool_out) / (df_out + kappa) * (1 + 1 / n_out)
    assert (s_in > 0).all() and (s_out > 0).all()
    A = (-(u - mu_in) ** 2 / (2 * s_in) - 0.5 * np.log(s_in)) - (-(u - mu_out) ** 2 / (2 * s_out) - 0.5 * np.log(s_out))
    assert np.isfinite(A).all()
    return (A, (u - mu_in) / np.sqrt(s_in), (u - mu_out) / np.sqrt(s_out)) if parts else A

def dev_P(names):
    return (np.vstack([np.stack([mean_loss(n) for n in names]), FIX4]),
            np.vstack([np.stack([MEM[n]["train"] for n in names]), np.zeros(FIX4.shape, bool)]))
P, Pin = dev_P(NAMES8)
assert np.allclose(lr_vin(DEV_MT, DEV_L, P, Pin, 8.0), LR4_DEV, rtol=0, atol=1e-6)
assert np.allclose(lr_vin(DEV_MT, DEV_L, P, Pin), lr_frozen(DEV_MT, DEV_L, P, Pin, kappa=KAP), rtol=0, atol=1e-12)

# ---------- 1. Standardized member residuals (raw κ=6 predictive variances) ----------
def zsum(z):
    q = np.percentile(z, [5, 25, 50, 75, 95])
    return {"n": z.size, "mean": z.mean(), "sd": z.std(ddof=1), "q05": q[0], "q25": q[1], "median": q[2], "q75": q[3],
            "q95": q[4], "|z|>2": np.mean(np.abs(z) > 2), "|z|>3": np.mean(np.abs(z) > 3)}
zrows = []
for t in ["real dev"] + MOCKS:
    excl = () if t == "real dev" else (t[:2] + "A", t[:2] + "B")
    names = [n for n in NAMES8 if n not in excl]
    Pp, Pinp = dev_P(names)
    mt, y = (DEV_MT, LABELS) if t == "real dev" else (mean_loss(t), MEM[t]["train"].astype(int))
    _, zin, zout = lr_vin(mt, DEV_L, Pp, Pinp, parts=True)
    pool = "pairs 1–3" if t[:2] == "p4" else "pairs 1–4"
    zrows.append({"target": t, "pool": pool, "class": "IN (members)", **zsum(zin[y == 1])})
    zrows.append({"target": t, "pool": pool, "class": "OUT (non-members)", **zsum(zout[y == 0])})
    if t == "real dev":                                                           # matched pool for the p4 mocks
        Pp3, Pin3 = dev_P([n for n in NAMES8 if n[:2] != "p4"])
        _, zin3, zout3 = lr_vin(DEV_MT, DEV_L, Pp3, Pin3, parts=True)
        zrows.append({"target": "real dev", "pool": "pairs 1–3", "class": "IN (members)", **zsum(zin3[LABELS == 1])})
        zrows.append({"target": "real dev", "pool": "pairs 1–3", "class": "OUT (non-members)", **zsum(zout3[LABELS == 0])})
pd.set_option("display.width", 250)
print("Standardized residuals with the scorer's predictive variances (raw κ=6). A well-calibrated side has sd ≈ 1:")
print(pd.DataFrame(zrows).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# ---------- 2. Candidate vs raw κ=6 on dev and fresh mocks ----------
def pboot(a, b, y, seed, B=2000):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); g = np.random.default_rng(seed); d = []
    for _ in range(B):
        ix = np.concatenate([g.choice(pos, pos.size), g.choice(neg, neg.size)])
        d.append(strict_fast(a[ix], y[ix]) - strict_fast(b[ix], y[ix]))
    lo, hi = np.percentile(d, [2.5, 97.5]); return f"{tpr_strict(y, a) - tpr_strict(y, b):+.3f} [{lo:+.3f}, {hi:+.3f}]"
def accepted(s, y):
    neg = np.sort(s[y == 0])[::-1]; thr = neg[int(np.ceil(0.01 * neg.size)) - 1]
    return set(np.flatnonzero(s > thr))

base = lr_vin(DEV_MT, DEV_L, P, Pin); cand = lr_vin(DEV_MT, DEV_L, P, Pin, c_in=C_IN)
A0, A1 = accepted(base, LABELS), accepted(cand, LABELS)
enter, leave = sorted(A1 - A0), sorted(A0 - A1)
print(f"\nDev: strict TPR {tpr_strict(LABELS, base):.4f} → {tpr_strict(LABELS, cand):.4f} | Δ [CI] {pboot(cand, base, LABELS, 20261017)} "
      f"| AUC {roc_auc_score(LABELS, base):.4f} → {roc_auc_score(LABELS, cand):.4f}")
print(f"Dev accepted set: {len(enter)} enter {[(i, int(LABELS[i])) for i in enter][:20]} | "
      f"{len(leave)} leave {[(i, int(LABELS[i])) for i in leave][:20]}  (index, label)")
mrows = []
for j, t in enumerate(MOCKS):
    names = [n for n in NAMES8 if n not in (t[:2] + "A", t[:2] + "B")]
    Pp, Pinp = dev_P(names); y = MEM[t]["train"].astype(int)
    b0, b1 = lr_vin(mean_loss(t), DEV_L, Pp, Pinp), lr_vin(mean_loss(t), DEV_L, Pp, Pinp, c_in=C_IN)
    mrows.append({"mock": t, "TPR κ=6": tpr_strict(y, b0), "TPR ×1.2": tpr_strict(y, b1), "Δ [CI]": pboot(b1, b0, y, 20261026 + j),
                  "AUC κ=6": roc_auc_score(y, b0), "AUC ×1.2": roc_auc_score(y, b1)})
mt_ = pd.DataFrame(mrows); d_ = mt_["TPR ×1.2"] - mt_["TPR κ=6"]
print(mt_.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"mock mean Δ {d_.mean():+.4f} | wins {int((d_ > 0).sum())} / losses {int((d_ < 0).sum())} / ties {int((d_ == 0).sum())}")

# ---------- 3. Validation scores and cutoff diagnostics vs raw κ=6 (ZIP built, not uploaded) ----------
train_ft = {norm(txt(r)) for r in read_json(ROOT / "data" / "train" / "train_finetune.json")}
in_tf_val = np.array([norm(t) in train_ft for t in TEXTS["val"]]); assert int(in_tf_val.sum()) == 39
VT = split_npz(OLD / "val_shadow_1001_results.npz", "target_losses")
VR = {"s1": split_npz(OLD / "val_shadow_1001_results.npz", "shadow_losses"), "s2": split_npz(OLD / "val_shadow_1002_losses.npz"),
      "fin": split_npz(OLD / "val_ref_final_losses.npz"), "trn": split_npz(OLD / "val_ref_train_losses.npz")}
FIXV = np.stack([tmn(VR["s1"]), tmn(VR["s2"]), tmn(VR["fin"]), np.where(in_tf_val, np.nan, tmn(VR["trn"]))])
PV = np.vstack([np.stack([tmn(REF[n]["val"]) for n in NAMES8]), FIXV])
PINV = np.vstack([np.stack([MEM[n]["val"] for n in NAMES8]), np.zeros(FIXV.shape, bool)])
MTV, LV = tmn(VT), np.array([a.size for a in VT], float)
sub4 = np.loadtxt(io.BytesIO(zipfile.ZipFile(DRIVE / "frozen_LR4_k8" / "project_phase2_val_submission_LR4_k8.zip").read("val/prediction.csv")))
assert np.abs(lr_vin(MTV, LV, PV, PINV, 8.0) - sub4).max() < 1e-4, "LR4 val not reproduced"
v0, v1 = lr_vin(MTV, LV, PV, PINV), lr_vin(MTV, LV, PV, PINV, c_in=C_IN)
rs, rb = rankdata(-v1, method="ordinal"), rankdata(-v0, method="ordinal")
print(f"\nVal vs raw κ=6: corr {np.corrcoef(v1, v0)[0, 1]:.5f} | top-700/750/800 overlap "
      f"{[len(set(np.argsort(-v1)[:k]) & set(np.argsort(-v0)[:k])) for k in (700, 750, 800)]} | "
      f"top-k sets changed for k in 767–776: {sum(set(np.flatnonzero(rs <= k)) != set(np.flatnonzero(rb <= k)) for k in range(767, 777))}/10 "
      f"| entering/leaving top-776: {int(((rs <= 776) != (rb <= 776)).sum())} | mean |rank Δ| in 650–900: "
      f"{np.abs(rs - rb)[(rb >= 650) & (rb <= 900)].mean():.2f}")
tag = "VIN12_k6"
pred = ROOT / "data" / "val" / f"prediction_{tag}.csv"; zp = ROOT / f"project_phase2_val_submission_{tag}.zip"
np.savetxt(pred, v1, fmt="%.17g")
for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", "val", "--prediction", str(pred), "--output", str(zp)],
            [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", "val"]):
    rr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); assert rr.returncode == 0, rr.stderr[-800:]
(DRIVE / "val_candidates_vin").mkdir(exist_ok=True); shutil.copy(zp, DRIVE / "val_candidates_vin")
print("Built", zp.name, "(not uploaded).")

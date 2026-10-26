# V-DROP — LR at κ=6 with one of LR4's four pairs removed (all four leave-one-pair-out subsets). Cached losses, CPU.
# Motivation (validation feedback): val fell as pairs were added (4: 76.6/76.7, 5: 75.0/73.5, 6: 71.3/70.6).
# Leaderboard search over 4 fixed subsets; not independent evidence. Run after N-2..N-4. Incumbent: raw LR4 κ=6 (76.7).
import zipfile, io
from scipy.stats import rankdata
KAP = 6.0
PAIRS4 = [("inA", "inB"), ("p2A", "p2B"), ("p3A", "p3B"), ("p4A", "p4B")]
assert [tuple(p) for p in PAIRS] == PAIRS4
txt = lambda r: r if isinstance(r, str) else r["text"]
tmn = lambda arrs: np.array([a.mean() for a in arrs])
OUTD = DRIVE / "val_candidates_drop"; OUTD.mkdir(exist_ok=True)

train_ft = {norm(txt(r)) for r in read_json(ROOT / "data" / "train" / "train_finetune.json")}
in_tf_val = np.array([norm(t) in train_ft for t in TEXTS["val"]]); assert int(in_tf_val.sum()) == 39
VT = split_npz(OLD / "val_shadow_1001_results.npz", "target_losses")
VR = {"s1": split_npz(OLD / "val_shadow_1001_results.npz", "shadow_losses"), "s2": split_npz(OLD / "val_shadow_1002_losses.npz"),
      "fin": split_npz(OLD / "val_ref_final_losses.npz"), "trn": split_npz(OLD / "val_ref_train_losses.npz")}
FIXV = np.stack([tmn(VR["s1"]), tmn(VR["s2"]), tmn(VR["fin"]), np.where(in_tf_val, np.nan, tmn(VR["trn"]))])
MTV, LV = tmn(VT), np.array([a.size for a in VT], float)

def val_lr(names, kappa=KAP):
    P = np.vstack([np.stack([tmn(REF[n]["val"]) for n in names]), FIXV])
    Pin = np.vstack([np.stack([MEM[n]["val"] for n in names]), np.zeros(FIXV.shape, bool)])
    fin = np.isfinite(P)
    return lr_frozen(MTV, LV, P, Pin, kappa=kappa), (Pin & fin).sum(0), (~Pin & fin).sum(0)

def dev_lr(mt, names, kappa=KAP):
    P = np.vstack([np.stack([mean_loss(n) for n in names]), FIX4])
    Pin = np.vstack([np.stack([MEM[n]["train"] for n in names]), np.zeros(FIX4.shape, bool)])
    return lr_frozen(mt, DEV_L, P, Pin, kappa=kappa)

ALL8 = [n for p in PAIRS4 for n in p]
sub4 = np.loadtxt(io.BytesIO(zipfile.ZipFile(DRIVE / "frozen_LR4_k8" / "project_phase2_val_submission_LR4_k8.zip").read("val/prediction.csv")))
assert np.abs(val_lr(ALL8, 8.0)[0] - sub4).max() < 1e-4, "LR4 κ=8 val not reproduced; stop"
s6, _, _ = val_lr(ALL8)
assert np.isclose(tpr_strict(LABELS, dev_lr(DEV_MT, ALL8)), 0.710, atol=5e-4)

def band(s, base, lo=767, hi=776):
    rs, rb = rankdata(-s, method="ordinal"), rankdata(-base, method="ordinal")
    return {"top-776 entering/leaving": int(((rs <= hi) != (rb <= hi)).sum()),
            "top-k sets changed 767–776": sum(set(np.flatnonzero(rs <= k)) != set(np.flatnonzero(rb <= k)) for k in range(lo, hi + 1))}

rows = []
for drop in PAIRS4:
    names = [n for p in PAIRS4 if p != drop for n in p]
    tag = f"D_no_{drop[0][:-1] if drop[0] != 'inA' else 'p1'}_k6"
    r = {"candidate": tag, "dropped": "/".join(drop), "dev TPR": tpr_strict(LABELS, dev_lr(DEV_MT, names)), "dev 4 pairs": 0.710}
    # Fresh mocks with all remaining pairs that exclude the mock's own pair (p4 mocks only when p4 is not dropped... skip own).
    md = []
    for t in ("p5A", "p5B", "p6A", "p6B"):
        if t in REF and t in MEM:
            y = MEM[t]["train"].astype(int)
            md.append(f"{t} {tpr_strict(y, dev_lr(mean_loss(t), names)) - tpr_strict(y, dev_lr(mean_loss(t), ALL8)):+.3f}")
    r["mock Δ vs 4 pairs"] = " ".join(md) if md else "p5/p6 not loaded"
    s, nin, nout = val_lr(names)
    assert np.isfinite(s).all() and (nin > 0).all() and (nout > 0).all()
    r.update({"val n_in (median)": int(np.median(nin)), "val n_out (median)": int(np.median(nout)),
              "corr vs κ=6": float(np.corrcoef(s, s6)[0, 1]), **band(s, s6)})
    pred = ROOT / "data" / "val" / f"prediction_{tag}.csv"; zp = ROOT / f"project_phase2_val_submission_{tag}.zip"
    np.savetxt(pred, s, fmt="%.17g")
    for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", "val", "--prediction", str(pred), "--output", str(zp)],
                [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", "val"]):
        rr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); assert rr.returncode == 0, rr.stderr[-800:]
    assert np.allclose(np.loadtxt(io.BytesIO(zipfile.ZipFile(zp).read("val/prediction.csv"))), s, rtol=0, atol=1e-9)
    shutil.copy(zp, OUTD); r["zip"] = zp.name
    rows.append(r)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

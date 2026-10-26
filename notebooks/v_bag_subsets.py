# V-BAG — Bag-mean over all 15 four-pair subsets of the six pairs (BAG-1 recipe: dev +3.4 vs LR4 at κ=8, never submitted).
# Each subset uses LR4's fixed references and masks; scores are averaged per text. Builds val ZIPs at κ=8 (as BAG-1)
# and κ=6. Cached losses, CPU. Run after N-2..N-4. Leaderboard attempt; incumbent raw LR4 κ=6 (76.7).
import zipfile, io
from itertools import combinations
from scipy.stats import rankdata
PAIRS6 = [("inA", "inB"), ("p2A", "p2B"), ("p3A", "p3B"), ("p4A", "p4B"), ("p5A", "p5B"), ("p6A", "p6B")]
assert [tuple(p) for p in PAIRS] == PAIRS6[:4]
txt = lambda r: r if isinstance(r, str) else r["text"]
tmn = lambda arrs: np.array([a.mean() for a in arrs])
for n in ("p5A", "p5B", "p6A", "p6B"):
    if n not in REF:
        REF[n] = {ph: split_npz(OLD / f"{ph}_shadow_{n}_losses.npz") for ph in PHASES}
    if n not in MEM:
        MEM[n] = membership(shadow_texts(n))
for a, b in PAIRS6[1:]:
    assert (MEM[a]["val"] ^ MEM[b]["val"]).all() and (MEM[a]["train"] ^ MEM[b]["train"]).all()
OUTG = DRIVE / "val_candidates_bag"; OUTG.mkdir(exist_ok=True)

train_ft = {norm(txt(r)) for r in read_json(ROOT / "data" / "train" / "train_finetune.json")}
in_tf_val = np.array([norm(t) in train_ft for t in TEXTS["val"]]); assert int(in_tf_val.sum()) == 39
VT = split_npz(OLD / "val_shadow_1001_results.npz", "target_losses")
VR = {"s1": split_npz(OLD / "val_shadow_1001_results.npz", "shadow_losses"), "s2": split_npz(OLD / "val_shadow_1002_losses.npz"),
      "fin": split_npz(OLD / "val_ref_final_losses.npz"), "trn": split_npz(OLD / "val_ref_train_losses.npz")}
FIXV = np.stack([tmn(VR["s1"]), tmn(VR["s2"]), tmn(VR["fin"]), np.where(in_tf_val, np.nan, tmn(VR["trn"]))])
MTV, LV = tmn(VT), np.array([a.size for a in VT], float)
VML = {n: tmn(REF[n]["val"]) for p in PAIRS6 for n in p}

def val_lr(names, kappa):
    P = np.vstack([np.stack([VML[n] for n in names]), FIXV])
    Pin = np.vstack([np.stack([MEM[n]["val"] for n in names]), np.zeros(FIXV.shape, bool)])
    return lr_frozen(MTV, LV, P, Pin, kappa=kappa)
def dev_lr(mt, names, kappa):
    P = np.vstack([np.stack([mean_loss(n) for n in names]), FIX4])
    Pin = np.vstack([np.stack([MEM[n]["train"] for n in names]), np.zeros(FIX4.shape, bool)])
    return lr_frozen(mt, DEV_L, P, Pin, kappa=kappa)

ALL8 = [n for p in PAIRS6[:4] for n in p]
sub4 = np.loadtxt(io.BytesIO(zipfile.ZipFile(DRIVE / "frozen_LR4_k8" / "project_phase2_val_submission_LR4_k8.zip").read("val/prediction.csv")))
assert np.abs(val_lr(ALL8, 8.0) - sub4).max() < 1e-4, "LR4 κ=8 val not reproduced; stop"
s6 = val_lr(ALL8, 6.0)
SUBS = list(combinations(range(6), 4)); assert len(SUBS) == 15 and (0, 1, 2, 3) in SUBS

def band(s, base, hi=776):
    rs, rb = rankdata(-s, method="ordinal"), rankdata(-base, method="ordinal")
    return int(((rs <= hi) != (rb <= hi)).sum())

rows = []
for k in (8.0, 6.0):
    dev_bag = np.mean([dev_lr(DEV_MT, [n for j in S for n in PAIRS6[j]], k) for S in SUBS], 0)
    val_bag = np.mean([val_lr([n for j in S for n in PAIRS6[j]], k) for S in SUBS], 0)
    assert np.isfinite(val_bag).all() and val_bag.shape == (2000,)
    tag = f"BAG15_k{k:g}"
    rows.append({"candidate": tag, "dev TPR": tpr_strict(LABELS, dev_bag), "dev LR4 same κ": tpr_strict(LABELS, dev_lr(DEV_MT, ALL8, k)),
                 "dev AUC": roc_auc_score(LABELS, dev_bag), "corr vs raw κ=6 (val)": float(np.corrcoef(val_bag, s6)[0, 1]),
                 "top-776 entering/leaving vs κ=6": band(val_bag, s6)})
    pred = ROOT / "data" / "val" / f"prediction_{tag}.csv"; zp = ROOT / f"project_phase2_val_submission_{tag}.zip"
    np.savetxt(pred, val_bag, fmt="%.17g")
    for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", "val", "--prediction", str(pred), "--output", str(zp)],
                [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", "val"]):
        rr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); assert rr.returncode == 0, rr.stderr[-800:]
    assert np.allclose(np.loadtxt(io.BytesIO(zipfile.ZipFile(zp).read("val/prediction.csv"))), val_bag, rtol=0, atol=1e-9)
    shutil.copy(zp, OUTG); rows[-1]["zip"] = zp.name
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

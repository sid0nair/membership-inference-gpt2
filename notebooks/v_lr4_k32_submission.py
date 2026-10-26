# V-4K32 — Frozen LR4 pipeline and reference set, only κ changed 8 → 32. Builds the validation ZIP only.
# Run after N-2, N-3, N-4 in the same runtime (KR-1 optional). CPU only. Nothing is uploaded.
# Status: leaderboard optimization chosen after validation feedback (LR6 κ=8/κ=32), not an independent evaluation.
import zipfile, io
KAPPA_NEW = 32.0
FZ4 = DRIVE / "frozen_LR4_k8"
FZ = DRIVE / "frozen_LR4_k32"
ZIP_NAME = "project_phase2_val_submission_LR4_k32.zip"
assert not (FZ / "settings.json").exists(), f"{FZ} already holds a build; inspect it before rebuilding"
FZ.mkdir(exist_ok=True)
NAMES4 = [n for p in PAIRS for n in p]
assert NAMES4 == ["inA", "inB", "p2A", "p2B", "p3A", "p3B", "p4A", "p4B"]
assert all(n in REF and "val" in REF[n] and n in MEM and "val" in MEM[n] for n in NAMES4)
tm = lambda arrs: np.array([a.mean() for a in arrs])
txt = lambda r: r if isinstance(r, str) else r["text"]

# Fixed OUT references for val, exactly as in LR4 (train target masked on the 39 val texts in its training data).
train_ft = {norm(txt(r)) for r in read_json(ROOT / "data" / "train" / "train_finetune.json")}
in_tf_val = np.array([norm(t) in train_ft for t in TEXTS["val"]])
assert int(in_tf_val.sum()) == 39
VT = split_npz(OLD / "val_shadow_1001_results.npz", "target_losses")
VR = {"s1": split_npz(OLD / "val_shadow_1001_results.npz", "shadow_losses"), "s2": split_npz(OLD / "val_shadow_1002_losses.npz"),
      "fin": split_npz(OLD / "val_ref_final_losses.npz"), "trn": split_npz(OLD / "val_ref_train_losses.npz")}
for arrs in list(VR.values()) + [REF[n]["val"] for n in NAMES4]:
    assert len(arrs) == 2000 and all(a.shape == b.shape for a, b in zip(arrs, VT))
MT_VAL, L_VAL = tm(VT), np.array([a.size for a in VT], float)
FIXED_VAL = np.stack([tm(VR["s1"]), tm(VR["s2"]), tm(VR["fin"]), np.where(in_tf_val, np.nan, tm(VR["trn"]))])
P_VAL = np.vstack([np.stack([tm(REF[n]["val"]) for n in NAMES4]), FIXED_VAL])
PIN_VAL = np.vstack([np.stack([MEM[n]["val"] for n in NAMES4]), np.zeros(FIXED_VAL.shape, bool)])
assert int(np.isnan(P_VAL).sum()) == 39, "only the 39 train-reference exclusions should be masked"

def zip_scores(path):
    with zipfile.ZipFile(path) as zf:
        return np.loadtxt(io.BytesIO(zf.read("val/prediction.csv")))

# ---------- 1. κ=8 must reproduce the SUBMITTED LR4 validation scores ----------
s8 = lr_frozen(MT_VAL, L_VAL, P_VAL, PIN_VAL, kappa=8.0)
sub4 = zip_scores(FZ4 / "project_phase2_val_submission_LR4_k8.zip")
d = float(np.abs(s8 - sub4).max())
same_rank = np.array_equal(np.argsort(-s8, kind="stable"), np.argsort(-sub4, kind="stable"))
print(f"κ=8 vs submitted LR4 ZIP: max |Δ| = {d:.2e} | identical ranking: {same_rank}")
assert d < 1e-4 and same_rank, "LR4 not reproduced; stop"

# ---------- 2. κ=32 must reproduce dev strict TPR 0.730 ----------
P_DEV = np.vstack([np.stack([mean_loss(n) for n in NAMES4]), FIX4])
PIN_DEV = np.vstack([np.stack([MEM[n]["train"] for n in NAMES4]), np.zeros(FIX4.shape, bool)])
assert np.allclose(lr_frozen(DEV_MT, DEV_L, P_DEV, PIN_DEV, kappa=8.0), LR4_DEV, rtol=0, atol=1e-6)
dev32 = lr_frozen(DEV_MT, DEV_L, P_DEV, PIN_DEV, kappa=KAPPA_NEW)
dev_tpr = tpr_strict(LABELS, dev32)
print(f"dev κ=32: strict TPR {dev_tpr:.4f} | AUC {roc_auc_score(LABELS, dev32):.4f}")
assert np.isclose(dev_tpr, 0.730, atol=5e-4), "dev TPR does not match KR-1 (0.730); stop"
if "SC" in globals() and "real dev|4 pairs, κ=32" in SC:
    assert np.allclose(dev32, SC["real dev|4 pairs, κ=32"], rtol=0, atol=1e-12)

# ---------- 3. Score val with κ=32 and export ----------
s32 = lr_frozen(MT_VAL, L_VAL, P_VAL, PIN_VAL, kappa=KAPPA_NEW)
assert s32.shape == (2000,) and np.isfinite(s32).all()
top = lambda s, k=750: set(np.argsort(-s)[:k])
print(f"val κ=32 vs κ=8: corr {np.corrcoef(s32, s8)[0, 1]:.5f} | top-750 overlap {len(top(s32) & top(s8))} "
      f"| top-50 overlap {len(top(s32, 50) & top(s8, 50))}")
pred = ROOT / "data" / "val" / "prediction_LR4_k32.csv"
zp = ROOT / ZIP_NAME
assert not zp.exists() or zp.resolve() != (ROOT / "project_phase2_val_submission_LR4_k8.zip").resolve()
np.savetxt(pred, s32, fmt="%.17g")
for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", "val",
             "--prediction", str(pred), "--output", str(zp)],
            [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", "val"]):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()); assert r.returncode == 0
assert np.allclose(zip_scores(zp), s32, rtol=0, atol=1e-9)
shutil.copy(zp, FZ); shutil.copy(zp, DRIVE)
np.savez(FZ / "val_scores.npz", val_LR4_k32=s32)
assert (FZ4 / "project_phase2_val_submission_LR4_k8.zip").exists(), "original LR4 submission missing"

json.dump({"method": "frozen LR4 pipeline and references (8 paired shadows + shadow1001, shadow1002, final target, "
                     "train target with 39 val texts masked); only κ changed 8 → 32",
           "kappa": KAPPA_NEW, "dev": {"LR4_k8": 0.712, "LR4_k32": round(dev_tpr, 4)},
           "fresh_mock_mean_TPR": {"LR4_k8": 0.7399, "LR4_k32": 0.7486},
           "fresh_mock_deltas": {"p4A": 0.006, "p4B": -0.004, "p5A": 0.0, "p5B": 0.019, "p6A": 0.0, "p6B": 0.031},
           "selection_status": "chosen after validation feedback on LR6 κ=8 and κ=32; leaderboard optimization, "
                               "not an independent evaluation; revises KR-1's highest-mock-mean rule",
           "validation_submission": "built, not uploaded"}, open(FZ / "settings.json", "w"), indent=2)
print("Built", zp, "| copy in", FZ, "| original LR4 ZIP untouched.")

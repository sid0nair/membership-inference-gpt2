# V-BATCH — Several LR4 variants as validation ZIPs, cached losses only (CPU). Run after N-2, N-3, N-4.
# Leaderboard optimization: candidates were fixed before seeing their validation scores; not an independent evaluation.
#
# Candidates (submit in this order):
#   C1  LR4, κ=16                         four-pair 8-mock tie with κ=8 (80.25 vs 80.22)
#   C2  LR4, final target dropped         the dev target verifiably trained on 39 val + 31 final texts; by the same
#                                         construction the final target likely trained on ~30–40 val texts, which LR4
#                                         counts as OUT. Dropping it removes those mislabeled references.
#   C3  LR4, dev target as IN on its 39   those 39 val texts are verifiably in the dev target's training data; LR4 masks
#                                         them, C3 uses them as a true IN reference instead.
#   C4  C2 + C3, κ=32                     both reference fixes with the stronger shrinkage.
import zipfile, io
FZ4 = DRIVE / "frozen_LR4_k8"
OUTB = DRIVE / "val_candidates_batch"; OUTB.mkdir(exist_ok=True)
NAMES4 = [n for p in PAIRS for n in p]
tm = lambda arrs: np.array([a.mean() for a in arrs])
txt = lambda r: r if isinstance(r, str) else r["text"]
train_ft = {norm(txt(r)) for r in read_json(ROOT / "data" / "train" / "train_finetune.json")}
in_tf_val = np.array([norm(t) in train_ft for t in TEXTS["val"]]); assert int(in_tf_val.sum()) == 39
VT = split_npz(OLD / "val_shadow_1001_results.npz", "target_losses")
VR = {"s1": split_npz(OLD / "val_shadow_1001_results.npz", "shadow_losses"), "s2": split_npz(OLD / "val_shadow_1002_losses.npz"),
      "fin": split_npz(OLD / "val_ref_final_losses.npz"), "trn": split_npz(OLD / "val_ref_train_losses.npz")}
MT_VAL, L_VAL = tm(VT), np.array([a.size for a in VT], float)
PAIRED = np.stack([tm(REF[n]["val"]) for n in NAMES4]); PAIRED_IN = np.stack([MEM[n]["val"] for n in NAMES4])

def build(drop_final=False, train_as_in=False):
    rows, rin = [PAIRED], [PAIRED_IN]
    rows += [tm(VR["s1"])[None], tm(VR["s2"])[None]]; rin += [np.zeros((2, 2000), bool)]
    if not drop_final:
        rows.append(tm(VR["fin"])[None]); rin.append(np.zeros((1, 2000), bool))
    if train_as_in:
        rows.append(tm(VR["trn"])[None]); rin.append(in_tf_val[None])        # IN on its 39 texts, OUT elsewhere
    else:
        rows.append(np.where(in_tf_val, np.nan, tm(VR["trn"]))[None]); rin.append(np.zeros((1, 2000), bool))
    return np.vstack(rows), np.vstack(rin)

def zip_scores(path):
    with zipfile.ZipFile(path) as zf:
        return np.loadtxt(io.BytesIO(zf.read("val/prediction.csv")))

# Gate: the builder with defaults must reproduce the submitted LR4 ranking.
P, Pin = build()
s8 = lr_frozen(MT_VAL, L_VAL, P, Pin, kappa=8.0)
sub4 = zip_scores(FZ4 / "project_phase2_val_submission_LR4_k8.zip")
assert np.abs(s8 - sub4).max() < 1e-4 and np.array_equal(np.argsort(-s8, kind="stable"), np.argsort(-sub4, kind="stable")), \
    "LR4 not reproduced; stop"

# Dev analog for C2 (drop the final target from dev's fixed refs) as supporting information only.
P_dev = np.vstack([np.stack([mean_loss(n) for n in NAMES4]), FIX4[[0, 1, 2]]])
Pin_dev = np.vstack([np.stack([MEM[n]["train"] for n in NAMES4]), np.zeros((3, 2000), bool)])
print(f"[dev analog] LR4 κ=8 without final target: strict TPR {tpr_strict(LABELS, lr_frozen(DEV_MT, DEV_L, P_dev, Pin_dev)):.4f} "
      f"(LR4 0.7120)")

CANDS = {"C1_LR4_k16": dict(kappa=16.0), "C2_LR4_nofinal": dict(kappa=8.0, drop_final=True),
         "C3_LR4_trainIN": dict(kappa=8.0, train_as_in=True),
         "C4_LR4_nofinal_trainIN_k32": dict(kappa=32.0, drop_final=True, train_as_in=True)}
top = lambda s, k: set(np.argsort(-s)[:k])
summary = []
for name, cfg in CANDS.items():
    P, Pin = build(cfg.get("drop_final", False), cfg.get("train_as_in", False))
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
    summary.append({"candidate": name, **{k: v for k, v in cfg.items()}, "refs": P.shape[0],
                    "corr vs LR4": np.corrcoef(s, s8)[0, 1], "top-50 overlap": len(top(s, 50) & top(s8, 50)),
                    "top-750 overlap": len(top(s, 750) & top(s8, 750)), "zip": zp.name})
print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
json.dump({"candidates": {k: {kk: (vv if not isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in CANDS.items()},
           "status": "fixed before validation scores; leaderboard optimization", "uploaded": {}},
          open(OUTB / "settings.json", "w"), indent=2)
print("ZIPs in", ROOT, "and", OUTB, "| original LR4 ZIP untouched.")

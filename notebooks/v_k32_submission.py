# V-K32 — Single candidate from KR-1: frozen LR formula, 6 pairs + LR4's fixed OUT refs, κ=32. Builds val + final ZIPs.
# Run after N-2, N-3, N-4 and KR-1 in the same runtime. CPU only. Nothing is uploaded; LR4/LR6 artifacts are read only.
import zipfile, io
KAPPA_NEW = 32.0
FZ4, FZ6 = DRIVE / "frozen_LR4_k8", DRIVE / "frozen_LR6_k8"
FZK = DRIVE / "frozen_LR6_k32"
assert not (FZK / "settings.json").exists(), f"{FZK} already holds a frozen candidate; inspect it before rebuilding"
FZK.mkdir(exist_ok=True)
PAIRS6 = list(PAIRS) + [("p5A", "p5B"), ("p6A", "p6B")]
NAMES4, NAMES6 = [n for p in PAIRS for n in p], [n for p in PAIRS6 for n in p]
assert all(n in REF and n in MEM and "val" in MEM[n] and "final" in MEM[n] for n in NAMES6), "run KR-1 first"
tm = lambda arrs: np.array([a.mean() for a in arrs])
txt = lambda r: r if isinstance(r, str) else r["text"]

def find_data(n):
    for p in (DRIVE / "data" / f"shadow_{n}", DRIVE / f"data_shadow_{n}", ROOT / "data" / f"shadow_{n}"):
        if (p / "train_finetune.json").exists():
            return p / "train_finetune.json"
    raise FileNotFoundError(f"training data for shadow_{n}")
mem_s = {n: membership([txt(r) for r in read_json(find_data(n))]) for n in ("1001", "1002")}
assert not mem_s["1001"]["val"].any() and not mem_s["1002"]["val"].any()
train_ft = {norm(txt(r)) for r in read_json(ROOT / "data" / "train" / "train_finetune.json")}
in_tf = {ph: np.array([norm(t) in train_ft for t in TEXTS[ph]]) for ph in ("val", "final")}
assert int(in_tf["val"].sum()) == 39

# Targets and fixed OUT references, exactly as in LR4 / LR6.
VT = split_npz(OLD / "val_shadow_1001_results.npz", "target_losses")
FL = {k: split_npz(OLD / f"final_{k}_losses.npz") for k in ("target", "shadow1", "shadow2", "train_tgt", "val_tgt")}
VR = {"s1": split_npz(OLD / "val_shadow_1001_results.npz", "shadow_losses"), "s2": split_npz(OLD / "val_shadow_1002_losses.npz"),
      "fin": split_npz(OLD / "val_ref_final_losses.npz"), "trn": split_npz(OLD / "val_ref_train_losses.npz")}
for arrs, tgt in [(v, VT) for v in VR.values()] + [(v, FL["target"]) for v in FL.values()] + \
                 [(REF[n]["val"], VT) for n in NAMES6] + [(REF[n]["final"], FL["target"]) for n in NAMES6]:
    assert len(arrs) == 2000 and all(a.shape == b.shape for a, b in zip(arrs, tgt))
TARGET = {"val": (tm(VT), np.array([a.size for a in VT], float)),
          "final": (tm(FL["target"]), np.array([a.size for a in FL["target"]], float))}
FIXED = {"val": np.stack([tm(VR["s1"]), tm(VR["s2"]), tm(VR["fin"]), np.where(in_tf["val"], np.nan, tm(VR["trn"]))]),
         "final": np.stack([np.where(mem_s["1001"]["final"], np.nan, tm(FL["shadow1"])),
                            np.where(mem_s["1002"]["final"], np.nan, tm(FL["shadow2"])),
                            np.where(in_tf["final"], np.nan, tm(FL["train_tgt"])), tm(FL["val_tgt"])])}

def score(ph, names, kappa):
    mt, L = TARGET[ph]; fx = FIXED[ph]
    P = np.vstack([np.stack([tm(REF[n][ph]) for n in names]), fx])
    Pin = np.vstack([np.stack([MEM[n][ph] for n in names]), np.zeros(fx.shape, bool)])
    fin = np.isfinite(P)
    return lr_frozen(mt, L, P, Pin, kappa=kappa), (Pin & fin).sum(0), (~Pin & fin).sum(0)

def zip_scores(path, ph):
    with zipfile.ZipFile(path) as zf:
        return np.loadtxt(io.BytesIO(zf.read(f"{ph}/prediction.csv")))

# ---------- 1. Reproduction gates: LR4 and LR6 must rebuild their saved/submitted scores ----------
for ph in ("val", "final"):
    d4 = np.abs(score(ph, NAMES4, 8.0)[0] - np.load(FZ4 / f"{ph}_scores.npz")[f"{ph}_LR"]).max()
    d6 = np.abs(score(ph, NAMES6, 8.0)[0] - np.load(FZ6 / f"{ph}_scores.npz")[f"{ph}_LR6"]).max()
    print(f"{ph}: |rebuilt LR4 − saved| = {d4:.2e} | |rebuilt LR6 − saved| = {d6:.2e}")
    assert d4 < 1e-4 and d6 < 1e-4, f"{ph}: LR4/LR6 not reproduced; stop"
sub4 = zip_scores(FZ4 / "project_phase2_val_submission_LR4_k8.zip", "val")
r4 = score("val", NAMES4, 8.0)[0]
assert np.array_equal(np.argsort(-r4, kind="stable"), np.argsort(-sub4, kind="stable")), "LR4 ranks differ from submitted ZIP"

# ---------- 2. Dev 6 pairs κ=32 must equal the KR-1 score that was selected ----------
dev_k32 = lr_frozen(DEV_MT, DEV_L, np.vstack([np.stack([mean_loss(n) for n in NAMES6]), FIX4]),
                    np.vstack([np.stack([MEM[n]["train"] for n in NAMES6]), np.zeros(FIX4.shape, bool)]), kappa=KAPPA_NEW)
assert np.allclose(dev_k32, SC["real dev|6 pairs, κ=32"], rtol=0, atol=1e-12)
print(f"dev 6 pairs κ=32: strict TPR {tpr_strict(LABELS, dev_k32):.4f} (KR-1: 0.7510)")

# ---------- 3. Build val + final ZIPs ----------
for ph in ("val", "final"):
    s, n_in, n_out = score(ph, NAMES6, KAPPA_NEW)
    assert s.shape == (2000,) and np.isfinite(s).all()
    s4, s6 = score(ph, NAMES4, 8.0)[0], score(ph, NAMES6, 8.0)[0]
    print(f"{ph}: IN {dict(zip(*np.unique(n_in, return_counts=True)))} | OUT {dict(zip(*np.unique(n_out, return_counts=True)))} "
          f"| corr vs LR4 {np.corrcoef(s, s4)[0, 1]:.4f} | vs LR6 {np.corrcoef(s, s6)[0, 1]:.4f} "
          f"| top-750 overlap vs LR4 {len(set(np.argsort(-s)[:750]) & set(np.argsort(-s4)[:750]))}")
    pred = ROOT / "data" / ph / "prediction_LR6_k32.csv"
    zp = ROOT / f"project_phase2_{ph}_submission_LR6_k32.zip"
    np.savetxt(pred, s, fmt="%.17g")
    for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", ph,
                 "--prediction", str(pred), "--output", str(zp)],
                [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", ph]):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print(r.stdout.strip(), r.stderr.strip()); assert r.returncode == 0
    assert np.allclose(zip_scores(zp, ph), s, rtol=0, atol=1e-9)
    shutil.copy(zp, FZK); shutil.copy(zp, DRIVE)
    np.savez(FZK / f"{ph}_scores.npz", **{f"{ph}_LR6_k32": s, "n_in": n_in, "n_out": n_out})

json.dump({"method": "frozen LR formula (unequal-variance Gaussian LR on sqrt(T)·mean loss, pooled shrinkage, 1+1/n, "
                     "finite masks) with 6 IN/OUT pairs and κ=32",
           "pairs": [list(p) for p in PAIRS6], "kappa": KAPPA_NEW,
           "val_fixed_out_refs": ["shadow1001", "shadow1002", "final target", "train target (39 masked)"],
           "final_fixed_out_refs": ["shadow1001 (masked)", "shadow1002 (masked)", "train target (masked)", "val target"],
           "selection": "KR-1 pre-declared rule: argmax mean fresh-mock strict TPR over {4,6 pairs}×κ{2,8,32,∞}; "
                        "consistent iff mean Δ>0, Δ>0 on ≥5/6 fresh mocks, dev Δ≥0 (vs 4 pairs κ=8)",
           "kr1": {"fresh_mean_TPR": 0.7686, "fresh_mean_delta_vs_LR4": 0.0287, "wins": "6/6", "worst_delta": 0.0020,
                   "dev_TPR": 0.751, "dev_delta": 0.039,
                   "vs_LR6_k8_fresh_deltas": {"p4A": -0.016, "p4B": -0.014, "p5A": 0.008, "p5B": 0.021,
                                              "p6A": 0.015, "p6B": 0.021}, "vs_LR6_k8_dev": 0.017},
           "known_risk": "LR6 (6 pairs, κ=8) improved dev and mocks but scored 71.3 on val vs LR4 76.6",
           "validation_submission": "built, not uploaded"}, open(FZK / "settings.json", "w"), indent=2)
print("Built", ROOT / "project_phase2_val_submission_LR6_k32.zip", "| copies in", FZK, "| LR4/LR6 artifacts untouched.")

# GN — Unattended experiment: per-text LoRA gradient norm through the per-text IN/OUT Gaussian LR (white-box signal).
# Paste as ONE cell after N-1..N-4 in a GPU runtime and leave it. Resumable: each model's statistics are saved to Drive
# as soon as they are computed; re-running skips finished models. Nothing is uploaded to Codabench.
#
# Statistic per (model, text): log ‖∇_LoRA mean-token CE‖₂, batch 1, no padding, eval mode (dropout off), fp32.
# Same references, memberships, masks as LR4 (κ=8). Scores compared, fixed in advance:
#   GN      Gaussian LR on log grad-norm (no √T scaling)
#   SUM     LR4 (mean loss) + GN                (heuristic sum; correlated statistics)
# Evaluation: real dev + fresh mocks p5A/p5B/p6A/p6B (8 paired refs + 4 fixed; mocks' pairs are not in the refs).
# Pre-declared: a val ZIP is worth submitting only if dev TPR ≥ LR4's 0.712 AND mean mock Δ > 0 with ≥ 3/4 mock wins.
import time, zipfile, io
GN = DRIVE / "cache" / "gradnorm"; GN.mkdir(parents=True, exist_ok=True)
KAP = 8.0
PAIRS4 = [("inA", "inB"), ("p2A", "p2B"), ("p3A", "p3B"), ("p4A", "p4B")]
assert [tuple(p) for p in PAIRS] == PAIRS4
NAMES8 = [n for p in PAIRS4 for n in p]
MOCKS = ["p5A", "p5B", "p6A", "p6B"]
txt = lambda r: r if isinstance(r, str) else r["text"]
tmn = lambda arrs: np.array([a.mean() for a in arrs])
for n in MOCKS:                                                     # p5/p6 cached losses and memberships (CPU)
    if n not in REF:
        REF[n] = {ph: split_npz(OLD / f"{ph}_shadow_{n}_losses.npz") for ph in PHASES}
    if n not in MEM:
        MEM[n] = membership(shadow_texts(n))

def adapter_dir(name):
    for p in (DRIVE / "models" / name, DRIVE / name, ROOT / "models" / name):
        if (p / "adapter_model.safetensors").exists():
            return p
    raise FileNotFoundError(name)
ADAPTERS = {"dev_target": ADAPTER_DIR, "val_target": ROOT / "models" / "val" / MODEL_NAME,
            "final_target": ROOT / "models" / "final" / MODEL_NAME,
            "s1001": adapter_dir("shadow_1001"), "s1002": adapter_dir("shadow_1002")} | \
           {n: adapter_dir(f"shadow_{n}") for n in NAMES8 + MOCKS}
# Which models are needed on which phase.
JOBS = [("train", m) for m in ["dev_target"] + NAMES8 + ["s1001", "s1002", "val_target", "final_target"] + MOCKS] + \
       [("val", m) for m in ["val_target"] + NAMES8 + ["s1001", "s1002", "final_target", "dev_target"]]
IDS = {ph: [tokenizer(t, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)["input_ids"][0] for t in TEXTS[ph]]
       for ph in ("train", "val")}
for ph in IDS:
    assert all(ids.numel() - 1 == a.size for ids, a in zip(IDS[ph], TGT[ph])), f"{ph}: token counts differ from cache"
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

def run_model(name):
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32)
    m = PeftModel.from_pretrained(base, ADAPTERS[name], is_trainable=True)
    m.config.pad_token_id = tokenizer.pad_token_id
    m.to(device).eval()
    params = [p for n_, p in m.named_parameters() if "lora_" in n_]
    for n_, p in m.named_parameters():
        p.requires_grad_("lora_" in n_)
    assert params, "no LoRA parameters found"
    return m, params

t_start = time.time()
done = sum((GN / f"{ph}_{m}.npz").exists() for ph, m in JOBS)
print(f"{done}/{len(JOBS)} model×phase jobs already on Drive")
loaded = None
for i, (ph, name) in enumerate(JOBS):
    out = GN / f"{ph}_{name}.npz"
    if out.exists():
        continue
    if loaded is None or loaded[0] != name:
        if loaded is not None:
            loaded[1].to("cpu"); del loaded; gc.collect(); torch.cuda.empty_cache()
        m, params = run_model(name); loaded = (name, m, params)
    _, m, params = loaded
    lg, ml = np.empty(2000), np.empty(2000)
    t0 = time.time()
    for j, ids in enumerate(IDS[ph]):
        x = ids[None].to(device)
        m.zero_grad(set_to_none=True)
        logits = m(input_ids=x, use_cache=False).logits[0, :-1].float()
        loss = F.cross_entropy(logits, x[0, 1:])
        loss.backward()
        gsq = sum((p.grad.double() ** 2).sum() for p in params if p.grad is not None)
        lg[j], ml[j] = 0.5 * float(torch.log(gsq)), float(loss.detach())
    m.zero_grad(set_to_none=True)
    np.savez(out, log_gradnorm=lg, mean_loss=ml)
    print(f"[{i + 1}/{len(JOBS)}] {ph} {name}: {time.time() - t0:.0f}s | elapsed {(time.time() - t_start) / 60:.1f} min")
if loaded is not None:
    loaded[1].to("cpu"); del loaded; gc.collect(); torch.cuda.empty_cache()

# ---------------- Evaluation (CPU) ----------------
G = {(ph, m): np.load(GN / f"{ph}_{m}.npz") for ph, m in JOBS}
# Alignment check: fp32 mean loss from this pass vs cached (fp16-forward) mean loss.
for ph, m, cached in [("train", "dev_target", TGT["train"]), ("val", "val_target", None)]:
    if cached is not None:
        d = np.abs(G[(ph, m)]["mean_loss"] - tmn(cached)).max()
        print(f"alignment {ph} {m}: max |fp32 − cached mean loss| = {d:.4f}"); assert d < 0.05, "texts/models misaligned"
for n in NAMES8[:2]:
    d = np.abs(G[("train", n)]["mean_loss"] - mean_loss(n)).max(); assert d < 0.05, f"{n} misaligned ({d})"

ONES = np.ones(2000)
def lr_stat(t, refs, refs_in, L=ONES):
    return lr_frozen(t, L, np.stack(refs), np.stack(refs_in), kappa=KAP)
Z = np.zeros(2000, bool)
dev_fixed = ["s1001", "s1002", "val_target", "final_target"]           # FIX4 order
assert np.allclose(np.stack([G[("train", k)]["mean_loss"] for k in dev_fixed]), FIX4, atol=0.05)

def dev_scores(tgt_name, tgt_loss):
    refs = NAMES8 + dev_fixed
    rin = [MEM[n]["train"] for n in NAMES8] + [Z] * 4
    gn = lr_stat(G[("train", tgt_name)]["log_gradnorm"], [G[("train", r)]["log_gradnorm"] for r in refs], rin)
    lr4 = lr_frozen(tgt_loss, DEV_L, np.vstack([np.stack([mean_loss(n) for n in NAMES8]), FIX4]),
                    np.vstack([np.stack([MEM[n]["train"] for n in NAMES8]), np.zeros(FIX4.shape, bool)]), kappa=KAP)
    return {"LR4": lr4, "GN": gn, "SUM": lr4 + gn}

def pboot(a, b, y, seed, B=2000):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); g = np.random.default_rng(seed); d = []
    for _ in range(B):
        ix = np.concatenate([g.choice(pos, pos.size), g.choice(neg, neg.size)])
        d.append(strict_fast(a[ix], y[ix]) - strict_fast(b[ix], y[ix]))
    lo, hi = np.percentile(d, [2.5, 97.5]); return f"{tpr_strict(y, a) - tpr_strict(y, b):+.3f} [{lo:+.3f}, {hi:+.3f}]"

lines, rows = [], []
D = dev_scores("dev_target", DEV_MT)
assert np.allclose(D["LR4"], LR4_DEV, atol=1e-6)
for k in ("GN", "SUM"):
    rows.append({"target": "real dev", "score": k, "TPR": tpr_strict(LABELS, D[k]), "AUC": roc_auc_score(LABELS, D[k]),
                 "Δ vs LR4 [CI]": pboot(D[k], D["LR4"], LABELS, 20261017)})
mock_d = {"GN": [], "SUM": []}
for j, t in enumerate(MOCKS):
    y = MEM[t]["train"].astype(int)
    S = dev_scores(t, mean_loss(t))
    rows.append({"target": t, "score": "LR4", "TPR": tpr_strict(y, S["LR4"]), "AUC": roc_auc_score(y, S["LR4"]), "Δ vs LR4 [CI]": "—"})
    for k in ("GN", "SUM"):
        mock_d[k].append(tpr_strict(y, S[k]) - tpr_strict(y, S["LR4"]))
        rows.append({"target": t, "score": k, "TPR": tpr_strict(y, S[k]), "AUC": roc_auc_score(y, S[k]),
                     "Δ vs LR4 [CI]": pboot(S[k], S["LR4"], y, 20261026 + j)})
res = pd.DataFrame(rows)
lines.append(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
lines.append(f"corr(LR4, GN) on dev: {np.corrcoef(D['LR4'], D['GN'])[0, 1]:.4f}")
verdict = {}
for k in ("GN", "SUM"):
    dev_ok = tpr_strict(LABELS, D[k]) >= 0.712
    mk_ok = np.mean(mock_d[k]) > 0 and sum(d > 0 for d in mock_d[k]) >= 3
    verdict[k] = bool(dev_ok and mk_ok)
    lines.append(f"{k}: dev ≥ 0.712: {dev_ok} | mock mean Δ {np.mean(mock_d[k]):+.4f}, wins {sum(d > 0 for d in mock_d[k])}/4 "
                 f"→ {'WORTH ONE VAL SUBMISSION' if verdict[k] else 'do not submit'}")

# Validation ZIPs (built regardless; submit only if the verdict says so).
train_ft = {norm(txt(r)) for r in read_json(ROOT / "data" / "train" / "train_finetune.json")}
in_tf_val = np.array([norm(t) in train_ft for t in TEXTS["val"]]); assert int(in_tf_val.sum()) == 39
VT = split_npz(OLD / "val_shadow_1001_results.npz", "target_losses")
VR = {"s1": split_npz(OLD / "val_shadow_1001_results.npz", "shadow_losses"), "s2": split_npz(OLD / "val_shadow_1002_losses.npz"),
      "fin": split_npz(OLD / "val_ref_final_losses.npz"), "trn": split_npz(OLD / "val_ref_train_losses.npz")}
FIXV = np.stack([tmn(VR["s1"]), tmn(VR["s2"]), tmn(VR["fin"]), np.where(in_tf_val, np.nan, tmn(VR["trn"]))])
PIN_V = np.vstack([np.stack([MEM[n]["val"] for n in NAMES8]), np.zeros((4, 2000), bool)])
lr4_val = lr_frozen(tmn(VT), np.array([a.size for a in VT], float), np.vstack([np.stack([tmn(REF[n]["val"]) for n in NAMES8]), FIXV]),
                    PIN_V, kappa=KAP)
sub4 = np.loadtxt(io.BytesIO(zipfile.ZipFile(DRIVE / "frozen_LR4_k8" / "project_phase2_val_submission_LR4_k8.zip").read("val/prediction.csv")))
assert np.abs(lr4_val - sub4).max() < 1e-4, "LR4 val not reproduced"
gv = [G[("val", r)]["log_gradnorm"] for r in NAMES8 + ["s1001", "s1002", "final_target"]] + \
     [np.where(in_tf_val, np.nan, G[("val", "dev_target")]["log_gradnorm"])]
gn_val = lr_frozen(G[("val", "val_target")]["log_gradnorm"], ONES, np.stack(gv), PIN_V, kappa=KAP)
VAL = {"GN": gn_val, "SUM": lr4_val + gn_val}
for k, v in VAL.items():
    assert np.isfinite(v).all()
    tag = f"GN_{k}_k8"
    pred = ROOT / "data" / "val" / f"prediction_{tag}.csv"; zp = ROOT / f"project_phase2_val_submission_{tag}.zip"
    np.savetxt(pred, v, fmt="%.17g")
    for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", "val", "--prediction", str(pred), "--output", str(zp)],
                [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", "val"]):
        rr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); assert rr.returncode == 0, rr.stderr[-800:]
    shutil.copy(zp, GN)
    top = lambda s, n: set(np.argsort(-s)[:n])
    lines.append(f"val {tag}: corr vs LR4 {np.corrcoef(v, lr4_val)[0, 1]:.4f} | top-776 overlap {len(top(v, 776) & top(lr4_val, 776))} | zip {zp.name}")
np.savez(GN / "scores.npz", **{f"dev_{k}": v for k, v in D.items()}, **{f"val_{k}": v for k, v in VAL.items()})
report = "\n".join(lines)
(GN / "RESULTS.txt").write_text(report)
print(report); print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Results: {GN / 'RESULTS.txt'}")

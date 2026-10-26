# =====================================================================================================================
# RES-1 — Class-specific residuals vs the frozen LR4 references (CPU, cached losses). Run after N-2..N-4.
#   members:      r_IN  = √T·(ℓ_target − mean ℓ over IN refs)        raw: ℓ_target − mean ℓ over IN refs
#   non-members:  r_OUT = √T·(ℓ_target − mean ℓ over OUT refs)       raw: same without √T
# Real dev target vs fresh mocks p4A–p6B. For each mock, the real target is scored with the SAME reference pool
# (the mock's own pair removed), so real − mock differences are matched. Official labels are kept; the two dev
# non-members that appear in the dev target's training data are flagged and reported with and without.
# =====================================================================================================================
txt = lambda r: r if isinstance(r, str) else r["text"]
PAIRS4 = [("inA", "inB"), ("p2A", "p2B"), ("p3A", "p3B"), ("p4A", "p4B")]
assert [tuple(p) for p in PAIRS] == PAIRS4
NAMES8 = [n for p in PAIRS4 for n in p]
MOCKS = ["p4A", "p4B", "p5A", "p5B", "p6A", "p6B"]
for n in MOCKS:
    if n not in REF:
        REF[n] = {ph: split_npz(OLD / f"{ph}_shadow_{n}_losses.npz") for ph in PHASES}
    if n not in MEM:
        MEM[n] = membership(shadow_texts(n))
ML = {n: mean_loss(n) for n in NAMES8 + MOCKS}
SQ = np.sqrt(DEV_L)

tf_norm = {norm(txt(r)) for r in read_json(ROOT / "data" / "train" / "train_finetune.json")}
in_tf_dev = np.array([norm(t) in tf_norm for t in TEXTS["train"]])
EXC = np.flatnonzero((LABELS == 0) & in_tf_dev)
print("Known exceptions (labelled non-member, present in dev target's training data):", EXC.tolist())
assert len(EXC) == 2

def residuals(t_loss, y, pool):
    """IN residual for y==1, OUT residual for y==0; pool = paired names; OUT side also gets the 4 fixed refs."""
    P = np.stack([ML[n] for n in pool]); Pin = np.stack([MEM[n]["train"] for n in pool])
    n_in = Pin.sum(0); assert (n_in[y == 1] > 0).all()
    mu_in = np.where(Pin, P, 0).sum(0) / np.maximum(n_in, 1)
    Pout = np.vstack([np.where(~Pin, P, np.nan), FIX4])
    mu_out = np.nanmean(Pout, 0)
    raw = np.where(y == 1, t_loss - mu_in, t_loss - mu_out)
    return raw * SQ, raw

def summarize(r, idx, seed, B=2000):
    g = np.random.default_rng(seed); x = r[idx]
    bm = [x[g.integers(0, x.size, x.size)].mean() for _ in range(B)]
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": x.size, "mean": x.mean(), "mean CI": f"[{np.percentile(bm, 2.5):+.3f}, {np.percentile(bm, 97.5):+.3f}]",
            "median": q[2], "q05": q[0], "q25": q[1], "q75": q[3], "q95": q[4]}

def diff_ci(ra, ia, rb, ib, seed, B=2000):
    """Bootstrap CI for mean(ra[ia]) − mean(rb[ib]), resampling each class independently."""
    g = np.random.default_rng(seed)
    d = [ra[ia][g.integers(0, ia.size, ia.size)].mean() - rb[ib][g.integers(0, ib.size, ib.size)].mean() for _ in range(B)]
    return f"{ra[ia].mean() - rb[ib].mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"

rows, cmp_rows = [], []
pools = {"LR4 pool (pairs 1–4)": NAMES8, "pairs 1–3 (p4 removed)": [n for n in NAMES8 if n[:2] != "p4"]}
for pname, pool in pools.items():
    rr, rraw = residuals(DEV_MT, LABELS, pool)
    mem, non = np.flatnonzero(LABELS == 1), np.flatnonzero(LABELS == 0)
    non_clean = np.setdiff1d(non, EXC)
    for cls, idx in (("IN (members)", mem), ("OUT (non-members)", non), ("OUT excl. 2 exceptions", non_clean)):
        rows.append({"target": "real dev", "pool": pname, "class": cls, "scale": "√T", **summarize(rr, idx, 1)})
        rows.append({"target": "real dev", "pool": pname, "class": cls, "scale": "raw", **summarize(rraw, idx, 2)})
    print(f"[{pname}] exception residuals (√T): {dict(zip(EXC.tolist(), np.round(rr[EXC], 3).tolist()))}")
for j, t in enumerate(MOCKS):
    pool = [n for n in NAMES8 if n not in (t[:2] + "A", t[:2] + "B")]
    pname = "pairs 1–3 (p4 removed)" if t[:2] == "p4" else "LR4 pool (pairs 1–4)"
    y = MEM[t]["train"].astype(int)
    rr, rraw = residuals(ML[t], y, pool)
    real_rr, real_raw = residuals(DEV_MT, LABELS, pool)                          # matched pool
    mem_t, non_t = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    mem_r, non_r = np.flatnonzero(LABELS == 1), np.setdiff1d(np.flatnonzero(LABELS == 0), EXC)
    for cls, idx in (("IN (members)", mem_t), ("OUT (non-members)", non_t)):
        rows.append({"target": t, "pool": pname, "class": cls, "scale": "√T", **summarize(rr, idx, 10 + j)})
        rows.append({"target": t, "pool": pname, "class": cls, "scale": "raw", **summarize(rraw, idx, 20 + j)})
    cmp_rows.append({"mock": t, "pool": pname,
                     "IN mean: real − mock (√T)": diff_ci(real_rr, mem_r, rr, mem_t, 30 + j),
                     "IN mean: real − mock (raw)": diff_ci(real_raw, mem_r, rraw, mem_t, 40 + j),
                     "OUT mean: real − mock (√T, excl. exc.)": diff_ci(real_rr, non_r, rr, non_t, 50 + j),
                     "IN median real / mock (√T)": f"{np.median(real_rr[mem_r]):+.3f} / {np.median(rr[mem_t]):+.3f}",
                     "IN q05–q95 real / mock (√T)": f"[{np.percentile(real_rr[mem_r], 5):+.2f}, {np.percentile(real_rr[mem_r], 95):+.2f}] / "
                                                   f"[{np.percentile(rr[mem_t], 5):+.2f}, {np.percentile(rr[mem_t], 95):+.2f}]"})
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
res = pd.DataFrame(rows)
print("\nResidual summaries (target loss minus matching-side reference mean; negative = target lower than refs):")
print(res.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print("\nMatched comparison, real dev target vs each fresh mock (same reference pool):")
print(pd.DataFrame(cmp_rows).to_string(index=False))
RES = DRIVE / "cache" / "residuals"; RES.mkdir(parents=True, exist_ok=True)
res.to_csv(RES / "res1_summaries.csv", index=False); pd.DataFrame(cmp_rows).to_csv(RES / "res1_matched.csv", index=False)


# =====================================================================================================================
# REP-1 — Controlled replica of the dev target: batch 2 × accumulation 4 vs batch 8, same precision, seed, data order.
# Separate cell; run only when decided (GPU, ~2 × training time). DIAGNOSTIC models: trained on the dev target's actual
# training set, so never usable as OUT references. Exact released records (duplicates preserved, released order).
# Precision is fixed explicitly to fp16 for both arms (matches our T4 shadows); bf16 would be a separate experiment.
# =====================================================================================================================
REP_SEED = 42                                   # repeat later with another seed before any retraining decision
ARMS = {"b2x4": ("2", "4"), "b8": ("8", "1")}
REP = DRIVE / "models" / "replica_dev"; REP.mkdir(parents=True, exist_ok=True)
assert len(read_json(ROOT / "data" / "train" / "train_finetune.json")) == 10000     # released file, duplicates kept
import torch as _t
print("GPU:", _t.cuda.get_device_name(0), "| bf16 supported:", _t.cuda.is_bf16_supported())

def rep_cmd(arm, out_dir):
    bs, ga = ARMS[arm]
    args = [sys.executable, str(ROOT / "ft_llm" / "train_shadow_model.py"),
            "--data-dir", str(ROOT / "data" / "train"), "--outdir", str(out_dir), "--model-name", BASE_MODEL_NAME,
            "--epochs", "3", "--batch-size", bs, "--gradient-accumulation-steps", ga,
            "--block-size", "512", "--lr", "2e-4", "--weight-decay", "0.01", "--seed", str(REP_SEED),
            "--lora-r", str(adapter_config["r"]), "--lora-alpha", str(adapter_config["lora_alpha"]),
            "--lora-dropout", str(adapter_config["lora_dropout"]),
            "--target-modules", ",".join(adapter_config["target_modules"]),
            "--precision", "fp16", "--gradient-checkpointing"]
    return args

@torch.inference_mode()
def rep_losses(model, texts, batch_size=2):           # same procedure as every cached target/shadow loss
    out = []
    for s0 in range(0, len(texts), batch_size):
        b = tokenizer(texts[s0:s0 + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        b = {k: v.to(device) for k, v in b.items()}
        tgt, mask = b["input_ids"][:, 1:], b["attention_mask"][:, 1:].bool()
        logits = model(**b, use_cache=False).logits[:, :-1, :]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
        out += [row[m].float().cpu().numpy() for row, m in zip(loss, mask)]
    return out

REPML = {}
for arm in ARMS:
    name = f"{arm}_s{REP_SEED}"
    cache = REP / f"{name}_dev_losses.npz"
    if not cache.exists():
        out_dir = ROOT / "models" / f"replica_dev_{name}"
        if not (REP / name / "adapter_model.safetensors").exists():
            assert not out_dir.exists() or not any(out_dir.iterdir()), f"{out_dir} not empty"
            log = ROOT / f"train_replica_{name}.log"
            with open(log, "w") as lf:
                r = subprocess.run(rep_cmd(arm, out_dir), cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
            assert r.returncode == 0, f"training failed; see {log}"
            shutil.copytree(out_dir, REP / name, dirs_exist_ok=True); shutil.copy(log, REP / f"{name}.log")
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=model_dtype)
        m = PeftModel.from_pretrained(base, REP / name); m.config.pad_token_id = tokenizer.pad_token_id; m.to(device).eval()
        ls = rep_losses(m, TEXTS["train"])
        assert all(x.size == t.size for x, t in zip(ls, TGT["train"]))
        np.savez_compressed(cache, lengths=np.asarray([x.size for x in ls]), losses=np.concatenate(ls))
        m.to("cpu"); del m, base; gc.collect(); torch.cuda.empty_cache()
    REPML[arm] = np.array([a.mean() for a in split_npz(cache)])

# Compare each replica with the real dev target on the dev evaluation texts (official labels; exceptions flagged).
mem, non = np.flatnonzero(LABELS == 1), np.setdiff1d(np.flatnonzero(LABELS == 0), EXC)
OUTREF = np.nanmean(np.vstack([np.where(~np.stack([MEM[n]["train"] for n in NAMES8]), np.stack([ML[n] for n in NAMES8]), np.nan), FIX4]), 0)
rows = []
for name, ml in [("real dev target", DEV_MT)] + [(f"replica {a}", REPML[a]) for a in ARMS]:
    eff = (OUTREF - ml) * SQ                               # memorization effect vs OUT-reference mean (√T)
    r = {"model": name, "member mean loss": ml[mem].mean(), "non-member mean loss": ml[non].mean(),
         "gap (non − mem)": ml[non].mean() - ml[mem].mean(),
         "member effect mean (√T)": eff[mem].mean(), "member effect median": np.median(eff[mem]),
         "non-member effect mean": eff[non].mean(), "AUC (loss only)": roc_auc_score(LABELS, -ml)}
    if name != "real dev target":
        r |= {"corr with target (members)": np.corrcoef(ml[mem], DEV_MT[mem])[0, 1],
              "corr with target (non-members)": np.corrcoef(ml[non], DEV_MT[non])[0, 1],
              "MAE vs target (members)": np.abs(ml[mem] - DEV_MT[mem]).mean(),
              "MAE vs target (non-members)": np.abs(ml[non] - DEV_MT[non]).mean()}
    rows.append(r)
print(pd.DataFrame(rows).T.to_string(header=False, float_format=lambda x: f"{x:.4f}"))
pd.DataFrame(rows).to_csv(REP / f"rep1_seed{REP_SEED}.csv", index=False)

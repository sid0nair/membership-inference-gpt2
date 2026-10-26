# =====================================================================================================================
# Matched experiment: does the auxiliary-data source matter when adding a fifth pair?
# q5A/q5B = pooled-auxiliary counterparts of p5A/p5B. Identical: candidate IN/OUT assignment (assign_sides(5)), candidate
# text representation, training-set size, aux-sampling seed, shuffle procedure and seed, training seed, optimizer settings.
# Changed: only the auxiliary SOURCE (pooled shadow-1001/1002 data, the pool used for pairs 2–3), filtered against all
# candidates and deduplicated. Names start with "q5" so no code that keys pairs on n[:2] can confuse them with p5.
# Run each cell after N-1..N-4 (GPU runtime for PT-1/PT-2). Incumbent stays raw LR4 κ=6 (val 76.7).
# =====================================================================================================================

# ---------------------------------------------------------------------------------------------------------------------
# PT-1 — Build q5A/q5B datasets (checked against p5) and train both in the background with automatic Drive backups.
# ---------------------------------------------------------------------------------------------------------------------
import random
K_PAIR, ORIG, NEW = 5, ("p5A", "p5B"), ("q5A", "q5B")
assert [tuple(p) for p in PAIRS] == [("inA", "inB"), ("p2A", "p2B"), ("p3A", "p3B"), ("p4A", "p4B")]
txt = lambda r: r if isinstance(r, str) else r["text"]

def find_data(n):
    for p in (DRIVE / "data" / f"shadow_{n}", DRIVE / f"data_shadow_{n}", ROOT / "data" / f"shadow_{n}"):
        if (p / "train_finetune.json").exists():
            return p / "train_finetune.json"
    raise FileNotFoundError(f"training data for shadow_{n}")
load_texts = lambda n: [txt(r) for r in read_json(find_data(n))]

# 1. Reconstruct p5's candidate sides exactly as P-1 did, and check them against p5's saved training data.
side_items, side_n = assign_sides(K_PAIR)
EV = {}
for s, name in enumerate(ORIG):
    EV[name] = list(dict.fromkeys(TEXTS[EVAL_ITEMS[j][0]][EVAL_ITEMS[j][1]] for j in side_items[s]))
    saved = load_texts(name)
    assert len(saved) == SHADOW_TRAIN_SIZE
    pk = prefix_keys(saved)
    saved_ev = [t for t, p in zip(saved, pk) if norm(t) in EVAL_NORM_SET or p in EVAL_PREF_SET]
    assert sorted(saved_ev) == sorted(EV[name]), f"{name}: reconstructed candidate side differs from saved p5 data"
    print(f"{name}: {len(EV[name])} candidate texts IN, {SHADOW_TRAIN_SIZE - len(EV[name])} aux — matches saved data")

# 2. Pooled auxiliary pool (pairs 2–3 source), filtered against all candidates, deduplicated.
cands = list(dict.fromkeys(load_texts("1001") + load_texts("1002")))
cpk = prefix_keys(cands)
POOL = [t for t, p in zip(cands, cpk) if norm(t) not in EVAL_NORM_SET and p not in EVAL_PREF_SET]
print(f"pooled aux pool: {len(POOL)} of {len(cands)} after excluding all candidate groups")

# 3. Same sampling and shuffle seeds and procedure as p5; only the pool differs.
PT_DIR = DRIVE / "cache" / "pooled_p5"; PT_DIR.mkdir(parents=True, exist_ok=True)
record = {"pair_no": K_PAIR, "names": list(NEW), "counterpart_of": list(ORIG), "aux_source": "pooled shadow-1001/1002",
          "group_seed": 7000 + K_PAIR, "models": {}}
for s, (orig, name) in enumerate(zip(ORIG, NEW)):
    n_aux = SHADOW_TRAIN_SIZE - len(EV[orig])
    assert len(POOL) >= n_aux, f"{name}: need {n_aux}, pool has {len(POOL)}"
    aux = random.Random(8000 + 10 * K_PAIR + s).sample(POOL, n_aux)
    tt = EV[orig] + aux
    random.Random(9000 + 10 * K_PAIR + s).shuffle(tt)
    assert len(tt) == SHADOW_TRAIN_SIZE and len(set(tt)) == len(tt)
    mem_new, mem_orig = membership(tt), membership(load_texts(orig))
    for ph in PHASES:
        assert np.array_equal(mem_new[ph], mem_orig[ph]), f"{name} {ph}: candidate membership differs from {orig}"
    d = DRIVE / "data" / f"shadow_{name}"
    if (d / "train_finetune.json").exists():
        assert load_texts(name) == tt, f"{d} exists with different content; inspect before overwriting"
    else:
        d.mkdir(parents=True, exist_ok=True)
        with (d / "train_finetune.json").open("w", encoding="utf-8") as f:
            json.dump([{"text": t} for t in tt], f, ensure_ascii=False)
    shutil.copytree(d, ROOT / "data" / f"shadow_{name}", dirs_exist_ok=True)
    shared = len({norm(t) for t in aux} & {norm(t) for t in load_texts(orig)})
    record["models"][name] = {"aux_sampling_seed": 8000 + 10 * K_PAIR + s, "shuffle_seed": 9000 + 10 * K_PAIR + s,
                              "train_seed": 5000 + 10 * K_PAIR + s, "eval_in": len(EV[orig]), "aux": n_aux,
                              "aux_shared_with_" + orig: shared}
json.dump(record, open(PT_DIR / "construction.json", "w"), indent=2)
print(json.dumps(record["models"], indent=1))

# 4. Train sequentially in the background (same command as every earlier shadow), copying each adapter to Drive.
def train_cmd(data_dir, out_dir, seed):
    args = [sys.executable, str(ROOT / "ft_llm" / "train_shadow_model.py"),
            "--data-dir", str(data_dir), "--outdir", str(out_dir), "--model-name", BASE_MODEL_NAME,
            "--epochs", "3", "--batch-size", "2", "--gradient-accumulation-steps", "4",
            "--block-size", "512", "--lr", "2e-4", "--weight-decay", "0.01", "--seed", str(seed),
            "--lora-r", str(adapter_config["r"]), "--lora-alpha", str(adapter_config["lora_alpha"]),
            "--lora-dropout", str(adapter_config["lora_dropout"]),
            "--target-modules", ",".join(adapter_config["target_modules"]),
            "--precision", "auto", "--gradient-checkpointing"]
    return " ".join(f"'{a}'" for a in args)
(DRIVE / "models").mkdir(exist_ok=True); (DRIVE / "logs").mkdir(exist_ok=True)
steps = []
for s, name in enumerate(NEW):
    if (DRIVE / "models" / f"shadow_{name}" / "adapter_model.safetensors").exists():
        print(f"{name}: adapter already on Drive, skipping"); continue
    out_dir, log = ROOT / "models" / f"shadow_{name}", ROOT / f"train_shadow_{name}.log"
    assert not out_dir.exists() or not any(out_dir.iterdir()), f"{out_dir} not empty"
    steps.append(f"{train_cmd(ROOT / 'data' / f'shadow_{name}', out_dir, 5000 + 10 * K_PAIR + s)} > '{log}' 2>&1")
    steps.append(f"cp -r '{out_dir}' '{DRIVE}/models/' && cp '{log}' '{DRIVE}/logs/'")
if steps:
    pt_proc = subprocess.Popen(" && ".join(steps), shell=True, cwd=ROOT, env=dict(os.environ),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Training {[n for n in NEW]} sequentially (PID {pt_proc.pid}); ~25 min each. "
          f"Progress: !tail -c 300 {ROOT}/train_shadow_q5A.log | done when pt_proc.poll() is not None")


# ---------------------------------------------------------------------------------------------------------------------
# PT-2 — Cache q5A/q5B per-token losses for train/val/final (GPU), same procedure as every earlier shadow.
# ---------------------------------------------------------------------------------------------------------------------
@torch.inference_mode()
def collect_losses_pt(model, texts, batch_size=2):
    out = []
    for s0 in tqdm(range(0, len(texts), batch_size), leave=False):
        b = tokenizer(texts[s0:s0 + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        b = {k: v.to(device) for k, v in b.items()}
        tgt, mask = b["input_ids"][:, 1:], b["attention_mask"][:, 1:].bool()
        logits = model(**b, use_cache=False).logits[:, :-1, :]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
        for row, m in zip(loss, mask):
            out.append(row[m].float().cpu().numpy())
    return out

for name in ("q5A", "q5B"):
    adapter = DRIVE / "models" / f"shadow_{name}"
    assert (adapter / "adapter_model.safetensors").exists(), f"{name}: adapter not on Drive yet"
    if all((OLD / f"{ph}_shadow_{name}_losses.npz").exists() for ph in PHASES):
        print(f"{name}: losses already cached"); continue
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=model_dtype)
    m = PeftModel.from_pretrained(base, adapter); m.config.pad_token_id = tokenizer.pad_token_id; m.to(device).eval()
    for ph in PHASES:                                                   # train first: needed for dev/mock evaluation
        ls = collect_losses_pt(m, TEXTS[ph])
        assert len(ls) == 2000 and all(x.size == t.size for x, t in zip(ls, TGT[ph]))
        np.savez_compressed(OLD / f"{ph}_shadow_{name}_losses.npz", lengths=np.asarray([x.size for x in ls]),
                            losses=np.concatenate(ls))
        print(f"{name} {ph}: cached")
    m.to("cpu"); del m, base; gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------------
# PT-3 — Matched comparison at κ=6 (CPU): four pairs vs + fresh p5 vs + pooled q5. Dev and fresh mocks p4A/p4B/p6A/p6B.
# p5/q5 are never evaluated as held-out targets. Saves all configurations and builds val ZIPs (nothing uploaded).
# ---------------------------------------------------------------------------------------------------------------------
import zipfile, io
KAP = 6.0
ALL = ["inA", "inB", "p2A", "p2B", "p3A", "p3B", "p4A", "p4B", "p5A", "p5B", "p6A", "p6B", "q5A", "q5B"]
for n in ALL:
    if n not in REF:
        REF[n] = {ph: split_npz(OLD / f"{ph}_shadow_{n}_losses.npz") for ph in PHASES}
    if n not in MEM:
        MEM[n] = membership(load_texts(n) if "load_texts" in globals() else shadow_texts(n))
for a, b in (("p5A", "p5B"), ("q5A", "q5B")):
    for ph in PHASES:
        assert (MEM[a][ph] ^ MEM[b][ph]).all()
assert all(np.array_equal(MEM["q5" + s][ph], MEM["p5" + s][ph]) for s in "AB" for ph in PHASES), "q5 assignment ≠ p5"
tmn = lambda arrs: np.array([a.mean() for a in arrs])
ML = {n: tmn(REF[n]["train"]) for n in ALL}

def pboot(a, b, y, seed, B=2000):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); g = np.random.default_rng(seed); d = []
    for _ in range(B):
        ix = np.concatenate([g.choice(pos, pos.size), g.choice(neg, neg.size)])
        d.append(strict_fast(a[ix], y[ix]) - strict_fast(b[ix], y[ix]))
    lo, hi = np.percentile(d, [2.5, 97.5]); return f"{tpr_strict(y, a) - tpr_strict(y, b):+.3f} [{lo:+.3f}, {hi:+.3f}]"

def dev_lr(mt, names):
    P = np.vstack([np.stack([ML[n] for n in names]), FIX4])
    Pin = np.vstack([np.stack([MEM[n]["train"] for n in names]), np.zeros(FIX4.shape, bool)])
    fin = np.isfinite(P)
    return lr_frozen(mt, DEV_L, P, Pin, kappa=KAP), (Pin & fin).sum(0), (~Pin & fin).sum(0)

BASE4 = ["inA", "inB", "p2A", "p2B", "p3A", "p3B", "p4A", "p4B"]
CONF = {"4 pairs": [], "+ fresh p5": ["p5A", "p5B"], "+ pooled q5": ["q5A", "q5B"]}
rows, saved = [], {}
for tname, mt, y, excl in [("real dev", DEV_MT, LABELS, ())] + \
        [(t, ML[t], MEM[t]["train"].astype(int), (t[:2] + "A", t[:2] + "B")) for t in ("p4A", "p4B", "p6A", "p6B")]:
    base = [n for n in BASE4 if n not in excl]
    S = {c: dev_lr(mt, base + extra) for c, extra in CONF.items()}
    if tname == "real dev":
        assert np.isclose(tpr_strict(y, S["4 pairs"][0]), 0.710, atol=5e-4), "dev 4 pairs κ=6 should be 0.710"
    for c, (v, nin, nout) in S.items():
        saved[f"{tname}|{c}"] = v
        rows.append({"target": tname, "config": c, "paired refs": len(base) + len(CONF[c]),
                     "n_in (median)": int(np.median(nin)), "n_out (median)": int(np.median(nout)),
                     "TPR": tpr_strict(y, v), "AUC": roc_auc_score(y, v),
                     "Δ vs 4 pairs [CI]": "—" if c == "4 pairs" else pboot(v, S["4 pairs"][0], y, 20261026),
                     "pooled − fresh [CI]": pboot(v, S["+ fresh p5"][0], y, 20261027) if c == "+ pooled q5" else ""})
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
mocks = res[res.target != "real dev"].pivot(index="target", columns="config", values="TPR")
print("\nFresh-mock mean TPR:", mocks.mean().round(4).to_dict())
print("pooled − fresh per mock:", (mocks["+ pooled q5"] - mocks["+ fresh p5"]).round(4).to_dict())

# Validation scores for all three configurations (LR4 fixed references; train target masked on its 39 val texts).
train_ft = {norm(txt(r)) for r in read_json(ROOT / "data" / "train" / "train_finetune.json")}
in_tf_val = np.array([norm(t) in train_ft for t in TEXTS["val"]]); assert int(in_tf_val.sum()) == 39
VT = split_npz(OLD / "val_shadow_1001_results.npz", "target_losses")
VR = {"s1": split_npz(OLD / "val_shadow_1001_results.npz", "shadow_losses"), "s2": split_npz(OLD / "val_shadow_1002_losses.npz"),
      "fin": split_npz(OLD / "val_ref_final_losses.npz"), "trn": split_npz(OLD / "val_ref_train_losses.npz")}
FIXV = np.stack([tmn(VR["s1"]), tmn(VR["s2"]), tmn(VR["fin"]), np.where(in_tf_val, np.nan, tmn(VR["trn"]))])
def val_lr(names, kappa=KAP):
    P = np.vstack([np.stack([tmn(REF[n]["val"]) for n in names]), FIXV])
    Pin = np.vstack([np.stack([MEM[n]["val"] for n in names]), np.zeros(FIXV.shape, bool)])
    return lr_frozen(tmn(VT), np.array([a.size for a in VT], float), P, Pin, kappa=kappa)
sub4 = np.loadtxt(io.BytesIO(zipfile.ZipFile(DRIVE / "frozen_LR4_k8" / "project_phase2_val_submission_LR4_k8.zip").read("val/prediction.csv")))
assert np.abs(val_lr(BASE4, 8.0) - sub4).max() < 1e-4, "LR4 κ=8 val not reproduced; stop"
for c, extra in CONF.items():
    v = val_lr(BASE4 + extra); assert np.isfinite(v).all(); saved[f"val|{c}"] = v
np.savez(PT_DIR / "pt3_scores_k6.npz", **saved); res.to_csv(PT_DIR / "pt3_results.csv", index=False)
for c, tag in (("+ fresh p5", "LR5fresh_k6"), ("+ pooled q5", "LR5pooled_k6")):
    pred = ROOT / "data" / "val" / f"prediction_{tag}.csv"; zp = ROOT / f"project_phase2_val_submission_{tag}.zip"
    np.savetxt(pred, saved[f"val|{c}"], fmt="%.17g")
    for cmd in ([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--phase", "val", "--prediction", str(pred), "--output", str(zp)],
                [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(zp), "--phase", "val"]):
        rr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); assert rr.returncode == 0, rr.stderr[-800:]
    shutil.copy(zp, PT_DIR)
print("Saved scores, results and both val ZIPs to", PT_DIR, "| nothing uploaded.")

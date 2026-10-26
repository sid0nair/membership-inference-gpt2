# KR-1 — Joint comparison of reference set × variance shrinkage κ. Dev + fresh mocks only; no validation use.
# Run in any notebook that has run N-2, N-3, N-4 (e.g. 04, 07, 08). Loads p5/p6 caches itself; CPU only.
# Pre-declared (before looking at any output):
#   configs   : reference set ∈ {4 pairs (LR4's), 6 pairs} × κ ∈ {2, 8, 32, ∞}; κ=∞ uses the pooled side-specific
#               variance (× the same 1+1/n factor). Separate IN/OUT variances, √T length scaling, fixed OUT refs and
#               masks are exactly as in the frozen LR.
#   targets   : real dev target; fresh mocks p4A, p4B, p5A, p5B, p6A, p6B (fresh WikiText aux). Each mock's own
#               pair is removed from its references (so a mock sees 3 or 4 pairs under "4 pairs", 5 under "6 pairs").
#               Every target uses the same 4 fixed OUT refs as real dev (shadow1001, shadow1002, val tgt, final tgt).
#   baseline  : (4 pairs, κ=8) within each target.
#   selection : candidate = highest mean strict TPR over the 6 fresh mocks. It "improves consistently" only if
#               mean Δ > 0, Δ > 0 on ≥ 5 of 6 fresh mocks, and dev Δ ≥ 0. Otherwise: keep LR4 and move on.
import itertools
KR = DRIVE / "cache" / "kappa_refs"; KR.mkdir(parents=True, exist_ok=True)
KAPPAS = (2.0, 8.0, 32.0, np.inf)
PAIRS6 = list(PAIRS) + [("p5A", "p5B"), ("p6A", "p6B")]
REFSETS = {"4 pairs": list(PAIRS), "6 pairs": PAIRS6}
FRESH_MOCKS = ["p4A", "p4B", "p5A", "p5B", "p6A", "p6B"]
# Load p5/p6 cached losses and memberships if this runtime does not have them yet (CPU only, no model loading).
for n in ("p5A", "p5B", "p6A", "p6B"):
    if n not in REF:
        REF[n] = {ph: split_npz(OLD / f"{ph}_shadow_{n}_losses.npz") for ph in PHASES}
    if n not in MEM:
        MEM[n] = membership(shadow_texts(n))
for a, b in PAIRS6:
    assert (MEM[a]["train"] ^ MEM[b]["train"]).all() or a == "inA", (a, b)
    assert all(r.shape == t.shape for r, t in zip(REF[a]["train"], TGT["train"])), a
assert all(n in REF and n in MEM for p in PAIRS6 for n in p)

def lr_kappa(mt, L, P, Pin, kappa):
    """Frozen LR with κ as a parameter; κ=∞ → per-side pooled variance × (1+1/n). Identical to lr_frozen for finite κ."""
    if np.isfinite(kappa):
        return lr_frozen(mt, L, P, Pin, kappa=kappa)
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
    s_in, s_out = pool_in * (1 + 1 / n_in), pool_out * (1 + 1 / n_out)
    A = (-(u - mu_in) ** 2 / (2 * s_in) - 0.5 * np.log(s_in)) - (-(u - mu_out) ** 2 / (2 * s_out) - 0.5 * np.log(s_out))
    assert np.isfinite(A).all()
    return A

# κ=∞ limit check: a very large finite κ must approach the explicit pooled form.
_P = np.vstack([np.stack([mean_loss(n) for n in REF_NAMES]), FIX4])
_Pin = np.vstack([np.stack([MEM[n]["train"] for n in REF_NAMES]), np.zeros(FIX4.shape, bool)])
assert np.allclose(lr_kappa(DEV_MT, DEV_L, _P, _Pin, 1e9), lr_kappa(DEV_MT, DEV_L, _P, _Pin, np.inf), atol=1e-5)

def score_cfg(mt, refset, kappa, exclude=()):
    names = [n for p in REFSETS[refset] if not set(p) & set(exclude) for n in p]
    P = np.vstack([np.stack([mean_loss(n) for n in names]), FIX4])
    Pin = np.vstack([np.stack([MEM[n]["train"] for n in names]), np.zeros(FIX4.shape, bool)])
    return lr_kappa(mt, DEV_L, P, Pin, kappa), len(names) // 2

CFGS = list(itertools.product(REFSETS, KAPPAS))
BASE = ("4 pairs", 8.0)
klab = lambda k: "∞" if not np.isfinite(k) else f"{k:g}"
cfg_name = lambda c: f"{c[0]}, κ={klab(c[1])}"

# Target list: (name, target mean loss, labels, pair to exclude, bootstrap indices).
def strat_boot(y, seed, B=2000):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); r = np.random.default_rng(seed)
    return [np.concatenate([r.choice(pos, pos.size), r.choice(neg, neg.size)]) for _ in range(B)]
partner = {a: b for a, b in PAIRS6} | {b: a for a, b in PAIRS6}
TARGETS_KR = [("real dev", DEV_MT, LABELS, (), boot)]
for j, t in enumerate(FRESH_MOCKS):
    y = MEM[t]["train"].astype(int)
    TARGETS_KR.append((t, mean_loss(t), y, (t, partner[t]), strat_boot(y, 20261026 + j)))

rows, SC = [], {}
for tname, mt, y, excl, bidx in TARGETS_KR:
    S = {c: score_cfg(mt, c[0], c[1], excl) for c in CFGS}
    if tname == "real dev":
        assert np.allclose(S[BASE][0], LR4_DEV, rtol=0, atol=1e-6), "baseline must reproduce LR4 on dev"
    base_bt = np.array([strict_fast(S[BASE][0][ix], y[ix]) for ix in bidx])
    base_tpr = tpr_strict(y, S[BASE][0])
    for c, (v, npairs) in S.items():
        SC[f"{tname}|{cfg_name(c)}"] = v
        d = np.array([strict_fast(v[ix], y[ix]) for ix in bidx]) - base_bt
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"target": tname, "refs": c[0], "κ": klab(c[1]), "pairs used": npairs,
                     "TPR": tpr_strict(y, v), "AUC": roc_auc_score(y, v),
                     "ΔTPR vs base": tpr_strict(y, v) - base_tpr, "95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                     "P(Δ>0)": float(np.mean(d > 0))})
res = pd.DataFrame(rows)
pd.set_option("display.width", 260)
fmt = lambda x: f"{x:.4f}"
print("Per target (baseline = 4 pairs, κ=8 within each target; mocks exclude their own pair):")
for tname in res["target"].unique():
    print(f"\n— {tname}"); print(res[res.target == tname].drop(columns="target").to_string(index=False, float_format=fmt))

# Wide views: strict TPR and Δ per config × target.
res["config"] = res["refs"] + ", κ=" + res["κ"]
order = [cfg_name(c) for c in CFGS]
wide = res.pivot(index="config", columns="target", values="TPR").loc[order, ["real dev"] + FRESH_MOCKS]
dwide = res.pivot(index="config", columns="target", values="ΔTPR vs base").loc[order, ["real dev"] + FRESH_MOCKS]
awide = res.pivot(index="config", columns="target", values="AUC").loc[order, ["real dev"] + FRESH_MOCKS]
summary = pd.DataFrame({
    "fresh mean TPR": wide[FRESH_MOCKS].mean(1), "fresh mean Δ": dwide[FRESH_MOCKS].mean(1),
    "fresh min Δ": dwide[FRESH_MOCKS].min(1), "fresh wins (Δ>0)": (dwide[FRESH_MOCKS] > 0).sum(1),
    "fresh losses (Δ<0)": (dwide[FRESH_MOCKS] < 0).sum(1),
    "fresh mean AUC": awide[FRESH_MOCKS].mean(1), "dev TPR": wide["real dev"], "dev Δ": dwide["real dev"],
    "dev AUC": awide["real dev"]})
print("\nStrict TPR by config × target:"); print(wide.to_string(float_format=fmt))
print("\nΔ strict TPR vs (4 pairs, κ=8):"); print(dwide.to_string(float_format=lambda x: f"{x:+.4f}"))
print("\nSummary:"); print(summary.to_string(float_format=fmt))

# Pre-declared selection.
cand = summary["fresh mean TPR"].idxmax()
s = summary.loc[cand]
consistent = bool(cand != cfg_name(BASE) and s["fresh mean Δ"] > 0 and s["fresh wins (Δ>0)"] >= 5 and s["dev Δ"] >= 0)
print(f"\nCandidate by mean fresh-mock TPR: {cand}  | mean Δ {s['fresh mean Δ']:+.4f}, "
      f"wins {int(s['fresh wins (Δ>0)'])}/6, worst {s['fresh min Δ']:+.4f}, dev Δ {s['dev Δ']:+.4f}")
print("Improves consistently → earns ONE validation attempt" if consistent else
      "No consistent improvement → keep LR4, move on")

res.to_csv(KR / "kr1_per_target.csv", index=False); summary.to_csv(KR / "kr1_summary.csv")
np.savez(KR / "kr1_scores.npz", **SC)
json.dump({"kappas": [klab(k) for k in KAPPAS], "refsets": {k: [list(p) for p in v] for k, v in REFSETS.items()},
           "fresh_mocks": FRESH_MOCKS, "fixed_out_refs": ["shadow1001", "shadow1002", "val target", "final target"],
           "baseline": cfg_name(BASE), "candidate": cand, "consistent": consistent,
           "rule": "argmax mean fresh TPR; consistent iff mean Δ>0, Δ>0 on ≥5/6 fresh mocks, dev Δ≥0",
           "note": "p5/p6 informed earlier post-freeze experiments; fresh mocks are not untouched"},
          open(KR / "kr1_notes.json", "w"), indent=1)
print("Saved to", KR)

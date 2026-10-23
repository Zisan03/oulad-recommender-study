"""Draw Figures 1-3 of the manuscript from the result CSVs in OUT_DIR (default: results)."""
import os, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.environ.get("OUT_DIR", "results")
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
C = ["FFF-2013J", "DDD-2013J", "CCC-2014J"]
rd = lambda f: pd.read_csv(os.path.join(OUT, f))
r = rd("rq1_early_detection.csv")
fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True)
for ax, c in zip(axs, C):
    a = r[(r.course == c) & (r.signal == "activity (clicks)")]; s = r[(r.course == c) & r.signal.str.startswith("assessment")]
    ax.plot(a.when, a.auc_nonpass, color="#2c7fb8", marker="o", ms=2.5, label="Activity"); ax.fill_between(a.when, a.lo, a.hi, color="#2c7fb8", alpha=.2)
    ax.errorbar(s.when, s.auc_nonpass, yerr=[s.auc_nonpass - s.lo, s.hi - s.auc_nonpass], fmt="s", ms=3, color="#c0392b", capsize=2, label="Assessment 1, 2, 3")
    ax.axhline(.5, color="grey", ls="--", lw=.7); ax.set_title(c); ax.set_xlabel("Course day"); ax.set_ylim(.45, .9)
axs[0].set_ylabel("AUC"); axs[0].legend(frameon=False, fontsize=6.5, loc="lower right"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1.png"), dpi=300); plt.close()
m = rd("rq2_main.csv"); pr = rd("pop_recent_main.csv")
m = pd.concat([m, pr[pr.model == "pop_recent_28"]]); m = m[(m.metric == "ndcg") & (m.K == 10)]
order = ["random", "popularity", "pop_recent_28", "cbf", "cbf_type", "cf", "hybrid", "hybrid_type", "ease"]
lab = ["Random", "Popularity", "Recent popularity*", "CBF", "CBF (type only)*", "CF", "Hybrid", "Hybrid (type only)*", "EASE"]
cols = ["#7f8c8d", "#7f8c8d", "#c0392b", "#7f8c8d", "#7f8c8d", "#7f8c8d", "#7f8c8d", "#7f8c8d", "#7f8c8d"]
fig, axs = plt.subplots(1, 3, figsize=(6.3, 3.0), sharex=True)
for ax, c in zip(axs, C):
    d = m[m.course == c].set_index("model").loc[order]
    ax.barh(range(9), d["mean"], xerr=[d["mean"] - d.lo, d.hi - d["mean"]], color=cols, ecolor="black", error_kw={"lw": .7}, capsize=1.5)
    ax.set_yticks(range(9)); ax.set_yticklabels(lab if c == C[0] else [""] * 9); ax.invert_yaxis(); ax.set_title(c); ax.set_xlabel("nDCG@10"); ax.set_xticks([0, .2, .4])
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2.png"), dpi=300); plt.close()
q = rd("rq3_weak_activity.csv"); q = q[q.metric == "ndcg10"]
ci = lambda s: [float(v) for v in re.findall(r"-?\d+\.\d+", s)]
fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.7), sharey=True)
for ax, c in zip(axs, C):
    d = q[q.course == c]; wv = d[d.split == "weak vs other"].iloc[0]
    rows = [(f"Weak (n={int(wv.n_weak)})", wv.hybrid_minus_cf_weak, *ci(wv.ci_weak)), (f"Other (n={int(wv.n_other)})", wv.hybrid_minus_cf_other, *ci(wv.ci_other))]
    for g in ("low", "mid", "high"):
        a = d[d.split == f"activity {g}"].iloc[0]; rows.append((f"{g.capitalize()} activity", a.hybrid_minus_cf_weak, *ci(a.ci_weak)))
    n, mu, lo, hi = zip(*rows); mu, lo, hi = map(np.array, (mu, lo, hi))
    ax.errorbar(range(5), mu, yerr=[mu - lo, hi - mu], fmt="o", ms=3, color="black", capsize=2, lw=.8)
    ax.axhline(0, color="#c0392b", lw=.8, ls="--"); ax.set_xticks(range(5)); ax.set_xticklabels(n, rotation=50, ha="right"); ax.set_title(c)
axs[0].set_ylabel("Hybrid minus CF (nDCG@10)"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3.png"), dpi=300); plt.close()
print("figures written to", OUT)

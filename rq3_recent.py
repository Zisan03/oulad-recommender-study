"""Exploratory RQ3 check with the recent-popularity baseline (W=28). Writes rq3_recent.csv."""
import numpy as np, pandas as pd
from scipy import stats
import pipeline as P
from recent_popularity import recent_scores, PRIMARY

rows = []
for mod, pres in P.COURSES:
    c = P.build(mod, pres, 70)
    R = P.all_models(c, P.tune_lambda(mod, pres, c["cut"]))
    R["pop_recent"] = P.evaluate(recent_scores(c, PRIMARY), c["X"], c["T"])
    meta = c["meta"]; sc = meta.early.notna().values
    weak = (meta.early < P.WEAK_T).values & sc
    other = (meta.early >= P.WEAK_T).values & sc
    terc = pd.qcut(meta.n_train, 3, labels=["low", "mid", "high"]).values
    for comp in ("cf", "hybrid", "ease", "popularity"):
        for key in ("ndcg10", "recall10"):
            d = R["pop_recent"][key] - R[comp][key]
            mw, lw, hw = P.boot_ci(d[weak]); mo, lo, ho = P.boot_ci(d[other])
            rng = np.random.RandomState(1)
            bw = d[weak][rng.randint(0, weak.sum(), (2000, weak.sum()))].mean(1)
            bo = d[other][rng.randint(0, other.sum(), (2000, other.sum()))].mean(1)
            dd = bw - bo
            try: pw = stats.wilcoxon(R["pop_recent"][key][weak], R[comp][key][weak]).pvalue
            except ValueError: pw = 1.0
            rows.append(dict(course=f"{mod}-{pres}", contrast=f"pop_recent minus {comp}", metric=key,
                             n_weak=int(weak.sum()), n_other=int(other.sum()),
                             diff_weak=mw, ci_weak=f"[{lw:.3f},{hw:.3f}]", diff_other=mo, ci_other=f"[{lo:.3f},{ho:.3f}]",
                             did=float(dd.mean()), did_lo=float(np.percentile(dd, 2.5)), did_hi=float(np.percentile(dd, 97.5)),
                             mwu_p=stats.mannwhitneyu(d[weak], d[other]).pvalue, wilcoxon_weak_p=pw,
                             ndcg_weak_recent=float(R["pop_recent"][key][weak].mean()), ndcg_weak_comp=float(R[comp][key][weak].mean()),
                             ndcg_other_recent=float(R["pop_recent"][key][other].mean()), ndcg_other_comp=float(R[comp][key][other].mean())))
    # activity terciles, recent minus cf, ndcg10
    d = R["pop_recent"]["ndcg10"] - R["cf"]["ndcg10"]
    for g in ("low", "mid", "high"):
        mk = terc == g; m_, l_, h_ = P.boot_ci(d[mk])
        rows.append(dict(course=f"{mod}-{pres}", contrast=f"pop_recent minus cf, activity {g}", metric="ndcg10",
                         n_weak=int(mk.sum()), diff_weak=m_, ci_weak=f"[{l_:.3f},{h_:.3f}]"))
df = pd.DataFrame(rows)
mask = (df.contrast.str.startswith("pop_recent minus")) & df.did.notna() & (df.metric == "ndcg10")
for comp in df[mask].contrast.unique():
    sel = mask & (df.contrast == comp)
    df.loc[sel, "mwu_p_holm_over_courses"] = P.holm(df.loc[sel, "mwu_p"].values)
df.to_csv(P.OUT + "/rq3_recent.csv", index=False)
pd.set_option("display.width", 220)
k = df[mask & (df.metric == "ndcg10")]
print(k[["course", "contrast", "n_weak", "n_other", "ndcg_weak_recent", "ndcg_weak_comp", "ndcg_other_recent", "ndcg_other_comp", "diff_weak", "diff_other", "did", "did_lo", "did_hi", "mwu_p", "mwu_p_holm_over_courses", "wilcoxon_weak_p"]].round(3).to_string(index=False))
print(df[df.contrast.str.contains("activity")][["course", "contrast", "n_weak", "diff_weak", "ci_weak"]].to_string(index=False))

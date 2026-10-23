"""Exploratory add-on: time-aware ("recent") popularity baseline.

Why: the plain popularity baseline counts openings over the whole training period. In a course the
set of materials students use changes over time, so a list based on what was opened most RECENTLY
before the cutoff may be a harder baseline. This script tests that.

Design (fixed before running):
  * score of a material = number of evaluated students who opened it in the last W days before the
    cutoff (uses training-period clicks only; nothing after the cutoff is used);
  * ties (many materials have the same recent count) are broken by overall training popularity,
    using a fractional term that can never overturn a difference in recent counts;
  * W = 28 days is the PRIMARY window; 14 and 56 days are sensitivity checks;
  * these analyses were added after the main results were seen, so they are EXPLORATORY and are
    corrected in their own Holm family; the pre-specified families in pipeline.py are untouched.

Nothing in the existing result files is changed. New files (in OUT_DIR):
  pop_recent_main.csv        means and bootstrap intervals, primary cutoff (70th percentile)
  pop_recent_wilcoxon.csv    paired tests against the primary recent-popularity baseline
  pop_recent_robustness.csv  cutoffs 60 and 80, all windows

Usage (after make_clicks.py has created clicks.pkl):
  OULAD_DIR=/path/to/oulad OUT_DIR=results python recent_popularity.py
"""
import numpy as np
import pandas as pd
from scipy import stats
import pipeline as P

WINDOWS = (14, 28, 56)
PRIMARY = 28
SHOW = ("random", "popularity", "pop_recent_14", "pop_recent_28", "pop_recent_56", "cf", "ease", "hybrid")
PAIRS = (("cf", f"pop_recent_{PRIMARY}"), ("hybrid", f"pop_recent_{PRIMARY}"),
         ("ease", f"pop_recent_{PRIMARY}"), (f"pop_recent_{PRIMARY}", "popularity"))


def recent_scores(c, window):
    """Same score for every student: recent opening counts, ties broken by overall popularity."""
    d, cut, items = c["d"], c["cut"], c["items"]
    users = list(c["meta"].index)
    r = d[(d.date >= cut - window) & (d.date < cut) & d.id_student.isin(users) & d.id_site.isin(items)]
    r = r[["id_student", "id_site"]].drop_duplicates()
    recent = r.groupby("id_site").size().reindex(items).fillna(0).values.astype(float)
    total = c["X"].sum(0)
    score = recent + 0.5 * total / (total.max() + 1.0)   # fractional part is always < 0.5
    return np.tile(score, (len(c["X"]), 1))


def run_course(mod, pres, cut_pct):
    c = P.build(mod, pres, cut_pct)
    lam = P.tune_lambda(mod, pres, c["cut"])
    R = P.all_models(c, lam)
    for w in WINDOWS:
        R[f"pop_recent_{w}"] = P.evaluate(recent_scores(c, w), c["X"], c["T"])
    return c, R


if __name__ == "__main__":
    main_rows, wx_rows, rob_rows = [], [], []
    for mod, pres in P.COURSES:
        course = f"{mod}-{pres}"
        c, R = run_course(mod, pres, 70)
        for m in SHOW:
            for key, v in R[m].items():
                met = key.rstrip("0123456789")
                K = int(key[len(met):])
                mean, lo, hi = P.boot_ci(v)
                main_rows.append(dict(course=course, model=m, metric=met, K=K, mean=mean, lo=lo, hi=hi,
                                      n_users=len(v)))
        for met in P.METRICS:
            for K in P.KS:
                key = f"{met}{K}"
                for a, b in PAIRS:
                    dd = R[a][key] - R[b][key]
                    try:
                        pw = stats.wilcoxon(R[a][key], R[b][key]).pvalue
                    except ValueError:
                        pw = 1.0
                    wx_rows.append(dict(course=course, metric=met, K=K, pair=f"{a} vs {b}",
                                        mean_diff=float(dd.mean()), r_rb=P.rank_biserial(dd), p=pw))
        print("done", course)

    wx = pd.DataFrame(wx_rows)
    wx["p_holm"] = P.holm(wx.p.values)          # its own (exploratory) family
    pd.DataFrame(main_rows).to_csv(P.OUT + "/pop_recent_main.csv", index=False)
    wx.to_csv(P.OUT + "/pop_recent_wilcoxon.csv", index=False)

    for cutp in (60, 80):
        for mod, pres in P.COURSES:
            c, R = run_course(mod, pres, cutp)
            for m in SHOW:
                res = R[m]
                rob_rows.append(dict(cutoff=cutp, course=f"{mod}-{pres}", model=m, ndcg10=res["ndcg10"].mean(),
                                     recall10=res["recall10"].mean(), hit10=res["hit10"].mean(),
                                     n_users=len(res["ndcg10"])))
    pd.DataFrame(rob_rows).to_csv(P.OUT + "/pop_recent_robustness.csv", index=False)

    mm = pd.DataFrame(main_rows)
    s = mm[(mm.metric == "ndcg") & (mm.K == 10)].groupby("model")["mean"].mean().reindex(SHOW).round(3)
    print("\nMean nDCG@10 over the three courses (primary cutoff):")
    print(s.to_string())
    print("DONE")

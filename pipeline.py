"""
OULAD weakness-aware recommendation study: full pipeline.
Run: python pipeline.py   (set OULAD_DIR and OUT_DIR; see README)
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
import os
D = os.environ["OULAD_DIR"].rstrip("/") + "/"
OUT = os.environ.get("OUT_DIR", "results")
os.makedirs(OUT, exist_ok=True)
clicks = pd.read_pickle(os.path.join(OUT, "clicks.pkl"))
vle = pd.read_csv(D + "vle.csv")
asm = pd.read_csv(D + "assessments.csv")
asm["date"] = pd.to_numeric(asm["date"], errors="coerce")
sa = pd.read_csv(D + "studentAssessment.csv")
sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
info = pd.read_csv(D + "studentInfo.csv")

COURSES = [("FFF", "2013J"), ("DDD", "2013J"), ("CCC", "2014J")]
KS = (5, 10, 20)
METRICS = ("recall", "ndcg", "hit")
TIE_SEEDS = (0, 1, 2, 3, 4)
WEAK_T = 55
MIN_TRAIN, MIN_TEST, MIN_ITEM_USERS = 10, 3, 5
DISC = 1.0 / np.log2(np.arange(2, 2 + max(KS)))
IDCG = np.cumsum(DISC)


def build(mod, pres, cut_pct, restrict_before=None):
    d = clicks[(clicks.code_module == mod) & (clicks.code_presentation == pres)]
    if restrict_before is not None:
        d = d[d.date < restrict_before]
    cut = np.percentile(d.date, cut_pct)
    tr, te = d[d.date < cut], d[d.date >= cut]
    iu = tr.groupby("id_site").id_student.nunique()
    warm = iu[iu >= MIN_ITEM_USERS].index
    trp = tr[tr.id_site.isin(warm)][["id_student", "id_site"]].drop_duplicates()
    tep = te[te.id_site.isin(warm)][["id_student", "id_site"]].drop_duplicates()
    ntr = trp.groupby("id_student").size()
    users = ntr[ntr >= MIN_TRAIN].index
    trp = trp[trp.id_student.isin(users)]
    tep = tep.merge(trp.assign(s=1), how="left", on=["id_student", "id_site"])
    tep = tep[tep.s.isna()][["id_student", "id_site"]]
    tep = tep[tep.id_student.isin(users)]
    nte = tep.groupby("id_student").size()
    users = nte[nte >= MIN_TEST].index
    trp, tep = trp[trp.id_student.isin(users)], tep[tep.id_student.isin(users)]
    items, U = sorted(warm), sorted(users)
    ui = {u: i for i, u in enumerate(U)}
    ii = {s: i for i, s in enumerate(items)}
    X = np.zeros((len(U), len(items)), dtype=np.float32)
    T = np.zeros_like(X, dtype=bool)
    for u, s in trp.itertuples(index=False):
        X[ui[u], ii[s]] = 1
    for u, s in tep.itertuples(index=False):
        T[ui[u], ii[s]] = True
    # item content: activity type (+ introduction-week bucket from TRAIN data only)
    v = vle[(vle.code_module == mod) & (vle.code_presentation == pres)].set_index("id_site").activity_type
    types = pd.Series([v.get(s, "?") for s in items])
    F_type = pd.get_dummies(types).values.astype(np.float32)
    first = tr[tr.id_site.isin(items)].groupby("id_site").date.min().reindex(items).values
    wk = np.digitize(first, np.percentile(first, [25, 50, 75]))
    F_week = np.hstack([F_type, np.eye(4)[wk]]).astype(np.float32)
    # weakness: mean score on non-exam assessments due before the cutoff
    A = asm[(asm.code_module == mod) & (asm.code_presentation == pres)]
    S = sa.merge(A[["id_assessment", "assessment_type", "date"]], on="id_assessment")
    S = S[(S.date < cut) & (S.assessment_type != "Exam")]
    early = S.groupby("id_student").score.mean()
    meta = pd.DataFrame(index=U)
    meta["early"] = early.reindex(U).values
    fr = info[(info.code_module == mod) & (info.code_presentation == pres)].set_index("id_student").final_result
    meta["final"] = fr.reindex(U).values
    meta["n_train"] = X.sum(1)
    return dict(X=X, T=T, F_type=F_type, F_week=F_week, meta=meta, items=items, cut=cut, d=d, early_all=early,
                fr_all=fr)


def mm(S, X):
    A = np.where(X > 0, np.nan, S)
    mn, mx = np.nanmin(A, 1, keepdims=True), np.nanmax(A, 1, keepdims=True)
    out = (S - mn) / np.maximum(mx - mn, 1e-12)
    out[X > 0] = 0
    return out


def cbf_scores(X, F):
    Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    prof = X @ Fn
    prof /= np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-9)
    return prof @ Fn.T


def knn_scores(X):
    Xn = X / np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-9)
    S = Xn.T @ Xn
    np.fill_diagonal(S, 0)
    return X @ S


def ease_scores(X, lam=100.0):
    G = X.T @ X + lam * np.eye(X.shape[1])
    P = np.linalg.inv(G)
    B = -P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0)
    return X @ B


def evaluate(score, X, T):
    n = len(X)
    nrel = T.sum(1)
    res = {m + str(k): np.zeros(n) for m in METRICS for k in KS}
    for s in TIE_SEEDS:
        rng = np.random.RandomState(s)
        sc = score + rng.rand(*score.shape) * 1e-9
        sc[X > 0] = -np.inf
        order = np.argsort(-sc, 1)[:, :max(KS)]
        H = T[np.arange(n)[:, None], order]
        for k in KS:
            h = H[:, :k]
            res[f"recall{k}"] += h.sum(1) / nrel
            res[f"hit{k}"] += h.any(1)
            res[f"ndcg{k}"] += (h * DISC[:k]).sum(1) / IDCG[np.minimum(k, nrel) - 1]
    return {k: v / len(TIE_SEEDS) for k, v in res.items()}


def tune_lambda(mod, pres, cut_outer):
    """Choose EASE regularisation on a validation split INSIDE the training period only."""
    c = build(mod, pres, 85, restrict_before=cut_outer)
    best, best_v = None, -1
    for lam in (10.0, 100.0, 1000.0, 10000.0):
        v = evaluate(ease_scores(c["X"], lam), c["X"], c["T"])["ndcg10"].mean()
        if v > best_v:
            best, best_v = lam, v
    return best


def all_models(c, lam=100.0):
    X, T = c["X"], c["T"]
    rng = np.random.RandomState(123)
    cbf_t, cbf_w = cbf_scores(X, c["F_type"]), cbf_scores(X, c["F_week"])
    cf, ez = knn_scores(X), ease_scores(X, lam)
    models = {
        "random": rng.rand(*X.shape),
        "popularity": np.tile(X.sum(0), (len(X), 1)).astype(float),
        "cbf_type": cbf_t,
        "cbf": cbf_w,  # main CBF: type + introduction week
        "cf": cf,
        "ease": ez,
    }
    for a in (0.25, 0.5, 0.75):
        models[f"hybrid_a{a}"] = a * mm(cbf_w, X) + (1 - a) * mm(cf, X)
    models["hybrid"] = models["hybrid_a0.5"]
    models["hybrid_type"] = 0.5 * mm(cbf_t, X) + 0.5 * mm(cf, X)
    return {k: evaluate(v, X, T) for k, v in models.items()}


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for i, idx in enumerate(o):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def boot_ci(x, B=2000, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(np.mean(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


if __name__ == "__main__":
    out = {}
    primary = {}
    # ---------------- RQ1: weakness validation -----------------
    rq1 = []
    for mod, pres in COURSES:
        c = build(mod, pres, 70)
        primary[(mod, pres)] = c
        e, fr = c["early_all"], c["fr_all"]
        d = pd.DataFrame({"early": e}).join(fr.rename("final"), how="inner")
        d = d.dropna(subset=["early"])
        d["notpass"] = ~d.final.isin(["Pass", "Distinction"])
        d["fail_wd"] = d.final.isin(["Fail", "Withdrawn"])
        for t in (50, 55, 60):
            w = d.early < t
            auc = stats.mannwhitneyu(d.early[d.notpass], d.early[~d.notpass]).statistic
            auc = 1 - auc / (d.notpass.sum() * (~d.notpass).sum())
            rq1.append(dict(course=f"{mod}-{pres}", threshold=t, n_scored=len(d), weak_n=int(w.sum()),
                            weak_share=round(w.mean(), 3),
                            pass_rate_weak=round(1 - d.notpass[w].mean(), 3),
                            pass_rate_other=round(1 - d.notpass[~w].mean(), 3),
                            fail_rate_weak=round((d.final[w] == "Fail").mean(), 3),
                            withdraw_rate_weak=round((d.final[w] == "Withdrawn").mean(), 3),
                            auc_notpass=round(float(auc), 3)))
    rq1 = pd.DataFrame(rq1)
    rq1.to_csv(OUT + "/rq1_weakness.csv", index=False)

    # ---------------- RQ2: main comparison -----------------
    rows, perstudent = [], {}
    for mod, pres in COURSES:
        c = primary[(mod, pres)]
        lam = tune_lambda(mod, pres, c["cut"])
        print("tuned lambda", mod, pres, lam)
        R = all_models(c, lam)
        perstudent[(mod, pres)] = R
        for mname, res in R.items():
            for k, v in res.items():
                m, K = k.rstrip("0123456789"), int(k[len(k.rstrip("0123456789")):])
                mean, lo, hi = boot_ci(v)
                rows.append(dict(course=f"{mod}-{pres}", model=mname, metric=m, K=K, mean=mean, lo=lo, hi=hi,
                                 n_users=len(v)))
    main = pd.DataFrame(rows)
    main.to_csv(OUT + "/rq2_main.csv", index=False)

    # significance
    fr_rows, wx_rows = [], []
    for (mod, pres), R in perstudent.items():
        for m in METRICS:
            for K in KS:
                key = f"{m}{K}"
                a, b, cc = R["cbf"][key], R["cf"][key], R["hybrid"][key]
                chi, p = stats.friedmanchisquare(a, b, cc)
                fr_rows.append(dict(course=f"{mod}-{pres}", metric=m, K=K, chi2=chi, p=p))
                for n1, n2 in (("hybrid", "cf"), ("hybrid", "cbf"), ("cf", "popularity"), ("hybrid", "popularity")):
                    x, y = R[n1][key], R[n2][key]
                    dd = x - y
                    try:
                        pw = stats.wilcoxon(x, y).pvalue
                    except ValueError:
                        pw = 1.0
                    wx_rows.append(dict(course=f"{mod}-{pres}", metric=m, K=K, pair=f"{n1} vs {n2}",
                                        mean_diff=float(dd.mean()), r_rb=rank_biserial(dd), p=pw))
    fr_df, wx_df = pd.DataFrame(fr_rows), pd.DataFrame(wx_rows)
    fr_df["p_holm"] = holm(fr_df.p.values)
    wx_df["p_holm"] = holm(wx_df.p.values)
    fr_df.to_csv(OUT + "/rq2_friedman.csv", index=False)
    wx_df.to_csv(OUT + "/rq2_wilcoxon.csv", index=False)

    # ---------------- RQ3: weak vs other, activity terciles -----------------
    rq3 = []
    for (mod, pres), R in perstudent.items():
        meta = primary[(mod, pres)]["meta"]
        for K in (10,):
            for m in ("ndcg", "recall"):
                key = f"{m}{K}"
                diff = R["hybrid"][key] - R["cf"][key]
                scored = meta.early.notna().values
                weak = (meta.early < WEAK_T).values & scored
                other = (meta.early >= WEAK_T).values & scored
                if weak.sum() >= 20:
                    mw, lw, hw = boot_ci(diff[weak])
                    mo, lo_, ho = boot_ci(diff[other])
                    rng = np.random.RandomState(1)
                    bw = diff[weak][rng.randint(0, weak.sum(), (2000, weak.sum()))].mean(1)
                    bo = diff[other][rng.randint(0, other.sum(), (2000, other.sum()))].mean(1)
                    dd = bw - bo
                    mwu = stats.mannwhitneyu(diff[weak], diff[other]).pvalue
                    rq3.append(dict(course=f"{mod}-{pres}", split="weak vs other", metric=key, n_weak=int(weak.sum()),
                                    n_other=int(other.sum()), hybrid_minus_cf_weak=mw, ci_weak=f"[{lw:.4f},{hw:.4f}]",
                                    hybrid_minus_cf_other=mo, ci_other=f"[{lo_:.4f},{ho:.4f}]",
                                    diff_in_diff=float(dd.mean()),
                                    did_ci=f"[{np.percentile(dd,2.5):.4f},{np.percentile(dd,97.5):.4f}]",
                                    mwu_p=mwu))
                terc = pd.qcut(meta.n_train, 3, labels=["low", "mid", "high"]).values
                for g in ("low", "mid", "high"):
                    mk = terc == g
                    mg, lg, hg = boot_ci(diff[mk])
                    rq3.append(dict(course=f"{mod}-{pres}", split=f"activity {g}", metric=key, n_weak=int(mk.sum()),
                                    hybrid_minus_cf_weak=mg, ci_weak=f"[{lg:.4f},{hg:.4f}]"))
    pd.DataFrame(rq3).to_csv(OUT + "/rq3_weak_activity.csv", index=False)

    # ---------------- robustness: other cutoffs, CBF variants, alpha, EASE -----------------
    rob = []
    for cutp in (60, 80):
        for mod, pres in COURSES:
            c = build(mod, pres, cutp)
            R = all_models(c, tune_lambda(mod, pres, c["cut"]))
            for mname, res in R.items():
                rob.append(dict(cutoff=cutp, course=f"{mod}-{pres}", model=mname, ndcg10=res["ndcg10"].mean(),
                                recall10=res["recall10"].mean(), hit10=res["hit10"].mean(), n_users=len(res["ndcg10"])))
    pd.DataFrame(rob).to_csv(OUT + "/robustness.csv", index=False)
    print("DONE")

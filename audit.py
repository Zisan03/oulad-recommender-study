import numpy as np, pandas as pd, warnings, json
from scipy import stats
warnings.filterwarnings("ignore")
import pipeline as P
OUT = P.OUT
out={}
for mod,pres in P.COURSES:
    k=f"{mod}-{pres}"; c=P.build(mod,pres,70); d=c["d"]; cut=c["cut"]; meta=c["meta"]
    info=P.info[(P.info.code_module==mod)&(P.info.code_presentation==pres)]
    tr=d[d.date<cut]
    iu=tr.groupby("id_site").id_student.nunique(); warm=iu[iu>=P.MIN_ITEM_USERS].index
    trp=tr[tr.id_site.isin(warm)][["id_student","id_site"]].drop_duplicates(); ntr=trp.groupby("id_student").size()
    e=c["early_all"]; fr=c["fr_all"]
    flow=dict(registered=len(info),clicked_before_cutoff=int(tr.id_student.nunique()),ge10_materials=int((ntr>=10).sum()),evaluable=len(meta),
              scored=int(e.notna().sum()),flagged_weak=int((e<55).sum()),weak_evaluable=int((meta.early<55).sum()))
    # selection: flagged weak, evaluable vs not
    weak_ids=e[e<55].index; ev=set(meta.index)
    clk=tr.groupby("id_student").sum_click.sum()
    rows=[]
    for name,ids in (("evaluable",[i for i in weak_ids if i in ev]),("not evaluable",[i for i in weak_ids if i not in ev])):
        f=fr.reindex(ids)
        rows.append(dict(group=name,n=len(ids),mean_early_score=float(e.reindex(ids).mean()),median_clicks_before_cutoff=float(clk.reindex(ids).fillna(0).median()),
                         pass_rate=float(f.isin(["Pass","Distinction"]).mean()),withdrawn_rate=float((f=="Withdrawn").mean())))
    # flag precision/recall
    fl=[]
    S=pd.DataFrame({"e":e}).join(fr.rename("f"),how="inner"); S["np"]=~S.f.isin(["Pass","Distinction"])
    for t in (50,55,60):
        w=S.e<t; fl.append(dict(rule=f"early score < {t}",flagged_share=float(w.mean()),precision=float(S.np[w].mean()),recall=float((w&S.np).sum()/S.np.sum())))
    I=info.set_index("id_student"); R=P.pd.read_csv(P.D+"studentRegistration.csv"); R=R[(R.code_module==mod)&(R.code_presentation==pres)].set_index("id_student")
    un=pd.to_numeric(R.date_unregistration,errors="coerce").reindex(I.index); keep=I[un.isna()|(un>14)]
    cl14=d[d.date<14].groupby("id_student").sum_click.sum().reindex(keep.index).fillna(0)
    thr=np.percentile(cl14,20); w=cl14<=thr; npz=~keep.final_result.isin(["Pass","Distinction"])
    fl.append(dict(rule="lowest-20% activity, day 14",flagged_share=float(w.mean()),precision=float(npz[w].mean()),recall=float((w&npz).sum()/npz.sum())))
    # relaxed-threshold sensitivity for RQ3
    P.MIN_TRAIN,P.MIN_TEST=5,1
    c2=P.build(mod,pres,70); R2=P.all_models(c2,100.0); m2=c2["meta"]
    diff=R2["hybrid"]["ndcg10"]-R2["cf"]["ndcg10"]; sc=m2.early.notna().values; wk=(m2.early<55).values&sc; ot=(m2.early>=55).values&sc
    rng=np.random.RandomState(1); bw=diff[wk][rng.randint(0,wk.sum(),(2000,wk.sum()))].mean(1); bo=diff[ot][rng.randint(0,ot.sum(),(2000,ot.sum()))].mean(1); dd=bw-bo
    sens=dict(users=len(m2),weak_n=int(wk.sum()),hyb_minus_cf_weak=float(diff[wk].mean()),hyb_minus_cf_other=float(diff[ot].mean()),did=float(dd.mean()),did_ci=[float(np.percentile(dd,2.5)),float(np.percentile(dd,97.5))],mwu_p=float(stats.mannwhitneyu(diff[wk],diff[ot]).pvalue),
              hyb_minus_cf_all_ndcg10=float(diff.mean()))
    P.MIN_TRAIN,P.MIN_TEST=10,3
    out[k]=dict(flow=flow,selection=rows,flags=fl,relaxed=sens)
json.dump(out,open(OUT + "/audit.json","w"),indent=1)
for k,v in out.items():
    print("\n==",k); print("FLOW",v["flow"])
    print(pd.DataFrame(v["selection"]).round(3).to_string(index=False)); print(pd.DataFrame(v["flags"]).round(3).to_string(index=False)); print("RELAXED",{a:(round(b,4) if isinstance(b,float) else b) for a,b in v["relaxed"].items()})
r=pd.read_csv(OUT + "/rq3_weak_activity.csv"); r=r[(r.split=="weak vs other")&(r.metric=="ndcg10")]
p=r.mwu_p.values; o=np.argsort(p); adj=np.empty(3); run=0
for i,ix in enumerate(o): run=max(run,(3-i)*p[ix]); adj[ix]=min(1,run)
print("\nMWU raw",dict(zip(r.course,p.round(4))),"Holm over 3 courses",dict(zip(r.course,adj.round(3))))

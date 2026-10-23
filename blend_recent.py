"""Exploratory: does adding CF to the recent-popularity list help? score = (1-w)*recent + w*CF (min-max normalised, as in the hybrid)."""
import numpy as np, pandas as pd
from scipy import stats
import pipeline as P
from recent_popularity import recent_scores, PRIMARY
rows=[]
for mod,pres in P.COURSES:
    c=P.build(mod,pres,70); X,T=c["X"],c["T"]
    rec=recent_scores(c,PRIMARY); cf=P.knn_scores(X)
    base=P.evaluate(rec,X,T)
    for w in (0.25,0.5):
        s=(1-w)*P.mm(rec,X)+w*P.mm(cf,X)
        r=P.evaluate(s,X,T)
        for key in ("ndcg10","recall10","hit10"):
            d=r[key]-base[key]
            try: p=stats.wilcoxon(r[key],base[key]).pvalue
            except ValueError: p=1.0
            rows.append(dict(course=f"{mod}-{pres}",cf_weight=w,metric=key,blend=r[key].mean(),recent=base[key].mean(),diff=d.mean(),r_rb=P.rank_biserial(d),p=p))
df=pd.DataFrame(rows); k=df.metric=="ndcg10"
df.loc[k,"p_holm"]=P.holm(df.loc[k,"p"].values)
df.to_csv(P.OUT+"/blend_recent.csv",index=False)
print(df[k].round(4).to_string(index=False))
print(df[k].groupby("cf_weight")[["blend","recent"]].mean().round(3))

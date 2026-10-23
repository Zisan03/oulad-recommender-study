import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
import os
D=os.environ["OULAD_DIR"].rstrip("/")+"/"
OUT=os.environ.get("OUT_DIR","results")
os.makedirs(OUT,exist_ok=True)
cl=pd.read_pickle(os.path.join(OUT,"clicks.pkl"))
asm=pd.read_csv(D+"assessments.csv"); asm["date"]=pd.to_numeric(asm["date"],errors="coerce")
sa=pd.read_csv(D+"studentAssessment.csv"); sa["score"]=pd.to_numeric(sa["score"],errors="coerce")
si=pd.read_csv(D+"studentInfo.csv")
reg=pd.read_csv(D+"studentRegistration.csv"); reg["date_unregistration"]=pd.to_numeric(reg["date_unregistration"],errors="coerce")
COURSES=[("FFF","2013J"),("DDD","2013J"),("CCC","2014J")]
DAYS=[7,14,21,28,42,56,84,112]

def auc(sig,y):
    """P(signal higher for pass than for non-pass); y=1 means outcome of concern (non-pass / fail)."""
    sig=np.asarray(sig,float); y=np.asarray(y)
    r=stats.rankdata(sig); n1=(y==1).sum(); n0=(y==0).sum()
    return (r[y==0].sum()-n0*(n0+1)/2)/(n0*n1)

def boot(sig,y,B=1000,seed=0):
    rng=np.random.RandomState(seed); n=len(sig); v=[]
    for _ in range(B):
        i=rng.randint(0,n,n)
        if y[i].min()==y[i].max(): continue
        v.append(auc(sig[i],y[i]))
    return np.percentile(v,2.5),np.percentile(v,97.5)

def report(rows,course,kind,when,sig,final,n_note=""):
    d=pd.DataFrame({"s":sig,"f":final}).dropna()
    y_np=(~d.f.isin(["Pass","Distinction"])).astype(int).values
    a=auc(d.s.values,y_np); lo,hi=boot(d.s.values,y_np)
    k=d[d.f!="Withdrawn"]; y_f=(k.f=="Fail").astype(int).values
    a2=auc(k.s.values,y_f); lo2,hi2=boot(k.s.values,y_f)
    rows.append(dict(course=course,signal=kind,when=when,n=len(d),auc_nonpass=a,lo=lo,hi=hi,
                     n_stayers=len(k),auc_fail_vs_pass=a2,lo_f=lo2,hi_f=hi2))

rows=[]
for mod,pres in COURSES:
    c=f"{mod}-{pres}"
    I=si[(si.code_module==mod)&(si.code_presentation==pres)].set_index("id_student")
    R=reg[(reg.code_module==mod)&(reg.code_presentation==pres)].set_index("id_student").date_unregistration
    I["unreg"]=R.reindex(I.index)
    ck=cl[(cl.code_module==mod)&(cl.code_presentation==pres)]
    for d in DAYS:
        keep=I[I.unreg.isna()|(I.unreg>d)]
        e=ck[ck.date<d].groupby("id_student").sum_click.sum().reindex(keep.index).fillna(0)
        report(rows,c,"activity (clicks)",d,np.log1p(e.values),keep.final_result.values)
    A=asm[(asm.code_module==mod)&(asm.code_presentation==pres)&(asm.assessment_type!="Exam")].sort_values("date")
    for n in (1,2,3):
        ids=A.id_assessment.head(n).tolist(); due=float(A.date.head(n).max())
        S=sa[sa.id_assessment.isin(ids)].groupby("id_student").score.mean()
        keep=I[(I.unreg.isna()|(I.unreg>due))&I.index.isin(S.index)]
        report(rows,c,f"assessment score (first {n})",due,S.reindex(keep.index).values,keep.final_result.values)
out=pd.DataFrame(rows); out.to_csv(OUT + "/rq1_early_detection.csv",index=False)
pd.set_option("display.width",250)
t=out.copy()
for col in ("auc_nonpass","lo","hi","auc_fail_vs_pass"): t[col]=t[col].round(2)
print(t[["course","signal","when","n","auc_nonpass","lo","hi","n_stayers","auc_fail_vs_pass"]].to_string(index=False))

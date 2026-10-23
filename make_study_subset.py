"""
Build a small OULAD subset containing only the three course presentations used in the study
(FFF 2013J, DDD 2013J, CCC 2014J). Writes the six CSV files the analysis needs and zips them.

Usage:  python make_study_subset.py /path/to/oulad_folder [output_folder]
Requires: pandas.  OULAD is released under CC BY 4.0 (Kuzilek, Hlosta and Zdrahal, Scientific Data, 2017).
"""
import sys, zipfile
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else "oulad_subset"); out.mkdir(exist_ok=True)
KEEP = {("FFF", "2013J"), ("DDD", "2013J"), ("CCC", "2014J")}
mask = lambda df: pd.Series([(m, p) in KEEP for m, p in zip(df.code_module, df.code_presentation)], index=df.index)

for name in ("studentVle", "vle", "assessments", "studentInfo", "studentRegistration"):
    df = pd.read_csv(src / f"{name}.csv")
    df = df[mask(df)]
    df.to_csv(out / f"{name}.csv", index=False); print(name, len(df))
a = pd.read_csv(src / "assessments.csv"); ids = set(a[mask(a)].id_assessment)
sa = pd.read_csv(src / "studentAssessment.csv"); sa = sa[sa.id_assessment.isin(ids)]
sa.to_csv(out / "studentAssessment.csv", index=False); print("studentAssessment", len(sa))
zp = out.parent / "oulad_subset.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(out.glob("*.csv")): z.write(f, f"oulad_subset/{f.name}")
print("zip:", zp, round(zp.stat().st_size / 1e6, 1), "MB")

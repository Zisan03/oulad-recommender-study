"""Step 1: read the raw OULAD click file once and cache it. Needs OULAD_DIR (folder with studentVle.csv)."""
import os, pandas as pd
D = os.environ["OULAD_DIR"].rstrip("/") + "/"
OUT = os.environ.get("OUT_DIR", "results"); os.makedirs(OUT, exist_ok=True)
pd.read_csv(D + "studentVle.csv").to_pickle(os.path.join(OUT, "clicks.pkl"))
print("clicks cached")

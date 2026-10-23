# Supplementary code: early identification of struggling students and hybrid recommendation (OULAD)

Reproduces every table and figure of the manuscript from the raw Open University Learning Analytics Dataset (OULAD).

## 1. Get the data
Download OULAD (CC BY 4.0; Kuzilek, Hlosta and Zdrahal, Scientific Data, 2017) from https://analyse.kmi.open.ac.uk/open_dataset
(UCI mirror: https://archive.ics.uci.edu/dataset/349). Unzip it. The folder must contain
`studentVle.csv`, `vle.csv`, `assessments.csv`, `studentAssessment.csv`, `studentInfo.csv`, `studentRegistration.csv`.

## 2. Run
    pip install -r requirements.txt
    OULAD_DIR=/path/to/oulad OUT_DIR=results bash run_all.sh

Reading the 450 MB click file is the slowest step; the full run takes several minutes on a laptop.

## 3. What each script does
| Script | Output (in OUT_DIR) | Manuscript |
|---|---|---|
| make_clicks.py | clicks.pkl (cached clicks) | all |
| pipeline.py | rq1_weakness, rq2_main, rq2_friedman, rq2_wilcoxon, rq3_weak_activity, robustness (.csv) | Tables 2-4, RQ2, RQ3 |
| early.py | rq1_early_detection.csv | Figure 1, RQ1 |
| recent_popularity.py | pop_recent_main, pop_recent_wilcoxon, pop_recent_robustness (.csv) | Exploratory time-aware popularity baseline |
| rq3_recent.py | rq3_recent.csv | Exploratory RQ3 with Recent Popularity |
| blend_recent.py | blend_recent.csv | Exploratory blend of Recent Popularity and CF |
| audit.py | audit.json (participant flow, selection check, relaxed-threshold sensitivity, flag precision/recall) | Table 1, RQ1, RQ3 |
| make_figures.py | fig1-3.png (300 dpi) | Figures 1-3 |

## 4. Design settings (all in pipeline.py)
Courses FFF-2013J, DDD-2013J, CCC-2014J; cutoff at the 70th percentile of click dates (60 and 80 for robustness);
materials kept if opened by at least 5 training students; students evaluated with at least 10 training materials and 3 new later materials;
weak = mean early non-exam assessment score below 55; hybrid weights 0.5/0.5 (0.25 and 0.75 as sensitivity);
EASE lambda chosen from {10, 100, 1000, 10000} on a validation split inside the training period;
tie-break seeds 0-4; bootstrap seeds fixed in the code. Results are deterministic: the files in `results/` were reproduced from scratch with identical values.

## 5. Exploratory items
The time-aware popularity baseline (recent_popularity.py) was added after the main results were seen. Score = openings by evaluated students in the last W days before the cutoff (W = 28 primary; 14 and 56 sensitivity), ties broken by overall training popularity; it uses training-period clicks only and is tested in its own Holm family. Existing result files are not modified.
The two "type only" variants (CBF and Hybrid using activity type alone) were added after seeing results.
Everything else was fixed before the final run.

Licence: [choose a licence compatible with CC BY-SA 4.0 before publishing]. Tested with Python 3.12.3.

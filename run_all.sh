#!/usr/bin/env bash
# Usage: OULAD_DIR=/path/to/oulad OUT_DIR=results bash run_all.sh
set -e
python make_clicks.py
python pipeline.py      # RQ1 weakness table, RQ2 models + statistics, RQ3 subgroups, robustness
python early.py         # RQ1 early-detection curves
python recent_popularity.py  # exploratory: time-aware popularity baseline (adds pop_recent_*.csv)
python rq3_recent.py   # exploratory: RQ3 with the recent-popularity baseline
python blend_recent.py # exploratory: recent popularity blended with CF
python audit.py         # participant flow, selection check, relaxed-threshold sensitivity
python make_figures.py  # Figures 1-3

#----------------------------------------------------------------------------------------
# This script runs the whole comparison again under different data choices, to
# see whether the result from 10_evaluate.py holds or depends on how I set
# things up.
#----------------------------------------------------------------------------------------
# Nothing here picks a better setup. The baseline stays the baseline. The point
# is only to see how far the answer moves.
#----------------------------------------------------------------------------------------
# Equivalents in R:
#   GroupKFold               caret::groupKFold
#   numpy.lexsort            order() on several keys
#   average_precision_score  PRROC::pr.curve
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

#---Libraries Part 2---
#Load in the configuration file to ensure that all
# paths and parameters are consistent across scripts.
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "config", Path(__file__).resolve().parent / "00_config.py"
)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
#----------------------------------------------------------------------------------------

#---Load---
panel = pd.read_parquet(cfg.PROCESSED / "panel.parquet")
folds = pd.read_parquet(cfg.PROCESSED / "folds.parquet")
saved = joblib.load(cfg.PROCESSED / "model.joblib")

FEATURES = saved["features"]
PARAMS = saved["params"]  # same settings throughout, so only the data choice changes

HALF = cfg.MIN_AREA_KM2          # the baseline cut-off, 50% of a full cell
FULL = (cfg.GRID_M ** 2) / 1e6   # a whole cell, 25 km2

print("features:", len(FEATURES), "| settings:", PARAMS)

#---A fresh forest with the baseline settings---
def new_forest():
    return RandomForestClassifier(n_estimators=500, random_state=cfg.RANDOM_SEED,
                                  n_jobs=-1, **PARAMS)

#---Share of the real fires inside the top 10% of ranked cells---
# Same tie-breaking as 10_evaluate.py. burned before is a mean over 18
# cell-months, so it takes few distinct values and the 10% cut falls inside a
# large group of equal scores. Sorting alone would settle those by row order,
# making the result depend on how the table happens to be sorted. Averaging
# over random tie-breaks removes that.
# R equivalent: rank(-s, ties.method = "random")
def top_pct(yy, s, pct, n_draws=20, seed=cfg.RANDOM_SEED):
    k = int(np.ceil(len(s) * pct))
    g = np.random.default_rng(seed)
    return np.mean([yy[np.lexsort((g.random(len(s)), -s))[:k]].sum() / yy.sum()
                    for _ in range(n_draws)])

def top10(yy, s):
    return top_pct(yy, s, 0.10)

#---One full run of the comparison---
# Fits on the training years with whatever predictors, cut-off and block size
# are passed in, scores it across the blocks, then tests on 2025 against
# burned before rebuilt from the same cells.
def run(name, features, min_area, block):
    p = panel[panel["area_km2"] >= min_area].merge(folds, on="ID", how="left")

    tr = p[p["year"].isin(cfg.TRAIN_YEARS)]
    te = p[p["year"] == cfg.VAL_YEAR]

    # At 50 km the fold column from 08_folds.py is used directly, so the
    # baseline reproduces the cross-validation score in 09_train.py exactly.
    # The other two block sizes have no saved folds, so they are regrouped
    # here. Regrouping depends on row order, which is why it is avoided
    # wherever a saved assignment exists.
    cv = []
    if block == "block_50km":
        splits = [(np.where(tr["fold"] != k)[0], np.where(tr["fold"] == k)[0])
                  for k in sorted(tr["fold"].unique())]
    else:
        splits = list(GroupKFold(n_splits=5).split(tr, groups=tr[block]))

    for i, j in splits:
        m = new_forest()
        m.fit(tr.iloc[i][features], tr.iloc[i]["fire"])
        p_fold = m.predict_proba(tr.iloc[j][features])[:, 1]
        cv.append(average_precision_score(tr.iloc[j]["fire"], p_fold))

    # Then the held-out year.
    m = new_forest()
    m.fit(tr[features], tr["fire"])

    y = te["fire"].astype(int).values
    p_rf = m.predict_proba(te[features])[:, 1]
    p_hb = te["ID"].map(tr.groupby("ID")["fire"].mean()).values

    print(f"{name} done | fold ap {min(cv):.4f} to {max(cv):.4f}")
    return {
        "variant": name,
        "cells": te["ID"].nunique(),
        "cv ap": np.mean(cv),
        "2025 ap": average_precision_score(y, p_rf),
        "ap before": average_precision_score(y, p_hb),
        "2025 top10": top10(y, p_rf),
        "top10 before": top10(y, p_hb),
    }

#----------------------------------------------------------------------------------------

#---Build the predictor lists for each variant---
# Each list changes one thing from the baseline and nothing else.
no_3m = [f for f in FEATURES if not f.endswith("_3m")]

lc_grouped = [f for f in FEATURES
              if f not in ("wetland", "lc_water", "grassland", "flooded_savanna")]
lc_grouped = lc_grouped + ["flood_responsive"]

main_roads = ["dist_road_main" if f == "dist_road" else f for f in FEATURES]
rivers_only = ["dist_river" if f == "dist_water" else f for f in FEATURES]
no_area = [f for f in FEATURES if f != "area_km2"]

#---Run them---
rows = [
    run("baseline",           FEATURES,    HALF,        "block_50km"),
    run("min area 75%",       FEATURES,    FULL * 0.75, "block_50km"),
    run("min area 100%",      FEATURES,    FULL,        "block_50km"),
    run("blocks 25 km",       FEATURES,    HALF,        "block_25km"),
    run("blocks 75 km",       FEATURES,    HALF,        "block_75km"),
    run("1 month lags only",  no_3m,       HALF,        "block_50km"),
    run("land cover grouped", lc_grouped,  HALF,        "block_50km"),
    run("main roads only",    main_roads,  HALF,        "block_50km"),
    run("rivers only",        rivers_only, HALF,        "block_50km"),
    run("no cell area",       no_area,     HALF,        "block_50km"),
]

results = pd.DataFrame(rows).set_index("variant").round(4)

print("\nsensitivity results")
print(results.to_string())

#---Does the forest beat burned before on 2025?---
# Both measures are reported, since 10_evaluate.py found them disagreeing:
# the forest is ahead on average precision and level on the top 10%.
results["ap gap"] = (results["2025 ap"] - results["ap before"]).round(4)
results["top10 gap"] = (results["2025 top10"] - results["top10 before"]).round(4)

print("\nforest minus burned before on 2025")
print(results[["ap gap", "top10 gap"]].to_string())

#---Save---
cfg.OUTPUT.mkdir(parents=True, exist_ok=True)
results.to_csv(cfg.OUTPUT / "sensitivity.csv")

print("\nsaved sensitivity.csv")

#---Tests from the proposal that are not run here---
# Both need the pipeline rebuilt from raw data rather than a different slice
# of the panel, so neither fits in this script.
print("""
not run here:
  cell size 10 km      needs 01_grid.py through 07_panel.py run again on a
                       10 km grid
  fire detections      02_fire.py already drops VIIRS types 2 and 3; using
                       only the more certain detections needs that rebuilt

run separately:
  min area 0% and 25%  07_panel.py already applied the 50% cut-off, so the
                       looser cut-offs run on the unfiltered panel in
                       additional_testing/a3_area_sweep.py
""")

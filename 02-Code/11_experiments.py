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
def top10(y, s):
    k = int(np.ceil(len(s) * 0.10))
    return y[np.argsort(-s)[:k]].sum() / y.sum()

#---One full run of the comparison---
# Fits on the training years with whatever predictors, cut-off and block size
# are passed in, scores it across the blocks, then tests on 2025 against
# burned before rebuilt from the same cells.
def run(name, features, min_area, block):
    p = panel[panel["area_km2"] >= min_area].merge(folds, on="ID", how="left")

    tr = p[p["year"].isin(cfg.TRAIN_YEARS)]
    te = p[p["year"] == cfg.VAL_YEAR]

    # Blocks regrouped into five folds at whichever block size is being tested.
    cv = []
    for i, j in GroupKFold(n_splits=5).split(tr, groups=tr[block]):
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

    print(name, "done")
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

#---Does the forest ever beat burned before on 2025?---
# A negative gap means burned before wins, which is what the baseline found.
results["ap gap"] = (results["2025 ap"] - results["ap before"]).round(4)
results["top10 gap"] = (results["2025 top10"] - results["top10 before"]).round(4)

print("\nforest minus burned before on 2025")
print(results[["ap gap", "top10 gap"]].to_string())

#---Save---
cfg.OUTPUT.mkdir(parents=True, exist_ok=True)
results.to_csv(cfg.OUTPUT / "sensitivity.csv")

print("\nsaved sensitivity.csv")

#---Three tests from the proposal that are not run here---
# All three need the pipeline rebuilt from raw data rather than a different
# slice of the panel, so none of them fit in this script.
print("""
not run here:
  min area 0% and 25%  07_panel.py already applied the 50% cut-off, so a
                       looser one needs the panel rebuilt
  cell size 10 km      needs 01_grid.py through 07_panel.py run again on a
                       10 km grid
  fire detections      02_fire.py already drops VIIRS types 2 and 3; using
                       only the more certain detections needs that rebuilt
""")

'''
Ten variants, each changing one thing from the baseline.

                    cells   cv ap  2025 ap  ap before  2025 top10  top10 before
  baseline           6042  0.4380   0.0529     0.0607      0.2984        0.4070
  min area 75%       5845  0.4373   0.0511     0.0596      0.3080        0.3840
  min area 100%      5484  0.4404   0.0506     0.0577      0.2991        0.3846
  blocks 25 km       6042  0.4685   0.0529     0.0607      0.2984        0.4070
  blocks 75 km       6042  0.4117   0.0529     0.0607      0.2984        0.4070
  1 month lags only  6042  0.4007   0.0576     0.0607      0.3140        0.4070
  land cover grouped 6042  0.4434   0.0545     0.0607      0.3140        0.4070
  main roads only    6042  0.4395   0.0482     0.0607      0.2907        0.4070
  rivers only        6042  0.4384   0.0542     0.0607      0.3101        0.4070
  no cell area       6042  0.4387   0.0541     0.0607      0.3178        0.4070

Forest minus burned before on 2025, top 10%:
  -0.1086 baseline | -0.0760 min area 75% | -0.0855 min area 100%
  -0.1086 blocks 25 km | -0.1086 blocks 75 km | -0.0930 1 month lags
  -0.0930 land cover grouped | -0.1163 main roads | -0.0969 rivers only
  -0.0892 no cell area

Every variant is negative, from -0.076 to -0.116. No data choice flips the
result from 10_evaluate.py, so the conclusion does not rest on how the
pipeline was set up.

Block size changes the cross-validation score and nothing else: 0.4685 at
25 km, 0.4380 at 50 km, 0.4117 at 75 km. Smaller blocks put training and
test cells closer together, so the score is more optimistic. Block size
cannot affect 2025, since the final model trains on all the training years
regardless. This is the clearest demonstration in the project of why
blocks are used at all.

The minimum cell area turns out to matter little. Cross-validation runs
0.4373 to 0.4404 across 50%, 75% and 100% of a full cell, and the 2025 gap
narrows but never flips. The exposure question that ran through 01_grid.py,
02_fire.py and 07_panel.py does not drive the answer.

Four variants score slightly better than the baseline on 2025: grouped
land cover, no cell area, rivers only and one month lags. None closes the
gap, and none is adopted, since choosing one after seeing 2025 would be
picking the answer.

Dropping the three month lags costs the most in cross-validation, 0.4380
to 0.4007, while slightly improving 2025. Using main roads only is the
worst variant on 2025, so keeping every drivable road was the better
choice.
'''
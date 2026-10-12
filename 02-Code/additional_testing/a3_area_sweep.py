#----------------------------------------------------------------------------------------
# The full minimum cell area sweep, from 0 to 100% of a full cell. It runs on
# the unfiltered panel from a1_panel_all.py, so it can test looser cut-offs
# than 11_experiments.py, which only sees the panel after the 50% cut.
#----------------------------------------------------------------------------------------
# Nothing here picks a better setup. 50% stays the baseline. The point is only
# to see how far the answer moves.
#
# The cross-validation scores here are NOT comparable to 11_experiments.py.
# Folds come from a2_folds_all.py, assigned once over all 6,584 cells, while
# 08_folds.py assigned them over the 6,042 cells left after the 50% cut. The
# two partitions differ, and fold composition moves the score: the block size
# test in 11_experiments.py runs 0.4161 to 0.5028 on the same data. The
# baseline figure quoted in the write-up is the one from 11_experiments.py.
# What this file does support is comparing its own five rows to each other,
# since all five are scored on one partition.
#----------------------------------------------------------------------------------------
# Equivalents in R:
#   numpy.lexsort            order() on several keys
#   average_precision_score  PRROC::pr.curve
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

#---Libraries Part 2---
#Load in the configuration file to ensure that all
# paths and parameters are consistent across scripts.
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "config", Path(__file__).resolve().parents[1] / "00_config.py"
)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
#----------------------------------------------------------------------------------------

#---Load---
panel = pd.read_parquet(cfg.PROCESSED / "panel_all.parquet")
folds = pd.read_parquet(cfg.PROCESSED / "folds_all.parquet")
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
# Fits on the training years at whichever cut-off is passed in, scores it
# across the folds, then tests on 2025 against burned before rebuilt from the
# same cells.
def run(name, features, min_area):
    # The panel holds every cell, so any cut-off can be applied here. Sorting
    # fixes the row order: a forest draws its bootstrap samples by row index,
    # so a reordered table gives a slightly different model.
    p = (panel[panel["area_km2"] >= min_area]
         .merge(folds, on="ID", how="left")
         .sort_values(["ID", "year", "month"])
         .reset_index(drop=True))

    tr = p[p["year"].isin(cfg.TRAIN_YEARS)]
    te = p[p["year"] == cfg.VAL_YEAR]

    # The fold column from a2_folds_all.py, not a fresh split. Re-splitting
    # here would give a different partition for every cut-off and the five
    # rows would not be comparable.
    # Folds were assigned over all cells, so a tighter cut-off takes more out
    # of some folds than others. The sizes are printed below as a check.
    cv = []
    for k in sorted(tr["fold"].unique()):
        i = np.where(tr["fold"] != k)[0]
        j = np.where(tr["fold"] == k)[0]
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

    # How uneven the folds became at this cut-off, and how far the five fold
    # scores spread. A wide spread means the single score is not firm.
    sizes = tr["fold"].value_counts().sort_index().values
    print(f"{name} done | fold rows {sizes.min():,} to {sizes.max():,}"
          f" | fold ap {min(cv):.4f} to {max(cv):.4f}")

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
#---Run the sweep---
# Each run changes only the cut-off. Everything else is held at the baseline.
rows = [
    run("min area 0%",   FEATURES, 0.0),
    run("min area 25%",  FEATURES, FULL * 0.25),
    run("min area 50%",  FEATURES, HALF),
    run("min area 75%",  FEATURES, FULL * 0.75),
    run("min area 100%", FEATURES, FULL),
]

results = pd.DataFrame(rows).set_index("variant").round(4)

print("\nsensitivity results")
print(results.to_string())

#---Does the forest beat burned before on 2025?---
# A positive gap means the forest wins. Both measures are reported, since the
# main run found them disagreeing: the forest is ahead on average precision
# and level on the top 10%.
results["ap gap"] = (results["2025 ap"] - results["ap before"]).round(4)
results["top10 gap"] = (results["2025 top10"] - results["top10 before"]).round(4)

print("\nforest minus burned before on 2025")
print(results[["ap gap", "top10 gap"]].to_string())

#---Save---
cfg.OUTPUT.mkdir(parents=True, exist_ok=True)
results.to_csv(cfg.OUTPUT / "sensitivity_area.csv")

print("\nsaved sensitivity_area.csv")
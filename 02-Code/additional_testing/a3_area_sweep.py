#----------------------------------------------------------------------------------------
# The full minimum cell area sweep, from 0 to 100% of a full cell. It runs on
# the unfiltered panel from a1_panel_all.py, so it can test looser cut-offs
# than 11_experiments.py, which only sees the panel after the 50% cut.
#----------------------------------------------------------------------------------------
# Nothing here picks a better setup. 50% stays the baseline. The point is only
# to see how far the answer moves.
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
def top10(y, s):
    k = int(np.ceil(len(s) * 0.10))
    return y[np.argsort(-s)[:k]].sum() / y.sum()

#---One full run of the comparison---
# Fits on the training years with whatever predictors, cut-off and block size
# are passed in, scores it across the blocks, then tests on 2025 against
# burned before rebuilt from the same cells.
def run(name, features, min_area, block):
    # The panel holds every cell, so any cut-off can be applied here.
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
#---Run the sweep---
# Each run changes only the cut-off. Everything else is held at the baseline.
rows = [
    run("min area 0%",   FEATURES, 0.0,         "block_50km"),
    run("min area 25%",  FEATURES, FULL * 0.25, "block_50km"),
    run("min area 50%",  FEATURES, HALF,        "block_50km"),
    run("min area 75%",  FEATURES, FULL * 0.75, "block_50km"),
    run("min area 100%", FEATURES, FULL,        "block_50km"),
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
results.to_csv(cfg.OUTPUT / "sensitivity_area.csv")

print("\nsaved sensitivity_area.csv")

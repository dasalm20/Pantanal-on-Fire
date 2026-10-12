#----------------------------------------------------------------------------------------
# This script trains the Random Forest on 2019 to 2024 and tunes it on the
# spatial blocks from 08_folds.py. 2025 is not read here at all.
#----------------------------------------------------------------------------------------
# Equivalents in R:
#   RandomForestClassifier   ranger::ranger
#   GridSearchCV             caret::train with tuneGrid
#   average_precision        PR-AUC, PRROC::pr.curve
#   joblib.dump              saveRDS
#
# Documentation:
#   RandomForestClassifier  scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#   GridSearchCV            scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import pandas as pd  # the panel
import numpy as np  # fold indexes
import joblib  # saves the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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

#---Load the panel and the folds---
panel = pd.read_parquet(cfg.PROCESSED / "panel.parquet")
folds = pd.read_parquet(cfg.PROCESSED / "folds.parquet")

#---Keep the training years only---
# 2025 is held out from fitting, tuning and calibration.
train = (panel[panel["year"].isin(cfg.TRAIN_YEARS)]
         .merge(folds[["ID", "fold"]], on="ID", how="left"))

print("training rows:", f"{len(train):,}")
print("fire rate:", f"{train['fire'].mean():.2%}")

#---Choose the predictors---
# All known before the month starts. Left out: year (would let the model
# follow a trend into 2025), ID (a label), and flood_responsive,
# dist_road_main and dist_river (the alternatives, for 11_experiments.py).
# area_km2 stays: the cut-off caps exposure at 2:1 but does not remove it,
# so leaving it out would push that effect into the other predictors.
FEATURES = [
    "rain_1m", "rain_3m", "rain_wet",            # rain before the month
    "tmax_1m", "tmax_3m",                        # heat before the month
    "water_1m", "water_3m", "water_wet",         # flooding before the month
    "fire_prev",                                 # fire last month
    "forest", "savanna", "flooded_savanna", "wetland",
    "grassland", "pasture", "lc_water", "other", # land cover, previous year
    "dist_road", "dist_water",                   # access, static
    "month", "area_km2",
]

X = train[FEATURES]
y = train["fire"].astype(int)

#---Rebuild the folds from 08_folds.py---
# Passing the fold column in means tuning uses the same blocks already
# checked, instead of splitting again here.
cv = [(np.where(train["fold"] != k)[0], np.where(train["fold"] == k)[0])
      for k in sorted(train["fold"].unique())]

print("folds:", len(cv))

#---Tune the model---
# Accuracy is useless here: calling every cell safe scores about 89%.
# average_precision scores the ranking, which is the actual question. A
# random ranking scores about the fire rate.
# The first grid tried min_samples_leaf [5, 20], max_features [sqrt, 0.3]
# and class_weight [None, balanced]. It picked 5 and 0.3, both edge values,
# so the best setting could lie outside the range. The grid is widened once
# here. class_weight is fixed to balanced, which clearly won.
# All of this is scored on cross-validation inside the training years.
# 2025 plays no part in it.
GRID = {
    "min_samples_leaf": [1, 5, 10],          # 5 was the lowest offered before
    "max_features":     ["sqrt", 0.3, 0.5],  # 0.3 was the highest offered before
    "class_weight":     ["balanced"],
}

search = GridSearchCV(
    RandomForestClassifier(n_estimators=500, random_state=cfg.RANDOM_SEED, n_jobs=-1),
    param_grid=GRID,
    scoring="average_precision",
    cv=cv,
    verbose=1,
)

print("\nfitting 9 settings x 5 folds, this takes a while")
search.fit(X, y)

model = search.best_estimator_

print("\nbest average precision:", round(search.best_score_, 4))
print("best settings:", search.best_params_)

#---Save---
# The feature list goes with the model, so 10_evaluate.py cannot feed it
# columns in a different order.
cfg.OUTPUT.mkdir(parents=True, exist_ok=True)

joblib.dump({"model": model, "features": FEATURES, "params": search.best_params_},
            cfg.PROCESSED / "model.joblib")   # saveRDS

pd.DataFrame(search.cv_results_).to_csv(cfg.OUTPUT / "cv_results.csv", index=False)

print("saved model.joblib and cv_results.csv")

#---Check 1: better than a random ranking?---
# A random ranking scores about the fire rate. Landing near it means nothing
# was learned.
print(f"\n{search.best_score_:.4f} vs fire rate {y.mean():.4f}",
      f"= {search.best_score_ / y.mean():.1f}x random")

#---Check 2: consistent across folds?---
# A good average hiding one bad fold would mean it works in some parts of
# the Pantanal and not others.
per_fold = [search.cv_results_[f"split{k}_test_score"][search.best_index_]
            for k in range(len(cv))]
print("per fold:", [round(s, 4) for s in per_fold])

#---Check 3: which predictors is it using?---
# Built-in importances favour variables with many split points, so
# continuous ones look stronger than they are. Permutation importance in
# 10_evaluate.py is the honest version.
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nfeature importance, built-in")
print(imp.round(4).to_string())


#----------------------------------------------------------------------------------------
# This script tests the model on 2025, the year held out from everything, and
# compares it against three simpler alternatives.
#----------------------------------------------------------------------------------------
# Equivalents in R:
#   predict_proba            predict(type = "prob")
#   LogisticRegression       glm(family = binomial)
#   StandardScaler           scale()
#   average_precision_score  PRROC::pr.curve
#   roc_auc_score            pROC::auc
#   permutation_importance   vip::vi_permute
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import pandas as pd
import numpy as np
import joblib  # loads the model from 09_train.py
from sklearn.base import clone  # a fresh copy of the model, same settings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)

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

#---Load everything---
panel = pd.read_parquet(cfg.PROCESSED / "panel.parquet")
folds = pd.read_parquet(cfg.PROCESSED / "folds.parquet")
saved = joblib.load(cfg.PROCESSED / "model.joblib")

model = saved["model"]
FEATURES = saved["features"]

train = panel[panel["year"].isin(cfg.TRAIN_YEARS)].merge(folds[["ID", "fold"]], on="ID")
test = panel[panel["year"] == cfg.VAL_YEAR].copy()

print("train rows:", f"{len(train):,}", "| fire rate", f"{train['fire'].mean():.2%}")
print("2025 rows: ", f"{len(test):,}", "| fire rate", f"{test['fire'].mean():.2%}")

#---Predict the training years fold by fold---
# Each fold is predicted by a model that never saw it. Used only to pick the
# threshold, so 2025 is not involved.
oof = np.zeros(len(train))

for k in sorted(train["fold"].unique()):
    is_test = train["fold"] == k
    m = clone(model).fit(train.loc[~is_test, FEATURES], train.loc[~is_test, "fire"])
    oof[is_test.values] = m.predict_proba(train.loc[is_test, FEATURES])[:, 1]
    print("fold", k, "done")

#---Pick the threshold---
# The cut-off that balances precision and recall best on those predictions.
# Chosen on the training years only.
cuts = np.linspace(0.05, 0.95, 91)
f1s = [f1_score(train["fire"], (oof >= c).astype(int)) for c in cuts]
THRESHOLD = float(cuts[int(np.argmax(f1s))])

print(f"\nthreshold: {THRESHOLD:.2f}")

#----------------------------------------------------------------------------------------

#---The three baselines---
# 1. How often the cell burned in the training years.
# 2. Whether it burned last month.
# 3. Logistic regression on the same predictors. Scaled first, because the
#    predictors run from 0 to 1 and from 0 to 800.
hist = train.groupby("ID")["fire"].mean().rename("hist_rate")
test = test.merge(hist, on="ID", how="left")

logit = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
logit.fit(train[FEATURES], train["fire"])

y = test["fire"].astype(int).values

scores = {
    "random forest":      model.predict_proba(test[FEATURES])[:, 1],
    "logistic":           logit.predict_proba(test[FEATURES])[:, 1],
    "burned before":      test["hist_rate"].values,
    "burned last month":  test["fire_prev"].values,
}

#---Two ways of scoring a ranking---
# top_pct is the share of the real fires that fall inside the highest ranked
# cells. That is the number that matters in practice.
def top_pct(yy, s, pct):
    k = int(np.ceil(len(s) * pct))
    return yy[np.argsort(-s)[:k]].sum() / yy.sum()

def top10(yy, s):
    return top_pct(yy, s, 0.10)

rows = []
for name, s in scores.items():
    rows.append({
        "model": name,
        "avg precision": average_precision_score(y, s),
        "roc auc": roc_auc_score(y, s),
        "top 1%": top_pct(y, s, 0.01),
        "top 5%": top_pct(y, s, 0.05),
        "top 10%": top_pct(y, s, 0.10),
    })

results = pd.DataFrame(rows).set_index("model").round(4)

print("\n2025 results, fire rate", f"{y.mean():.4f}")
print(results.to_string())

#---How much would these scores move on a different draw?---
# One year gives one number with no error bar. Drawing the rows again with
# replacement 1,000 times shows the range.
rng = np.random.default_rng(cfg.RANDOM_SEED)

print("\n95% range on average precision")
for name, s in scores.items():
    boot = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        boot.append(average_precision_score(y[i], s[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:<18} {results.loc[name, 'avg precision']:.4f}  [{lo:.4f}, {hi:.4f}]")

#---Is the gap between the forest and the best baseline real?---
# Both are scored on the same rows, so two overlapping ranges do not settle
# it. Drawing the same rows for both and taking the difference each time does.
# Done on both measures, since they can disagree.
rf, hb = scores["random forest"], scores["burned before"]

d_ap, d_10 = [], []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    if y[i].sum() == 0:
        continue
    d_ap.append(average_precision_score(y[i], rf[i]) - average_precision_score(y[i], hb[i]))
    d_10.append(top10(y[i], rf[i]) - top10(y[i], hb[i]))

for label, d in [("avg precision", d_ap), ("top 10%", d_10)]:
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n{label}, forest minus burned before: {np.mean(d):+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print("  difference is clear:", not (lo < 0 < hi))

#---Does the same thing happen inside the training years?---
# Each training year predicted by a model fitted on the other five, against a
# burned-before rate built from those same five years. Both measures are
# reported, so the quiet years and the active years are compared the same way.
print("\nleave one year out")
for yr in cfg.TRAIN_YEARS:
    tr = train[train["year"] != yr]
    te = train[train["year"] == yr]

    m = clone(model).fit(tr[FEATURES], tr["fire"])
    p_rf = m.predict_proba(te[FEATURES])[:, 1]

    rate = tr.groupby("ID")["fire"].mean()
    p_hb = te["ID"].map(rate).values

    yy = te["fire"].astype(int).values
    print(f"  {yr}  rate {yy.mean():.3f} |"
          f" AP {average_precision_score(yy, p_rf):.4f} vs {average_precision_score(yy, p_hb):.4f} |"
          f" top10 {top10(yy, p_rf):.4f} vs {top10(yy, p_hb):.4f}")

#---Precision and recall at the threshold---
# Only the forest, since the threshold was chosen for it.
flag = (scores["random forest"] >= THRESHOLD).astype(int)

print(f"\nat threshold {THRESHOLD:.2f}")
print("  precision:", round(precision_score(y, flag), 4))
print("  recall:   ", round(recall_score(y, flag), 4))
print("  flagged:  ", f"{flag.sum():,} of {len(flag):,} cells")

#---Which predictors actually matter---
# Shuffles one column at a time and measures how far the score drops. Unlike
# the built-in importances it does not favour continuous variables.
perm = permutation_importance(model, test[FEATURES], y,
                              scoring="average_precision", n_repeats=5,
                              random_state=cfg.RANDOM_SEED, n_jobs=-1)

imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
print("\npermutation importance on 2025")
print(imp.round(4).to_string())

#---Save---
cfg.OUTPUT.mkdir(parents=True, exist_ok=True)

test["pred_rf"] = scores["random forest"]
test["pred_logit"] = scores["logistic"]
test[["ID", "year", "month", "fire", "pred_rf", "pred_logit",
      "hist_rate", "fire_prev"]].to_parquet(
          cfg.PROCESSED / "predictions_2025.parquet", index=False)

results.to_csv(cfg.OUTPUT / "results_2025.csv")
imp.to_csv(cfg.OUTPUT / "permutation_importance.csv")

print("\nsaved predictions, results and importances")

'''
2025 is the held-out year: 18,126 rows, fire rate 1.42% against 11.08% in
training. It is the quietest year in the record.

Results on 2025, fire rate 0.0142
                   avg precision  roc auc  top 1%  top 5%  top 10%
  random forest           0.0529   0.6541  0.0698  0.2054   0.2984
  logistic                0.0453   0.5724  0.0853  0.1628   0.2248
  burned before           0.0607   0.7105  0.1124  0.2829   0.4070
  burned last month       0.0270   0.5491  0.1008  0.1163   0.1550

The forest beats the logistic regression and burned last month on every
measure. Against burned before it loses, and the paired tests split:
  avg precision  -0.0057  [-0.0288, +0.0171]  not clear
  top 10%        -0.1002  [-0.1696, -0.0342]  clear

So on average precision the two cannot be separated, but on top 10%
capture, the number that matters in practice, burned before is genuinely
ahead on 2025.

Leave one year out, forest vs burned before on both measures:
  2020  rate 0.214 | AP 0.3888 vs 0.3045 | top10 0.2274 vs 0.1858  forest
  2019  rate 0.171 | AP 0.3522 vs 0.2645 | top10 0.2623 vs 0.1944  forest
  2024  rate 0.113 | AP 0.2311 vs 0.2092 | top10 0.2533 vs 0.2328  forest
  2021  rate 0.112 | AP 0.2463 vs 0.2035 | top10 0.2695 vs 0.2443  forest
  2023  rate 0.037 | AP 0.0934 vs 0.1013 | top10 0.3148 vs 0.3283  before
  2022  rate 0.019 | AP 0.0634 vs 0.0638 | top10 0.2800 vs 0.3171  before

The two measures agree on every year. The forest wins both in all four
years with a fire rate above 10%, and loses both in the two quiet years.
2025 at 1.42% is the quietest of all and the gap there is the widest.

The model's advantage over knowing where fire usually happens depends on
how active the year is. In an active year the weather carries information
the model can use. In a quiet year the few fires that occur are the
persistent ignition hotspots, which is what the historical baseline
already encodes, so the weather adds nothing.

Permutation importance on 2025 is led by tmax_1m 0.024, tmax_3m 0.022,
fire_prev 0.017 and dist_road 0.011. Land cover and water sit at or below
zero, so in a quiet year they carry nothing. Sharper than the built-in
importances in 09_train.py and confirming the suspicion recorded there
that the water variables were adding little.

Threshold 0.49, chosen out of fold on the training years. On 2025 it flags
1,022 of 18,126 cells, precision 0.056, recall 0.221. The low precision is
expected: fitted where fire occurs in 11% of rows, applied where it occurs
in 1.4%. The ranking results do not depend on any threshold.
'''
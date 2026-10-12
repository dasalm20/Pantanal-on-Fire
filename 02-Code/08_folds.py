#----------------------------------------------------------------------------------------
# This script builds the splits used to test the model: spatial blocks of
# neighbouring cells, and one fold per training year.
#----------------------------------------------------------------------------------------
# A fold is one slice of the data held back for testing while the model learns
# from the rest. With five folds the data is cut into five parts, the model
# trains on four and is tested on the fifth, and that repeats five times, so
# every row gets tested once by a model that never saw it.
#----------------------------------------------------------------------------------------
# Equivalents in R:
#   //  (integer division)   %/%
#   GroupKFold               caret::groupKFold, or blockCV::cv_spatial
#   .nunique                 dplyr::n_distinct
#   .value_counts            dplyr::count
#
# Documentation:
#   GroupKFold  scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import pandas as pd  # the table we build
import geopandas as gpd  # the grid, for cell coordinates
from sklearn.model_selection import GroupKFold  # keeps whole blocks together

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

#---Load the grid and the panel---
grid  = gpd.read_file(cfg.OUTPUTS["grid"], layer="grid")
panel = pd.read_parquet(cfg.PROCESSED / "panel.parquet")

#---Build the spatial blocks---
# Fire is clustered, so neighbouring cells look alike. A random split would
# put a cell in training and its neighbour in testing, and the model would
# score well by recognising the neighbourhood rather than predicting
# anything. Holding out whole blocks stops that.
# The block ID is the coordinate divided by the block size and rounded down,
# so cells in the same square share a number. No algorithm to tune.
# rep_x and rep_y are in metres (EPSG:31981), set in 01_grid.py.
# 50 km is the baseline; 25 and 75 km feed the sensitivity test in
# 11_experiments.py.
BLOCK_SIZES_M = [25_000, 50_000, 75_000]

blocks = grid[["ID", "rep_x", "rep_y"]].copy()

for size in BLOCK_SIZES_M:
    bx = (blocks["rep_x"] // size).astype(int)   # %/%
    by = (blocks["rep_y"] // size).astype(int)
    blocks[f"block_{size // 1000}km"] = bx * 100_000 + by   # one number per square

for size in BLOCK_SIZES_M:
    col = f"block_{size // 1000}km"
    per = blocks[col].value_counts()
    print(f"{col:<12} {blocks[col].nunique():>4} blocks | cells per block: "
          f"min {per.min()}, median {int(per.median())}, max {per.max()}")

#---Assign each block to a fold---
# GroupKFold splits on the block, never the cell, so every cell in a block
# lands in the same fold.
# Training years only. 2025 is held out completely and sits in no fold.
N_FOLDS = 5
BASELINE = "block_50km"

train = panel[panel["year"].isin(cfg.TRAIN_YEARS)].merge(blocks, on="ID", how="left")

gkf = GroupKFold(n_splits=N_FOLDS)
train["fold"] = -1
for k, (_, test_idx) in enumerate(gkf.split(train, groups=train[BASELINE])):
    train.iloc[test_idx, train.columns.get_loc("fold")] = k

#---Save---
# One row per cell. A cell always sits in the same block, so it always sits
# in the same fold.
folds = (train[["ID", "fold"]].drop_duplicates()
         .merge(blocks.drop(columns=["rep_x", "rep_y"]), on="ID", how="left"))

folds.to_parquet(cfg.PROCESSED / "folds.parquet", index=False)

print("\nrows:", f"{len(folds):,}")
print("columns:", list(folds.columns))

#---Check 1: does every cell get exactly one fold?---
# A cell appearing twice would mean it sits in two blocks, which cannot
# happen and would mean the block IDs are wrong.
print("\ncells:", f"{folds['ID'].nunique():,} | rows:", f"{len(folds):,}")
print("every cell once:", folds["ID"].nunique() == len(folds))

#---Check 2: is each fold big enough, and does each one contain fire?---
# Fire is rare, so a fold with no fires cannot be scored. A fold with very
# few would mean changing the block size or the number of folds.
by_fold = train.groupby("fold").agg(
    rows=("fire", "size"),
    fires=("fire", "sum"),
    rate=("fire", "mean"),
    blocks=(BASELINE, "nunique"),
    cells=("ID", "nunique"),
)
by_fold["rate"] = (by_fold["rate"] * 100).round(2)
print("\nfolds on", BASELINE)
print(by_fold.to_string())

#---Check 3: leave one year out---
# The second test holds out a whole training year instead of a block. Each
# year needs enough fires to be worth scoring.
by_year = panel[panel["year"].isin(cfg.TRAIN_YEARS)].groupby("year").agg(
    rows=("fire", "size"), fires=("fire", "sum"), rate=("fire", "mean"),
)
by_year["rate"] = (by_year["rate"] * 100).round(2)
print("\nleave one year out")
print(by_year.to_string())

'''
rows: 6,042, one per cell kept

Blocks:
  25 km  319 blocks | median 25 cells
  50 km   98 blocks | median 87 cells
  75 km   47 blocks | median 146 cells

Check 1: every cell appears exactly once, so no cell sits in two blocks.

Check 2, folds on 50 km blocks:
  Even at 21,744 to 21,762 rows, 18 to 20 blocks, about 1,208 cells each.
  Fire rate runs 9.49% to 12.20%, so no fold is unusually easy or hard.
  Every fold holds over 2,000 fires, so all five can be scored.

Check 3, leave one year out:
  2019 17.06% | 2020 21.35% | 2021 11.20%
  2022  1.93% | 2023  3.66% | 2024 11.28%
  2022 (350 fires) and 2023 (664) are thin against 2020 (3,870), so those
  two years will score noisily. Noted for 10_evaluate.py.
'''
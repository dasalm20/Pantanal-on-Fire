#----------------------------------------------------------------------------------------
# This script joins everything built so far into one table, builds the lagged
# predictors, and applies the minimum cell area.
#----------------------------------------------------------------------------------------
# Equivalents in R, since that is where I usually work:
#   pd.read_parquet      arrow::read_parquet
#   gpd.read_file        sf::st_read
#   .merge               dplyr::left_join
#   .groupby             dplyr::group_by
#   .shift(1)            dplyr::lag
#   .rolling(3).sum()    zoo::rollsum / slider::slide_dbl
#   .agg                 dplyr::summarise
#   .isin                %in%
#   .isna().sum()        colSums(is.na(x))
#   .nunique             dplyr::n_distinct
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import pandas as pd  # the joins and the lags
import geopandas as gpd  # the grid, for cell area

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

#---Load everything the earlier scripts produced---
# No raw data here. A column renamed upstream fails now, not mid join.
grid      = gpd.read_file(cfg.OUTPUTS["grid"], layer="grid")   # sf::st_read
fire      = pd.read_parquet(cfg.OUTPUTS["fire"])               # arrow::read_parquet
climate   = pd.read_parquet(cfg.OUTPUTS["climate"])
water     = pd.read_parquet(cfg.PROCESSED / "water_month.parquet")
landcover = pd.read_parquet(cfg.PROCESSED / "landcover_year.parquet")
access    = pd.read_parquet(cfg.PROCESSED / "access.parquet")

for name, df in [("grid", grid), ("fire", fire), ("climate", climate),
                 ("water", water), ("landcover", landcover), ("access", access)]:
    print(f"{name:<10} {len(df):>8,} rows | {list(df.columns)}")

#---Put the column names onto one standard---
# The earlier scripts named their columns independently. Everything from
# here on uses rain and fire.
RENAME = {
    "precip":      "rain",   # 03_climate.py
    "fire_binary": "fire",   # 02_fire.py, the 0/1 indicator
}
climate = climate.rename(columns=RENAME)   # dplyr::rename
fire = fire.rename(columns=RENAME)

# landcover already has a column called water, which would read as the same
# thing as the monthly water share. Renamed so the two cannot be confused.
landcover = landcover.rename(columns={"water": "lc_water"})

NEEDED = {"climate": (climate, ["ID", "year", "month", "rain", "tmax"]),
          "fire":    (fire,    ["ID", "year", "month", "fire"]),
          "water":   (water,   ["ID", "year", "month", "water_pct"])}
for name, (df, cols) in NEEDED.items():
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing {missing}, fix RENAME above")

#----------------------------------------------------------------------------------------

#---Build one monthly table---
# All four sit on cell and month, so they join on the same keys.
monthly = (climate[["ID", "year", "month", "rain", "tmax"]]
           .merge(water[["ID", "year", "month", "water_pct"]],      # dplyr::left_join
                  on=["ID", "year", "month"], how="left")
           .merge(fire[["ID", "year", "month", "fire"]],
                  on=["ID", "year", "month"], how="left"))

monthly["t"] = monthly["year"] * 12 + monthly["month"]  # running month count, so a lag is one row
monthly = monthly.sort_values(["ID", "t"]).reset_index(drop=True)   # dplyr::arrange

print("\nmonthly table:", f"{len(monthly):,} rows,",
      monthly["year"].min(), "to", monthly["year"].max())

#---Build the lagged predictors---
# Every one is shifted back a month, so a row never sees the month being
# predicted. rolling(3) ends on the current month, so shift(1) moves the
# window to the three months before it.
# transform keeps the shift inside each cell. Shifting the grouped result
# instead would pull the previous cell's last value into the next cell.
g = monthly.groupby("ID")   # dplyr::group_by

monthly["rain_1m"]   = g["rain"].shift(1)        # dplyr::lag
monthly["tmax_1m"]   = g["tmax"].shift(1)
monthly["water_1m"]  = g["water_pct"].shift(1)
monthly["fire_prev"] = g["fire"].shift(1)

monthly["rain_3m"]  = g["rain"].transform(lambda s: s.rolling(3).sum().shift(1))
monthly["tmax_3m"]  = g["tmax"].transform(lambda s: s.rolling(3).mean().shift(1))
monthly["water_3m"] = g["water_pct"].transform(lambda s: s.rolling(3).mean().shift(1))

#---Build the wet season predictors---
# The wet season crosses the new year, so Nov and Dec are labelled with the
# year after. Each season then has one label: the year it feeds.
# An August 2020 row uses Nov 2019 to Mar 2020, finished long before.
wet = monthly[monthly["month"].isin(cfg.WET_MONTHS)].copy()   # %in%
wet["season"] = wet["year"] + (wet["month"] >= 11).astype(int)

wet = (wet.groupby(["ID", "season"])
          .agg(rain_wet=("rain", "sum"), water_wet=("water_pct", "mean"))  # dplyr::summarise
          .reset_index())

#----------------------------------------------------------------------------------------

#---Cut down to the months being predicted---
# Aug to Oct, 2019 to 2025. Earlier months exist only to feed the lags.
YEARS = cfg.TRAIN_YEARS + [cfg.VAL_YEAR]

panel = monthly[monthly["year"].isin(YEARS)
                & monthly["month"].isin(cfg.TARGET_MONTHS)].copy()

print("target rows before joins:", f"{len(panel):,}")

#---Attach the wet season, land cover, access and cell area---
panel = panel.merge(wet, left_on=["ID", "year"], right_on=["ID", "season"], how="left")
panel = panel.drop(columns="season")

lc = landcover.copy()
lc["year"] = lc["year"] + 1   # last year's land cover, so the target year never leaks in

# 05_landcover.py found wetland, water, grassland and flooded savanna swinging
# against each other between years: MapBiomas labels the same ground as
# wetland or water in a wet year and grassland in a dry one. Their sum is
# stable, so it is kept here for the alternative grouping in 11_experiments.py.
lc["flood_responsive"] = lc[["wetland", "lc_water", "grassland", "flooded_savanna"]].sum(axis=1)

panel = panel.merge(lc, on=["ID", "year"], how="left")

panel = panel.merge(access, on="ID", how="left")        # static, one 2020 snapshot
panel = panel.merge(grid[["ID", "area_km2"]], on="ID", how="left")

#---Apply the minimum cell area---
# Set in the config at 50% of a full cell, after 01_grid.py measured the
# spread and 02_fire.py measured its effect. This one line is all that
# changes when 11_experiments.py varies it from 0 to 100%.
before = len(panel)
panel = panel[panel["area_km2"] >= cfg.MIN_AREA_KM2].copy()

print(f"\nminimum cell area: {cfg.MIN_AREA_KM2:.2f} km2")
print(f"rows dropped: {before - len(panel):,} of {before:,}",
      f"({(before - len(panel)) / before:.1%})")
print("cells kept:", f"{panel['ID'].nunique():,} of {len(grid):,}")   # n_distinct

#---Save---
panel = panel.drop(columns=["t", "rain", "tmax", "water_pct"])  # unlagged, not predictors
panel.to_parquet(cfg.PROCESSED / "panel.parquet", index=False)   # arrow::write_parquet

print("\nrows:", f"{len(panel):,}")
print("columns:", len(panel.columns))

#---Check 1: is anything missing?---
# A gap means a join failed or a lag ran off the start of the record.
missing = panel.isna().sum()   # colSums(is.na(x))
print("\ncolumns with missing values")
print(missing[missing > 0].to_string() if missing.any() else "none")

#---Check 2: do the lags only look backwards?---
# One row printed so it can be traced back to the monthly table by hand.
row = panel[(panel["year"] == 2020) & (panel["month"] == 9)].iloc[0]
print("\none row, cell", int(row["ID"]), "September 2020")
print(row[["fire", "fire_prev", "rain_1m", "rain_3m", "rain_wet",
           "tmax_1m", "water_1m", "water_wet"]].to_string())

# And the same check across the whole panel. On Aug, Sep and Oct rows,
# rain_1m is Jul, Aug and Sep rainfall, so all three must be dry season.
print("\nmean rain_1m by target month, mm")
print(panel.groupby("month")["rain_1m"].mean().round(1).to_string())

#---Check 3: how often does fire occur?---
print("\nfire rate by year, %")
print((panel.groupby("year")["fire"].mean() * 100).round(2).to_string())

#---Check 4: is the August rainfall spread real?---
# Check 2 threw up 114.7 mm of August rain in one cell, which looked far too
# high for the dry season. Printed here so the spread is on the record.
aug = monthly[(monthly["year"] == 2020) & (monthly["month"] == 8)]
print("\nAugust 2020 rain across cells, mm")
print(aug["rain"].describe().round(1).to_string())

'''
rows: 126,882  (6,042 cells x 7 years x 3 months)
columns: 26
missing values: none

Check 2, lags look backwards only:
  Cell 2, September 2020. rain_1m = 114.72, which is exactly August's rain
  in the monthly table, so the shift lands on the right month.
  Across the panel, mean rain_1m is 3.3 mm on August rows (July rain),
  24.7 mm on September rows and 29.5 mm on October rows. All dry season.
  The 3.3 mm matches the 3.7 mm July mean from 03_climate.py, the small
  difference being the 542 dropped cells.

Check 3, fire rate in the target months:
  2019 17.06 | 2020 21.35 | 2021 11.20 | 2022 1.93
  2023  3.66 | 2024 11.28 | 2025  1.42

Check 4, the 114.7 mm value:
  August 2020 rain runs 0 to 122.9 mm across the biome, median 7.7 mm.
  Cell 2 sits at the 99.9th percentile but is not isolated. Its
  representative point is at about 22 degrees south, the far southern edge
  of the biome, where winter fronts from the south bring rain and the dry
  season is weaker than in the north. A real north to south gradient, not
  an error.
'''
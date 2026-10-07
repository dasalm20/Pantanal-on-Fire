#----------------------------------------------------------------------------------------
# This script is used to turn the VIIRS detection CSV into the project's response 
# variable: one row per cell per month, with a fire count and a binary flag.
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import pandas as pd  # the CSV and the panel
import geopandas as gpd  # points and the spatial join
import matplotlib.pyplot as plt # for plotting the checks

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

#---Read in data ---
firms = pd.read_csv(cfg.INPUTS["firms"])  # Plain CSV, not spatial

# Inspect the columns and value counts 
print(firms.columns)
print(firms["type"].value_counts())
print(firms["confidence"].value_counts())
print(firms["instrument"].value_counts())
print(firms["satellite"].value_counts())

#---Load the grid built in 01_grid.py---
grid = gpd.read_file(cfg.OUTPUTS["grid"], layer="grid")

#---Keep only real vegetation fires---
# The satellite flags what kind of hot spot it saw:
#   0 = a vegetation fire, what we want
#   2 = a permanent heat source on land, like a gas flare or a factory.
#       These show up in the same place every single month, so the model
#       would learn that spot is always on fire. We drop them.
#   3 = a hot spot over water. Not a fire. We drop these too.
#
# The satellite also rates how sure it is: low, nominal or high.
# We keep all three for now. Dropping the low ones is a judgement call,
# so instead of guessing we test it later in the sensitivity analysis.
before = len(firms)
firms = firms[firms["type"] == 0].copy()
print(f"type filter: {before:,} -> {len(firms):,} detections")

#---Keep only detections near the Pantanal---
# The file covers all of Brazil. Building 9.8 million points and joining
# them all is slow and uses a lot of memory. The grid's bounding box in
# lat/lon throws away most of the country in one cheap step, before any
# geometry is built. The spatial join below does the exact cut.
before = len(firms)
bounds = grid.to_crs("EPSG:4326").total_bounds  # minx, miny, maxx, maxy
firms = firms[
    firms["longitude"].between(bounds[0], bounds[2]) &
    firms["latitude"].between(bounds[1], bounds[3])
].copy()
print(f"bounding box: {before:,} -> {len(firms):,} detections")

#---Extract year and month from acquisition date---
firms["acq_date"] = pd.to_datetime(firms["acq_date"])
firms["year"] = firms["acq_date"].dt.year
firms["month"] = firms["acq_date"].dt.month

#---Convert the CSV to a GeoDataFrame---
firms_gdf = gpd.GeoDataFrame(
    firms,
    geometry=gpd.points_from_xy(firms.longitude, firms.latitude),
    crs="EPSG:4326"
)  # FIRMS is always WGS84

#Reproject the GeoDataFrame to the defined CRS for the project, which is in meters.
firms_gdf = firms_gdf.to_crs(cfg.CRS)  

#---Spatial join with the grid---
firms_gdf = gpd.sjoin(firms_gdf,               
                      grid[["ID","geometry"]],
                       predicate="within")  # Drops everything outside the Pantanal

#---Check what the Pantanal itself contains---
# The counts printed at the top were for all of Brazil. The mix inside the
# biome can be different, and that is the number worth recording.
print("detections inside the Pantanal:", f"{len(firms_gdf):,}")
print(firms_gdf["confidence"].value_counts())

#---Count the number of fires per cell per month---
firms_count = firms_gdf.groupby(["ID","year","month"]).size()

#---Complete the panel by reindexing to include all combinations of grid cell, year and month---
# groupby only produces rows where a fire happened. A cell with no fire in a
# given month simply does not exist yet. Without this step every row would be
# a fire and the model would have nothing to learn from.
years = range(2019, cfg.VAL_YEAR + 1)  #Range is exclusive of the stop value, so we add 1 to include the last year.

full = pd.MultiIndex.from_product( #this creates a MultiIndex with all combinations of the three levels.
    [grid["ID"], years, range(1, 13)],
    names=["ID", "year", "month"], 
)

 #---This will fill in any missing combinations with a count of 0, 
 # so that the model sees that there were no fires in those cells and months---  
firms_count = firms_count.reindex(full, fill_value=0)

#---Build the final table: one row per cell per month---
# Keep the count as well as the flag. The count is needed later for the
# previous-month fire predictors and for descriptive maps.
panel = firms_count.rename("fire_count").reset_index()
panel["fire_binary"] = (panel["fire_count"] > 0).astype(int)

#--Final checks --
print("rows:", len(panel), "expected:", len(grid) * len(years) * 12)
print("cell-months with fire:", panel["fire_binary"].sum())
print("prevalence:", panel["fire_binary"].mean())

#---Does a small cell record fewer fires just because it is small?---
# This is the evidence to drop cells later with the minimum-area rule. Two tests:
#
# 1. Detection rate by cell size. If small cells record fewer fires, either
#    they are safer, or there is simply less ground on which to see a fire.
#
# 2. The same thing divided by area. If dividing by area flattens the pattern,
#    then the first table was mostly about exposure, not about fire risk.
#
# What we found: the raw rate rises about 12-fold with cell size, and dividing
# by area cuts that to about 2.4-fold. So most of the gap is exposure, but a
# smaller real difference remains, which fits the different land cover and
# terrain at the edge of the biome. Either way, edge cells are not comparable
# to full cells. The cut-off itself is applied in 07_panel.py.
chk = panel.merge(grid[["ID", "area_km2"]], on="ID")
chk["area_bin"] = pd.cut(chk["area_km2"], [0, 1, 5, 12.5, 20, 25])
chk["rate_per_km2"] = chk["fire_count"] / chk["area_km2"]

print(chk.groupby("area_bin", observed=True)["fire_binary"].mean())
print(chk.groupby("area_bin", observed=True)["rate_per_km2"].mean())

#---Plot to have a visual reference--
chk["bin_mid"] = pd.cut(chk["area_km2"], bins=25).apply(lambda b: b.mid)
by_size = chk.groupby("bin_mid", observed=True)[["fire_binary", "rate_per_km2"]].mean()

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(by_size.index, by_size["fire_binary"], color="#15594d",
        marker="o", markersize=3, label="Share of months with a fire")
ax.plot(by_size.index, by_size["rate_per_km2"], color="#9c4326",
        marker="o", markersize=3, linestyle="--", label="Detections per km²")

# Candidate cut-off: half of a full cell.
ax.axvline(12.5, color="#555", linewidth=1, linestyle=":")
ax.text(12.5, ax.get_ylim()[1], " candidate cut-off\n 12.5 km²",
        ha="left", va="top", fontsize=8, color="#555")

ax.set_xlabel("Cell area (km²)")
ax.set_ylabel("Mean per cell-month")
ax.set_title("Small cells record fewer fires, mostly because they cover less ground",
             fontsize=11, loc="left")
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

fig.savefig(cfg.OUTPUTS["figures"] / "fire_rate_by_cell_area.png", dpi=300)
plt.show()

#---Seasonal and yearly pattern---
# A sanity check on the data. Fires should peak in the dry season,
# August to October, and 2020 was a well documented extreme fire year.
print(panel.groupby("month")["fire_binary"].mean())
print(panel.groupby("year")["fire_binary"].mean())

#Save the panel to a parquet file for later use in the panel model.
#Parquet is a very efficient format for storing large datasets,
#  and it preserves the data types and structure of the DataFrame.
panel.to_parquet(cfg.OUTPUTS["fire"], index=False)
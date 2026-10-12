#----------------------------------------------------------------------------------------
# This script works out how much of each cell was covered by each land cover
# class, for every year from 2018 to 2025.
#----------------------------------------------------------------------------------------
# Functions used here that are not basic Python, and where they are documented:
#
#   rasterio.windows.from_bounds  reads only part of a raster
#                                 rasterio.readthedocs.io/en/stable/api/rasterio.windows.html
#   rasterio.features.rasterize   turns polygons into a raster of values
#                                 rasterio.readthedocs.io/en/stable/api/rasterio.features.html
#   numpy.bincount                counts how many times each integer appears
#                                 numpy.org/doc/stable/reference/generated/numpy.bincount.html
#   array indexing with an array  used as a lookup table from code to group
#                                 numpy.org/doc/stable/user/basics.indexing.html
#   DataFrame.to_parquet          saves a table in Parquet format
#                                 pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html
#
# Class codes and names come from the published Collection 11 legend table:
# CodigosDeLegenda_LegendCodes_MapBiomas_Brazil_Collection11.pdf
# brasil.mapbiomas.org/codigos-de-legenda/
#
# Zonal counting with rasterize plus bincount is the same approach used in
# 04_water.py, and the method rasterstats uses:
# pythonhosted.org/rasterstats/manual.html
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import pandas as pd  # the table we build
import geopandas as gpd  # the grid
import numpy as np  # counting pixels per cell
import rasterio  # reads the land cover rasters
from rasterio.windows import from_bounds  # cuts the window over the study area
from rasterio.features import rasterize  # paints the cell IDs onto the raster grid

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

#---Load the grid built in 01_grid.py---
grid = gpd.read_file(cfg.OUTPUTS["grid"], layer="grid")

#---Inspect the MapBiomas files---
# These may cover all of Brazil rather than just the Pantanal. If so the
# raster is far too large to read whole, and only the window covering the
# study area should be read. Checking the bounds and shape tells us which.
files = sorted(cfg.INPUTS["landcover"].glob("*.tif"))
print(len(files), "files")
for f in files:
    print(" ", f.name)

with rasterio.open(files[0]) as src:
    print("\nbounds:", src.bounds)
    print("shape: ", src.shape)
    print("pixels:", f"{src.shape[0] * src.shape[1]:,}")
    print("bands: ", src.count)
    print("crs:   ", src.crs, "| res:", src.res)
    print("dtype: ", src.dtypes[0], "| nodata:", src.nodata)

'''
8 files
  brazil_coverage-col11_2018.tif
  brazil_coverage-col11_2019.tif
  brazil_coverage-col11_2020.tif
  brazil_coverage-col11_2021.tif
  brazil_coverage-col11_2022.tif
  brazil_coverage-col11_2023.tif
  brazil_coverage-col11_2024.tif
  brazil_coverage-col11_2025.tif

bounds: BoundingBox(left=-74.02099974839176, bottom=-34.05818669293707, right=-32.39217116700901, top=5.42303953870114)
shape:  (146501, 154470)
pixels: 22,630,009,470
bands:  1
crs:    EPSG:4326 | res: (0.00026949458523585647, 0.00026949458523585647)
dtype:  uint8 | nodata: None
'''
#----------------------------------------------------------------------------------------

#---Work out the window covering the study area---
# The rasters cover all of Brazil, 22.6 billion pixels, which is far too
# large to read into memory. from_bounds converts the grid's bounding box
# into a row and column range, so only that block is read. The rasters are
# in degrees, so the grid is reprojected to match rather than warping the
# raster, which would resample the class values.
grid_wgs = grid.to_crs("EPSG:4326")
left, bottom, right, top = grid_wgs.total_bounds

with rasterio.open(files[0]) as src:
    window = from_bounds(left, bottom, right, top, src.transform)
    # Rounding outwards to whole pixels keeps the window aligned to the
    # raster, so the transform below describes it exactly.
    window = window.round_lengths().round_offsets()
    transform = src.window_transform(window)
    lc_2018 = src.read(1, window=window)

height, width = lc_2018.shape
print("window shape:", lc_2018.shape)
print("pixels:", f"{height * width:,}")

#---Burn each cell ID into the raster grid---
# Same approach as 04_water.py: every pixel gets the ID of the cell it
# falls in, and 0 where no cell covers it. A pixel is assigned to a cell
# when its centre falls inside the cell, so no pixel is shared.
ids = rasterize(
    zip(grid_wgs.geometry, grid_wgs["ID"]),
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="int32",
)

print("cells burned in:", f"{len(np.unique(ids)) - 1:,} of {len(grid):,}")

#---Measure which classes are actually present---
# MapBiomas has around 40 classes nationally, but most do not occur in the
# Pantanal. Measuring their shares first decides which are worth keeping
# and which are too rare to matter, before any grouping is chosen.
inside = ids > 0
counts = np.bincount(lc_2018[inside])
total = counts.sum()

present = pd.DataFrame({
    "code": np.nonzero(counts)[0],
    "pixels": counts[counts > 0],
})
present["share"] = (present["pixels"] / total * 100).round(2)
present = present.sort_values("share", ascending=False)

print("\nclasses present in the Pantanal, 2018")
print(present.to_string(index=False))

'''
classes present in the Pantanal, 2018
 code   pixels  share   class
    4 34659029  19.51   Savanna Formation
   11 32202694  18.12   Wetland
   12 31126622  17.52   Grassland Formation
   33 28626693  16.11   River, Lake and Ocean
    3 25722779  14.48   Forest Formation
   15 19641103  11.05   Pasture
    7  4769235   2.68   Flooded Savanna (beta)
   21   668488   0.38   Mosaic of Uses
   24    71943   0.04   Urban Area
   40    75404   0.04   Rice
   41    36031   0.02   Other Temporary Crops
   29    37674   0.02   Rocky Outcrop
   39    16488   0.01   Soybean
   30     9379   0.01   Mining
    0     4770   0.00   No data
   31      692   0.00   Aquaculture
   25     8310   0.00   Other non Vegetated Areas
   20        2   0.00   Sugar cane
    9     2415   0.00   Forest Plantation
   77        1   0.00   Herbaceous and Shrub Formation

The rasters carry no embedded colour table, so the codes cannot be matched
against the legend colours from the file itself. They are taken from the
published Collection 11 legend table instead.
'''
#----------------------------------------------------------------------------------------

#---Group the classes---
# Seven classes cover 99.5% of the Pantanal. Everything else is under half
# a percent combined, so it goes into one "other" column rather than
# carrying a dozen near-empty predictors into the model.
#
# Codes 6, 49, 50 and 77 do not appear in the 2018 Pantanal but are listed
# here anyway, so that a class appearing in a later year lands in the right
# group instead of falling into "other".
#
# Flooded Savanna (7) is new in Collection 11 and still marked beta. It is
# kept separate rather than folded into savanna, because flooded savanna
# burns differently from dry savanna and at 2.7% it is not negligible. The
# beta status is recorded as a limitation.
GROUPS = {
    "forest":          [3, 6, 49],     # forest formation, floodable forest, wooded sandbank
    "savanna":         [4],            # savanna formation
    "flooded_savanna": [7],            # flooded savanna, beta
    "wetland":         [11],           # campo alagado e area pantanosa
    "grassland":       [12, 77, 50],   # grassland, herbaceous and shrub, herbaceous sandbank
    "pasture":         [15],           # pasture
    "water":           [33, 31],       # river, lake and ocean, aquaculture
}
# Everything not listed above falls into "other": mosaic of uses, urban,
# the crop classes, rocky outcrop, mining, forest plantation, other
# non-vegetated, and code 0 (no data, 4,770 pixels in the whole study area).

#---Build a lookup table from code to group---
# Rather than testing each code one at a time, a 256-long lookup array maps
# every possible uint8 value straight to its group number. Indexing an
# array with the raster is far faster than repeated comparisons.
names = list(GROUPS) + ["other"]
OTHER = len(names) - 1

lut = np.full(256, OTHER, dtype="uint8")
for g, name in enumerate(GROUPS):
    lut[GROUPS[name]] = g

#---Count pixels per cell, per class, per year---
# ids is stored as uint16 to halve the memory it takes, since 6,584 cells
# fit well inside the 65,535 limit. The int32 version is dropped once the
# smaller copy exists, which frees around 1.5 GB.
ids16 = ids.astype("uint16")
del ids, inside, lc_2018

# Every year uses the same grid, so the denominator is counted once.
# Position 0 holds every pixel outside the grid, so counts are read from 1.
total_px = np.bincount(ids16.ravel(), minlength=len(grid) + 1)

rows = []

for f in files:
    year = int(f.stem[-4:])
    with rasterio.open(f) as src:
        arr = src.read(1, window=window)

    grp = lut[arr]  # every pixel becomes a group number

    out = {"ID": grid["ID"].values, "year": year}
    for g, name in enumerate(names):
        # Keep only the cell IDs of pixels in this group, then count how
        # many times each ID appears. Same zonal logic as 04_water.py.
        hit = np.bincount(ids16[grp == g], minlength=len(grid) + 1)
        out[name] = hit[1:] / total_px[1:]

    rows.append(pd.DataFrame(out))
    print(year, "done")

landcover = pd.concat(rows, ignore_index=True)

#---Save---
# Parquet keeps the column types and is much smaller than CSV.
landcover.to_parquet(cfg.PROCESSED / "landcover_year.parquet", index=False)

print("\nrows:", f"{len(landcover):,}")
print("years:", landcover["year"].min(), "to", landcover["year"].max())

#---Check 1: do the shares add up to 1 in every cell?---
# Every pixel belongs to exactly one group, so the shares in a cell must
# sum to 1. If they do not, a code was counted twice or missed entirely.
totals = landcover[names].sum(axis=1)
print("\nshare totals: min", round(totals.min(), 6), "max", round(totals.max(), 6))

#---Check 2: how does land cover change over the period?---
# Pasture should creep up and the natural classes down if conversion is
# happening. A sudden jump in one year would point to a problem with that
# year's file rather than a real change on the ground.
print("\nmean share by year, %")
print((landcover.groupby("year")[names].mean() * 100).round(2).to_string())

#---Check 3: which classes are land cover and which are flood state?---
# Forest and pasture move slowly and in one direction, which is real
# conversion. Wetland, water, grassland and flooded savanna swing by a
# factor of several and swing against each other, because MapBiomas labels
# the same ground as wetland or water in a wet year and as grassland in a
# dry one. Summing those four shows they are stable, confirming it is one
# piece of ground being relabelled rather than land changing hands.
FLOOD_RESPONSIVE = ["wetland", "water", "grassland", "flooded_savanna"]

by_year = landcover.groupby("year")[names].mean() * 100
print("\nflood-responsive classes combined, %")
print(by_year[FLOOD_RESPONSIVE].sum(axis=1).round(2).to_string())
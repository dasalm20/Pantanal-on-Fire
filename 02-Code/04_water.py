#----------------------------------------------------------------------------------------
# This script works out how much of each cell was under water, for every month
# from 2018 to 2025.
#----------------------------------------------------------------------------------------
# Functions used here that are not basic Python, and where they are documented:
#
#   rasterio.merge.merge        joins tiles into one array
#                               rasterio.readthedocs.io/en/stable/api/rasterio.merge.html
#   rasterio.features.rasterize turns polygons into a raster of values
#                               rasterio.readthedocs.io/en/stable/api/rasterio.features.html
#   numpy.ravel                 flattens a 2D array into 1D
#                               numpy.org/doc/stable/reference/generated/numpy.ravel.html
#   numpy.bincount              counts how many times each integer appears
#                               numpy.org/doc/stable/reference/generated/numpy.bincount.html
#   GeoDataFrame.to_crs         reprojects a vector layer
#                               geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_crs.html
#   DataFrame.to_parquet        saves a table in Parquet format
#                               pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html
#
# Zonal counting with rasterize plus bincount is the standard way to summarise a
# raster by polygon without a spatial join. It is the same idea rasterstats uses:
# pythonhosted.org/rasterstats/manual.html
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import pandas as pd  # the table we build
import geopandas as gpd  # the grid
import numpy as np  # counting pixels per cell
import rasterio  # reads the water rasters
from rasterio.merge import merge  # joins the two tiles of each year
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

#---Inpect files from GEE--
# Earth Engine split each yearly export into two tiles. We need to know whether
# they sit side by side or overlap, because overlapping tiles would count some
# pixels twice.
files = sorted(cfg.INPUTS["water"].glob("*.tif"))
print(len(files), "files")

for f in files[:2]:
    with rasterio.open(f) as src:
        print(f.name)
        print("  bounds:", src.bounds)
        print("  shape: ", src.shape)
        print("  bands: ", src.count, "|", src.descriptions[:2], "...")
#----------------------------------------------------------------------------------------
#---Rasterize the grid once---
# The water rasters are in degrees (EPSG:4326), so the grid has to be in
# degrees too before it can be laid over them.
# to_crs reprojects the layer:
# geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_crs.html
grid_wgs = grid.to_crs("EPSG:4326")

# All 16 tiles share the same left and right edge, so one mosaic per year
# always has the same resolution. The first year is opened to read the
# grid shape, transform and size that every year will use.
# merge returns the joined array and the transform that places it on the
# ground. Where tiles overlap it keeps the first value, but these tiles do
# not overlap, so nothing is dropped or double counted:
# rasterio.readthedocs.io/en/stable/api/rasterio.merge.html
year_files = sorted(cfg.INPUTS["water"].glob("*2018*.tif")) #read the first year to get the shape and transform
mosaic, transform = merge([rasterio.open(f) for f in year_files]) #run the merge to get the mosaic and transform
height, width = mosaic.shape[1], mosaic.shape[2] #get the height and width of the mosaic

print("mosaic shape:", mosaic.shape)
print("pixels per band:", f"{height * width:,}")

#---Burn each cell ID into the raster grid---
# rasterize paints every pixel with the ID of the cell it falls in, and 0
# where no cell covers it. ID starts at 1 in the grid, so 0 means outside.
# It takes pairs of (geometry, value), writes into an array of out_shape,
# and uses transform to line that array up with the water rasters:
# rasterio.readthedocs.io/en/stable/api/rasterio.features.html
# A pixel is assigned to a cell when its centre falls inside the cell, so no
# pixel is shared between two cells.
shapes = zip(grid_wgs.geometry, grid_wgs["ID"])
ids = rasterize(
    shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="int32",
)

print("cells burned in:", f"{len(np.unique(ids)) - 1:,} of {len(grid):,}")

#---Count total pixels per cell, once---
# Every year uses the same grid, so the denominator is counted once.
# ravel lays the 2D array out as one long 1D array:
# numpy.org/doc/stable/reference/generated/numpy.ravel.html
# bincount then counts how many times each ID appears in it, returning an
# array where position i holds the count of value i. minlength makes sure
# the result is long enough to hold every cell ID even if a cell is absent:
# numpy.org/doc/stable/reference/generated/numpy.bincount.html
# Position 0 is every pixel outside the grid, so the counts are read from 1
# onwards. This is the whole zonal statistic: one pass over the array
# instead of looping over 6,584 cells.
total_px = np.bincount(ids.ravel(), minlength=len(grid) + 1)

print("smallest cell, pixels:", total_px[1:].min())
print("largest cell, pixels:", f"{total_px[1:].max():,}")

#---Count water pixels per cell, per month---
# Water pixels carry the month number as their value, so anything above 0
# is water. One band is read at a time to keep memory down.
rows = []

for year in range(cfg.LAG_START, cfg.VAL_YEAR + 1):
    year_files = sorted(cfg.INPUTS["water"].glob(f"*{year}*.tif"))

    for month in range(1, 13):
        # Merge the two tiles for this one band only. indexes picks the band,
        # so only one month is in memory rather than all twelve.
        src = [rasterio.open(f) for f in year_files]
        band, _ = merge(src, indexes=[month])
        for s in src:
            s.close()

        # The same bincount, but with weights. Instead of adding 1 for every
        # pixel of a cell, it adds that pixel's weight, which here is True
        # for water and False for everything else. So the result is the
        # number of water pixels in each cell.
        water = (band[0] > 0).ravel()
        water_px = np.bincount(ids.ravel(), weights=water, minlength=len(grid) + 1)

        rows.append(pd.DataFrame({
            "ID": grid["ID"].values,
            "year": year,
            "month": month,
            "water_pct": water_px[1:] / total_px[1:],
        }))

    print(year, "done")

water_month = pd.concat(rows, ignore_index=True)

#---Save---
# Parquet keeps the column types and is much smaller than CSV:
# pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html
water_month.to_parquet(cfg.PROCESSED / "water_month.parquet", index=False)

print("rows:", f"{len(water_month):,}")
print("years:", water_month["year"].min(), "to", water_month["year"].max())
print(water_month["water_pct"].describe())

#----------------------------------------------------------------------------------------
#-Inspection of the results-
#----------------------------------------------------------------------------------------

#---Check 1: does water follow the wet season?---
# MapBiomas Water should peak in the wet season (roughly December to May)
# and bottom out in the dry season. If the months come out flat, the band
# order was read wrong.
# groupby splits the table by a column and applies a function to each group:
# pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
by_month = water_month.groupby("month")["water_pct"].mean()
print("\nmean water share by month")
print((by_month * 100).round(2))

#---Check 2: does water vary between years?---
# 2019 and 2020 were drought years in the Pantanal, so they should sit
# below the wetter years.
by_year = water_month.groupby("year")["water_pct"].mean()
print("\nmean water share by year")
print((by_year * 100).round(2))

#---Check 3: how reliable is the measure in the smallest cells?---
# A cell with very few pixels can only return a handful of values, so its
# water share is coarse. This is recorded, not acted on. The minimum cell
# area is decided in 07_panel.py.
px = pd.DataFrame({"ID": grid["ID"].values, "pixels": total_px[1:]})
print("\npixels per cell")
print(px["pixels"].describe().round(0))
print("cells with under 100 pixels:", int((px["pixels"] < 100).sum()))
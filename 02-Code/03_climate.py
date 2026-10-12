#----------------------------------------------------------------------------------------
# This script is used to get the rainfall and the maximum temperature for every cell,
# for every month from 2018 to 2025.
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import pandas as pd  # the table we build
import geopandas as gpd  # the grid
import numpy as np  # for marking missing values
import rasterio  # reads the climate rasters

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

#---Check the climate rasters to see what projection they are in, how big a pixel is
# what value means "no data", and how many bands each file has--
for name in ("chirps", "tmax"): #Run a loop for each of the two climate datasets.                  
    one = sorted(cfg.INPUTS[name].glob("*.tif"))[0]
    with rasterio.open(one) as src:
        print(name, one.name)
        print("  crs:   ", src.crs)
        print("  pixel: ", src.res)
        print("  nodata:", src.nodata)
        print("  bands: ", src.count, src.dtypes)

#---Put the sampling points in the same projection as the rasters---
# The grid is in metres and the rasters are in degrees. We move the points to
# match the rasters, not the other way round, because reprojecting 192 rasters
# would be slow and would change the pixel values.
points = grid.set_geometry(
    gpd.points_from_xy(grid["rep_x"], grid["rep_y"]), crs=cfg.CRS
).to_crs("EPSG:4326")

coords = list(zip(points.geometry.x, points.geometry.y))
print("sampling points:", len(coords))

#---Read the value under each point, one file at a time---
# Each file holds one month. The year and the month are in the file name, so we
# read them from there. 

# Both file names end the same way, for example:
#   chirps-v3.0.2018.01.tif
#   CHIRTS-ERA5.monthly_Tmax.2018.01.tif
# so splitting on the dots gives the year third from the end and the month second.

#Define a function to sample the climate rasters for each month and year, 
#and return a DataFrame with the results.
def sample_folder(folder, column):
    rows = []
    files = sorted(folder.glob("*.tif"))

    for f in files:
        parts = f.stem.split(".")
        year = int(parts[-2])
        month = int(parts[-1])

        with rasterio.open(f) as src: #using with rasterio to open the file and sample the values at the coordinates
            values = [v[0] for v in src.sample(coords)]

        rows.append(pd.DataFrame({ #append a new DataFrame to the list of rows, with the ID, 
                                    #year, month, and sampled values
            "ID": grid["ID"].values,
            "year": year,
            "month": month,
            column: values,
        }))

    print(f"{column}: read {len(files)} files")
    return pd.concat(rows, ignore_index=True)

#Then call the function for both rainfall and temperature, and store the results in two DataFrames.
precip = sample_folder(cfg.INPUTS["chirps"], "precip")
tmax = sample_folder(cfg.INPUTS["tmax"], "tmax")

#---Join the two together---
climate = precip.merge(tmax, on=["ID", "year", "month"], how="outer")

#---Mark missing values---
# The files themselves do not carry a nodata tag (src.nodata was None), so the
# fill value has to come from the documentation.
#
# CHIRPS: the _FillValue and missing_value attributes are both -9999.0, and the
#   units are millimetres. Source: NOAA ERDDAP metadata for CHIRPS v2.0 P05.
#
# CHIRTS-ERA5: no fill value is documented for this product, and the Earth Engine
#   catalogue entry for CHIRTS gives the units as degrees Celsius with an
#   estimated range of about 10 to 40. Because nothing is documented, the lowest
#   values found are printed below and checked by hand rather than assumed.
bad_precip = climate["precip"] < 0
bad_tmax = climate["tmax"] < -50
print("impossible rainfall values:", bad_precip.sum())
print("impossible temperature values:", bad_tmax.sum())

climate.loc[bad_precip, "precip"] = np.nan
climate.loc[bad_tmax, "tmax"] = np.nan

#--Final checks --
print("rows:", len(climate), "expected:", len(grid) * 8 * 12)
print("missing rainfall:", climate["precip"].isna().sum())
print("missing temperature:", climate["tmax"].isna().sum())
print(climate[["precip", "tmax"]].describe())

#---Does the weather behave the way it should?---
# The Pantanal has a wet summer and a dry winter. Rainfall should be high from
# December to March and close to nothing from June to August. If that pattern
# is not there, something went wrong with the sampling.
print(climate.groupby("month")[["precip", "tmax"]].mean().round(1))

#Save the table to a parquet file for later use in the panel.
climate.to_parquet(cfg.OUTPUTS["climate"], index=False)
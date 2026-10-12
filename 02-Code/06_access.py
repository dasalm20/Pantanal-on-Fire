#----------------------------------------------------------------------------------------
# This script works out how far each cell is from the nearest road and the
# nearest mapped waterway.
#----------------------------------------------------------------------------------------
# Documentation for the functions used here:
#   geopandas.read_file(bbox=)  geopandas.org/en/stable/docs/reference/api/geopandas.read_file.html
#   geopandas.sjoin_nearest     geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.sjoin_nearest.html
#   OSM fclass values           download.geofabrik.de/osm-data-in-gis-formats-free.pdf
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import geopandas as gpd  # the grid, roads and waterways

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

#---Work out the box to read---
# The Centro-Oeste extract covers three states, so only the features near
# the study area are read. The box is padded, because the nearest road to an
# edge cell can sit outside the biome.
PAD_DEG = 1.5  # about 160 km, wider than any distance found

grid_wgs = grid.to_crs("EPSG:4326")  # the OSM files are in degrees
left, bottom, right, top = grid_wgs.total_bounds
bbox = (left - PAD_DEG, bottom - PAD_DEG, right + PAD_DEG, top + PAD_DEG)

print("study area bounds:", [round(v, 2) for v in grid_wgs.total_bounds])
print("box read:         ", [round(v, 2) for v in bbox])

#---Inspect the roads---
# fclass runs from motorway down to footpath. Which types I keep changes the
# distances a lot, so I count what is there before choosing.
roads = gpd.read_file(cfg.INPUTS["roads"], bbox=bbox)

print("\nroads read:", f"{len(roads):,}")
print("crs:", roads.crs)
print("columns:", list(roads.columns))
print("\nroad types present")
print(roads["fclass"].value_counts().to_string())

#---Inspect the waterways---
waterways = gpd.read_file(cfg.INPUTS["waterways"], bbox=bbox)

print("\nwaterways read:", f"{len(waterways):,}")
print("\nwaterway types present")
print(waterways["fclass"].value_counts().to_string())

#---Choose which lines to keep---
# Footpaths say nothing about vehicle access, so they go. Tracks stay, since
# ranch tracks are the main way across most of the Pantanal.
# Main roads and rivers only are built here too, for the sensitivity test in
# 11_experiments.py: residential streets are over half the roads and sit in
# towns, and OSM stream mapping is patchy while rivers are consistent.
NOT_DRIVABLE = ["footway", "path", "cycleway", "steps", "pedestrian"]
MAIN_ROADS = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link",
]

roads_drivable = roads[~roads["fclass"].isin(NOT_DRIVABLE)]
roads_main = roads[roads["fclass"].isin(MAIN_ROADS)]
water_rivers = waterways[waterways["fclass"] == "river"]

print("drivable roads:", f"{len(roads_drivable):,}")
print("main roads:    ", f"{len(roads_main):,}")
print("all waterways: ", f"{len(waterways):,}")
print("rivers only:   ", f"{len(water_rivers):,}")

#---Reproject to metres---
# Distances need a projected CRS. In degrees they would be meaningless and
# would change with latitude.
layers = {
    "dist_road":       roads_drivable,
    "dist_road_main":  roads_main,
    "dist_water":      waterways,
    "dist_river":      water_rivers,
}
layers = {k: v.to_crs(cfg.CRS)[["geometry"]] for k, v in layers.items()}

#---Measure the distance to the nearest line---
# sjoin_nearest uses a spatial index, so it does not test every cell against
# every line. Distance is from the cell polygon, so a cell a road crosses
# gets zero, which is the right answer for that cell.
access = grid[["ID"]].copy()

for name, lines in layers.items():
    joined = gpd.sjoin_nearest(
        grid[["ID", "geometry"]], lines,
        how="left", distance_col="d",
    )
    # A tie returns two rows for one cell, so take the smallest.
    d = joined.groupby("ID")["d"].min() / 1000  # metres to km
    access = access.merge(d.rename(name), on="ID", how="left")
    print(name, "done")

#---Save---
access.to_parquet(cfg.PROCESSED / "access.parquet", index=False)

print("\nrows:", f"{len(access):,}")
print("\ndistances in km")
print(access.drop(columns="ID").describe().round(2).to_string())

#---Check: is the box wide enough?---
# A distance bigger than the pad cannot be trusted, since a nearer line
# could sit just outside what I read.
PAD_KM = PAD_DEG * 111 * 0.95  # a degree is about 111 km, less for longitude here

worst = access.drop(columns="ID").max()
print("\npad:", round(PAD_KM), "km")
for name, value in worst.items():
    flag = "ok" if value < PAD_KM else "TOO NARROW, widen PAD_DEG"
    print(f"{name:<16} max {value:>6.2f} km  {flag}")

'''
roads read: 110,530
waterways read: 7,569
drivable roads: 106,650 | main roads: 14,826
all waterways: 7,569 | rivers only: 1,837

distances in km
       dist_road  dist_road_main  dist_water  dist_river
mean        5.12           16.30       12.15       13.86
50%         2.23           10.58        8.33        9.29
max        38.00           98.84       65.06       68.75

pad: 158 km, all four under it.
Run again with PAD_DEG = 0.5 gave identical numbers to two decimals,
so the padding does not affect the result.
'''
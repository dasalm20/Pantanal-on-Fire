#----------------------------------------------------------------------------------------
#This script is used to build the spatial skeleton of the whole project: 
#one table of 5 km cells covering the Pantanal, 
#where every later script attaches a column.
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import geopandas as gpd  # noqa: I001
from shapely.geometry import box  #Builds each square cell from four coordinates.
import numpy as np
import math
import matplotlib.pyplot as plt
#For the study area map, we need a scale bar, a north arrow, and a legend.
from matplotlib.patches import Rectangle          # for the scale bar
from matplotlib.lines import Line2D               # for the legend keys
from matplotlib.ticker import FuncFormatter       # to show metres as km

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

#---Overview of data ---
#The biome boundary is in degrees.
#thus, we need a fixed set of analysis units with stable IDs, 
#measured in meters, each carrying a point we can sample rasters at.

#---Load the biome boundary ---
biome = gpd.read_file(cfg.INPUTS["ibge"])
#Check the unique biome names and the 
# number of records in the Pantanal biome.
print(biome["NM_BIOMA"].unique())
print(len(biome["NM_BIOMA"].unique()))

#---Filter the biome to the Pantanal ---
pantanal = biome[biome["NM_BIOMA"].str.strip() == "Pantanal"].copy()

#Check CRS of the Pantanal and biome GeoDataFrames to ensure they are consistent.
print(len(pantanal), pantanal.crs)
print(biome.crs)

#---Reproject the Pantanal to the defined CRS ---
pantanal = pantanal.to_crs(cfg.CRS)
#Check CRS again
print(len(pantanal), pantanal.crs)
#Plot the Pantanal to ensure it looks correct.
pantanal.plot()

#---Get the extent ---
minx, miny, maxx, maxy = pantanal.total_bounds #total bounds returns the bounding box of the geometry in the GeoDataFrame.

#---Snap sets the origin of the grid to a multiple of 5000 m, 
# so that the grid does not shift if the boundary file changes.---
S = cfg.GRID_M
minx = math.floor(minx / S) * S
maxx = math.ceil(maxx / S) * S
miny = math.floor(miny / S) * S
maxy = math.ceil(maxy / S) * S

#---Build the grid of 5 km cells covering the Pantanal---
grid_cells = [box(x, y, x+S, y+S)
              for x in range(minx, maxx, S)
              for y in range(miny, maxy, S)]


#---Build a GeoDataFrame from the grid cells and set the CRS to the defined CRS.---
grid = gpd.GeoDataFrame({"geometry": grid_cells}, crs=cfg.CRS)


#
#---Prefilter the grid to only include cells that intersect the Pantanal boundary---
pantanal_union = pantanal.geometry.union_all()  ##unary_union will be deprecated and changed to union_all() in shapely 2.0
grid = grid[grid.intersects(pantanal_union)]
#plot grid 
grid.plot()

#---Cuts edge cells to the boundary---
grid = gpd.clip(grid, pantanal_union)##This will makle edge cells smaller than 25 km².
grid = grid[~grid.geometry.is_empty & grid.geometry.notna()].reset_index(drop=True)

#---Measure and check the area in km2---
grid["area_km2"] = grid.geometry.area / 1e6 #Area in m², divided to km².

print("Number of grid cells:", len(grid))
print("Total area (km²):", grid["area_km2"].sum())

#---Describe the area distribution---
# Clipping the 5 km grid to the biome leaves partial cells of very different sizes.
# A small cell records fewer fires simply because it covers less ground, which would
# look like lower fire risk. So a minimum-area cut-off is needed.
# This block only measures the problem. No cells are dropped here;
# the cut-off is applied in 07_panel.py and varied in the sensitivity analysis.
FULL_KM2 = (cfg.GRID_M ** 2) / 1e6   # 25.0

print(grid["area_km2"].describe()) #this will give us a summary of the area distribution

for frac in (0.04, 0.25, 0.50, 0.75): #loop through the fractions of the full cell area
    t = frac * FULL_KM2
    below = grid["area_km2"] < t
    print(f"below {frac:>4.0%} of a cell ({t:5.2f} km2): "
          f"{below.sum():>5} cells, {grid.loc[below, 'area_km2'].sum():>9.1f} km2")

#---Add a representative point to each cell---

# A point guaranteed inside the polygon.
# A centroid can land outside a clipped L-shape and sample the wrong pixel.
rep_point = grid.geometry.representative_point()
grid["rep_x"] = rep_point.x
grid["rep_y"] = rep_point.y

#---Common ID---
#It is good practice to have a stable key for every join mobving forward. 
#This allows us to join the grid to any other table, and to rejoin the grid to itself after filtering.
grid["ID"] = np.arange(1, len(grid)+1)  


#--Final checks --
print("Cells after clipping:", len(grid))
print("Cells below 1 km²:", (grid["area_km2"] < 1).sum())
print("Total clipped area:", grid["area_km2"].sum())
print("Sample points inside own cell:", grid.geometry.covers(rep_point).all())
print("CRS of saved file:", grid.crs)


#---Save the grid to a GeoPackage---
# Save as GeoPackage, not shapefile: this is because
# GPKG has no 10-character column limit, no five-file sprawl, and has a proper CRS storage.
path = cfg.OUTPUTS["grid"]
grid.to_file(path, layer="grid", driver="GPKG")



#---Study area map---

# Pantanal is taller than it is wide.
fig, ax = plt.subplots(figsize=(8.5, 10))

# Draw the grid first, then the boundary on top .
grid.boundary.plot(ax=ax, linewidth=0.15, color="#9aa5a0", zorder=2) # zorder controls what sits above what.
pantanal.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.4, zorder=3)

# Equal aspect so the map is not stretched.
ax.set_aspect("equal")

#---Coordinates---
# Our CRS is in metres, which gives huge tick numbers. 
km = FuncFormatter(lambda v, p: f"{v/1000:,.0f}") #Divide by 1000 to show km.
ax.xaxis.set_major_formatter(km)
ax.yaxis.set_major_formatter(km)
ax.set_xlabel("Easting (km)")      # correct names for a UTM projection
ax.set_ylabel("Northing (km)")
ax.tick_params(labelsize=8)
ax.grid(True, linewidth=0.3, color="#dde1df", zorder=0)

# Get the map edges, so everything below can be placed relative to them
# instead of with hardcoded coordinates.
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
w = x1 - x0
h = y1 - y0

#---Scale bar---
# A black bar exactly 100 km long, placed near the bottom left.
bx = x0 + w * 0.05
by = y0 + h * 0.04
ax.add_patch(Rectangle((bx, by), 100_000, h * 0.006, facecolor="black", zorder=5))
ax.text(bx + 50_000, by + h * 0.015, "100 km", ha="center", fontsize=8)

#---North arrow---
# An arrow pointing up with an N on top, near the top right.
nx = x0 + w * 0.93
ny = y0 + h * 0.90
ax.annotate("N", xy=(nx, ny), xytext=(nx, ny - h * 0.055),
            arrowprops=dict(facecolor="black", width=2.5, headwidth=9),
            ha="center", fontsize=11, fontweight="bold")

#---Legend---
# Line2D makes a fake line just for the legend key. 
ax.legend(handles=[
    Line2D([], [], color="black", lw=1.4, label="Pantanal boundary"),
    Line2D([], [], color="#9aa5a0", lw=0.6, label=f"{cfg.GRID_M // 1000} km analysis grid"),
], loc="upper left", fontsize=8, frameon=False)

#---Title and caption---
ax.set_title("Study area: the Brazilian Pantanal", fontsize=14, fontweight="bold", loc="left")

# The caption reads the real numbers from the grid. 
fig.text(0.01, 0.01,
         f"{len(grid):,} grid cells · {grid['area_km2'].sum():,.0f} km² · {grid.crs.to_string()}\n"
         "Boundary: IBGE Biomas do Brasil, 1:250,000",
         fontsize=7, color="#555")

#---Locator inset---
# A small second map showing where the Pantanal sits among Brazil's biomes.
# The four numbers are [left, bottom, width, height] as fractions of the figure.
axin = fig.add_axes([0.63, 0.14, 0.24, 0.24])
biome.to_crs(cfg.CRS).plot(ax=axin, facecolor="#e8ece9", edgecolor="white", linewidth=0.4)
pantanal.plot(ax=axin, facecolor="#9c4326", edgecolor="none")
axin.set_axis_off()
axin.set_title("Location in Brazil", fontsize=7)

#---Save---
# 300 dpi because this goes on a printed poster.
plt.savefig(cfg.OUTPUTS["figures"] / "study_area.png", dpi=300, bbox_inches="tight")
#clean all plots
plt.close('all')
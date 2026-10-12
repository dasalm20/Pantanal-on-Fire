#----------------------------------------------------------------------------------------
# This script makes the maps and plots for the poster. It only reads what the
# earlier scripts saved and computes nothing new.
#----------------------------------------------------------------------------------------
# Equivalents in R, since that is where I usually plot:
#   plt.rcParams.update      theme_set()
#   plt.subplots             patchwork, or facet_wrap for panels
#   GeoDataFrame.plot        geom_sf(aes(fill = ...))
#   .boundary.plot           geom_sf(fill = NA)
#   union_all                sf::st_union
#   ax.set_axis_off          theme_void()
#   ax.axhline / axvline     geom_hline / geom_vline
#   ax.barh                  geom_col() + coord_flip()
#   ax.annotate              geom_text()
#   ax.plot_surface          rayshader, or plot3D::persp3D
#   fig.savefig              ggsave()
#
# Documentation:
#   GeoDataFrame.plot  geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.plot.html
#   colour maps        matplotlib.org/stable/users/explain/colors/colormaps.html
#   annotate           matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html
#   plot_surface       matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface.html
#----------------------------------------------------------------------------------------
#---Libraries Part 1---
import pandas as pd  # the result tables
import numpy as np  # the grid behind the 3D surface
import geopandas as gpd  # the grid, for the maps
import matplotlib.pyplot as plt  # the plotting

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

#---Where the figures go, and one look for all of them---
# The palette comes from the config, so these match any other figure in the
# project. Fire runs green to red: green is safe, red is likely to burn.
FIG = cfg.OUTPUTS["figures"]
FIG.mkdir(parents=True, exist_ok=True)

C = cfg.COLOURS

plt.rcParams.update({"font.size": 9})   # theme_set()

#---Load what the earlier scripts saved---
grid = gpd.read_file(cfg.OUTPUTS["grid"], layer="grid")
panel = pd.read_parquet(cfg.PROCESSED / "panel.parquet")
pred = pd.read_parquet(cfg.PROCESSED / "predictions_2025.parquet")

by_year = pd.read_csv(cfg.OUTPUT / "by_year.csv")
results = pd.read_csv(cfg.OUTPUT / "results_2025.csv", index_col=0)
sens = pd.read_csv(cfg.OUTPUT / "sensitivity.csv", index_col=0)
imp = pd.read_csv(cfg.OUTPUT / "permutation_importance.csv", index_col=0).iloc[:, 0]

# Only the cells kept by the area cut-off are mapped.
cells = grid[grid["ID"].isin(panel["ID"])]
print("cells mapped:", f"{len(cells):,}")

# One outline around the whole study area, drawn on every map so the pale
# cells do not run into the white page. union_all merges the cells into a
# single shape; boundary takes just its edge.
outline = gpd.GeoSeries([cells.union_all()], crs=cells.crs).boundary

#----------------------------------------------------------------------------------------

#---Figure 1: fire before, and fire in 2025---
# Left is the burned before baseline, drawn as a map. Right is what the model
# was asked to predict.
past = panel[panel["year"].isin(cfg.TRAIN_YEARS)].groupby("ID")["fire"].mean()
burned_2025 = pred.groupby("ID")["fire"].max()

m1 = cells.merge(past.rename("past"), on="ID")
m1 = m1.merge(burned_2025.rename("now"), on="ID")

fig, ax = plt.subplots(1, 2, figsize=(10, 5.5))   # patchwork

# Colour bar placed under the map rather than beside it, so it does not sit
# in the gap between the two panels.
m1.plot(column="past", cmap=C["fire"], linewidth=0, ax=ax[0],
        legend=True, legend_kwds={"shrink": 0.7, "pad": 0.02,
                                  "orientation": "horizontal",
                                  "label": "share of months burned"})
outline.plot(ax=ax[0], color="#444444", linewidth=0.7)
ax[0].set_title("Fire 2019 to 2024\nshare of August to October months")
ax[0].set_axis_off()

# Grey everywhere, then red on the cells that burned.
m1.plot(color=C["neutral"], linewidth=0, ax=ax[1])
m1[m1["now"] == 1].plot(color=C["mark"], linewidth=0, ax=ax[1])
outline.plot(ax=ax[1], color="#444444", linewidth=0.7)
ax[1].set_title("Fire in 2025\ncells that burned at least once")
ax[1].set_axis_off()

fig.tight_layout()
fig.savefig(FIG / "fig1_fire_maps.png", dpi=300)   # ggsave()
plt.close(fig)

#---Figure 2: what the model predicted for September 2025---
# September is the peak fire month. Cells that really burned are outlined, so
# hits and misses can be read straight off the map.
# rep_x and rep_y are kept here because figure 7 needs them.
sept = pred[pred["month"] == 9]
m2 = cells[["ID", "rep_x", "rep_y", "geometry"]].merge(
    sept[["ID", "pred_rf", "fire"]], on="ID")

fig, ax = plt.subplots(figsize=(6, 6))

m2.plot(column="pred_rf", cmap=C["fire"], linewidth=0, ax=ax,
        legend=True, legend_kwds={"shrink": 0.6, "label": "predicted probability"})

# boundary draws the outline only, so the fill below stays visible.
# Thick enough to be seen at poster size.
m2[m2["fire"] == 1].boundary.plot(ax=ax, color="black", linewidth=0.9)
outline.plot(ax=ax, color="#444444", linewidth=0.7)

ax.set_title("Predicted fire risk, September 2025\noutlined cells are where fire occurred")
ax.set_axis_off()
fig.tight_layout()
fig.savefig(FIG / "fig2_prediction_map.png", dpi=300)
plt.close(fig)

#---Figure 3: the main finding---
# For each year: rank every cell, take the top 10%, and see what share of that
# year's real fires fell inside it. Two bars per year, the model and the
# burned before baseline, with the years ordered from quietest to busiest.
b = by_year.sort_values("fire rate")

# 2025 is marked because it is the held-out year. The rest come from leaving
# one training year out at a time.
ticks = []
for _, r in b.iterrows():
    year = int(r["year"])
    mark = "\n(held out)" if year == cfg.VAL_YEAR else ""
    ticks.append(f"{year}\n{r['fire rate'] * 100:.1f}%{mark}")

x = np.arange(len(b))

fig, ax = plt.subplots(figsize=(8, 4.2))

ax.bar(x - 0.2, b["top10 before"], width=0.4, color=C["mark"],
       label="burned before")
ax.bar(x + 0.2, b["top10 forest"], width=0.4, color=C["accent"],
       label="random forest")

ax.set_xticks(x)
ax.set_xticklabels(ticks)
ax.set_xlabel("year, ordered from quietest to busiest, with the share of cell-months that burned")
ax.set_ylabel("share of that year's fires\ncaught in the top 10% of cells")
ax.set_title("The model only beats the baseline in active years")
ax.legend(title="", frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig3_finding.png", dpi=300)
plt.close(fig)

#---Figure 4: the four models on 2025---
# Share of the real 2025 fires that fall inside the top 1, 5 and 10% of cells.
# Three fixed colours rather than a colour map, because the pale end of a ramp
# disappears on white paper. The legend goes outside, where it cannot sit on
# top of a bar.
order = ["burned before", "random forest", "logistic", "burned last month"]
cols = ["top 1%", "top 5%", "top 10%"]

fig, ax = plt.subplots(figsize=(7.5, 4))

# geom_col(position = "dodge") + coord_flip()
results.loc[order, cols].plot.barh(ax=ax, width=0.75, color=C["ramp"])

ax.set_ylabel("")
ax.set_xlabel("share of 2025 fires captured")
ax.set_title("2025: how many real fires fall in the top ranked cells")
ax.legend(title="", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig4_baselines.png", dpi=300)
plt.close(fig)

#---Figure 5: does any data choice change the answer?---
# Every bar sits left of zero, so burned before wins under all ten setups.
s = sens.sort_values("top10 gap")

fig, ax = plt.subplots(figsize=(6, 4))

ax.barh(s.index, s["top10 gap"], color=C["accent"])   # geom_col() + coord_flip()
ax.axvline(0, color="black", linewidth=0.8)           # geom_vline
ax.set_xlabel("forest minus burned before, top 10% capture on 2025")
ax.set_title("No data choice changes the result")
fig.tight_layout()
fig.savefig(FIG / "fig5_sensitivity.png", dpi=300)
plt.close(fig)

#---Figure 6: which predictors carried weight on 2025---
# Sorted small to large, because barh draws from the bottom up.
top = imp.head(10).sort_values()

fig, ax = plt.subplots(figsize=(6, 4))

ax.barh(top.index, top.values, color=C["accent"])
ax.set_xlabel("drop in average precision when shuffled")
ax.set_title("What the model actually used in 2025")
fig.tight_layout()
fig.savefig(FIG / "fig6_importance.png", dpi=300)
plt.close(fig)

#---Figure 7: the fire history as a surface---
# The grid is a regular fishnet, so each cell goes back into a row and column
# by dividing its coordinates by the cell size. Height is how often the cell
# burned between 2019 and 2024, so the ridges are the persistent hotspots that
# the burned before baseline relies on. Cells outside the biome stay empty and
# leave holes in the surface.
S = cfg.GRID_M

col = ((m2["rep_x"] - m2["rep_x"].min()) / S).round().astype(int)
row = ((m2["rep_y"] - m2["rep_y"].min()) / S).round().astype(int)

# how="left" keeps the order of m2, so the values line up with row and col.
history = m2[["ID"]].merge(past.rename("past"), on="ID", how="left")

z = np.full((row.max() + 1, col.max() + 1), np.nan)
z[row, col] = history["past"].values

# Axes in kilometres, so the shape stays at true scale.
x = np.arange(z.shape[1]) * S / 1000
y = np.arange(z.shape[0]) * S / 1000
x, y = np.meshgrid(x, y)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(projection="3d")

ax.plot_surface(x, y, z, cmap=C["fire"], linewidth=0, antialiased=True,
                rstride=1, cstride=1)

ax.view_init(elev=50, azim=-65)   # looking down from the south west
ax.set_zlim(0, float(np.nanmax(z)))
ax.set_xlabel("km east")
ax.set_ylabel("km north")
ax.set_zlabel("share of months burned")
ax.set_title("Where fire keeps coming back, 2019 to 2024")

fig.tight_layout()
fig.savefig(FIG / "fig7_fire_history_3d.png", dpi=300)
plt.close(fig)

print("saved seven figures to", FIG)
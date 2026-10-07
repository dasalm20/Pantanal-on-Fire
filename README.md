# Pantanal-on-Fire
### GEOG 654
### Daylan Salmeron

## Planned workflow

This is the plan I am working to. Scripts 00, 01 and 02 are built. The rest will change as I go.

```text
03-Data/  (raw, not tracked)
│
├── IBGE Biomes ───────────→ study boundary
├── NASA FIRMS VIIRS ──────→ fires
├── CHIRPS v3 ─────────────→ rainfall
├── CHIRTS-ERA5 ───────────→ maximum temperature
├── MapBiomas Collection 11 → land cover
├── MapBiomas Water ───────→ flooding
└── OpenStreetMap ─────────→ roads and rivers
                │
                ▼
02-Code/  one script per step, each one picking up where the last left off

  00_config.py      paths, map projection, settings, checks the data is there
  01_grid.py        5 km grid cut to the Pantanal          → grid.gpkg
  02_fire.py        fires per cell per month               → fire_month.parquet
  03_climate.py     rainfall and temperature per cell      → climate_month.parquet
  04_water.py       how much of each cell is flooded       → water_month.parquet
  05_landcover.py   what each cell is covered in           → landcover_year.parquet
  06_access.py      distance to the nearest road and river → access.parquet
  07_panel.py       join everything, add the lags          → panel.parquet
  08_folds.py       split the data for testing             → folds.parquet
  09_train.py       Random Forest, 2019-2024               → model.joblib
  10_evaluate.py    how well it did, and how sure we are
  11_experiments.py what happens if the choices were different
  12_figures.py     maps and plots
                │
                ▼
04-Processed/  (made by the scripts, not tracked)
                │
                ▼
05-Output/  figures, tables, reports
```

### How the workflow is set up

- Each script only reads what the one before it produced. After 06, nothing goes back to the raw data.
- The lagged variables are all built in `07_panel.py`, not inside the model.
- The smallest cell size we keep is decided in `07_panel.py`, not in the grid itself. That way it can be changed without rebuilding everything.
- Every script prints how many rows it made and what years it covers, so a bad join shows up straight away.

### What one row means

```text
one row = one grid cell, one year, one month
```

The months being predicted are August, September and October. The model learns from 2019 to 2024, and 2025 is held back to test it.

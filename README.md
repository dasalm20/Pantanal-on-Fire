# Pantanal-on-Fire
### GEOG 654
### Daylan Salmeron 

## Planned workflow

Proposed pipeline. Scripts 00 and 01 are built; the rest is a plan and will change as the project develops.

```text
03-Data/  (raw, not tracked)
│
├── IBGE Biomes ───────────→ study boundary
├── NASA FIRMS VIIRS ──────→ fire response
├── CHIRPS v3 ─────────────→ rainfall
├── CHIRTS-ERA5 ───────────→ maximum temperature
├── MapBiomas Collection 11 → land cover
├── MapBiomas Water ───────→ inundation
└── OpenStreetMap ─────────→ roads and waterways
                │
                ▼
02-Code/  one script per stage, each reading only the stage before it

  00_config.py      paths, CRS, constants, input check
  01_grid.py        5 km grid clipped to the Pantanal      → grid.gpkg
  02_fire.py        VIIRS detections per cell per month    → fire_month.parquet
  03_climate.py     CHIRPS and Tmax sampled to cells       → climate_month.parquet
  04_water.py       monthly surface water per cell         → water_month.parquet
  05_landcover.py   annual class shares per cell           → landcover_year.parquet
  06_access.py      distance to roads and waterways        → access.parquet
  07_panel.py       join all, build lags, apply area rule  → panel.parquet
  08_folds.py       spatial blocks and leave-one-year-out  → folds.parquet
  09_train.py       Random Forest, 2019-2024               → model.joblib
  10_evaluate.py    baselines, metrics, intervals, calibration
  11_experiments.py ablation and sensitivity
  12_figures.py     maps and plots
                │
                ▼
04-Processed/  (generated, not tracked)
                │
                ▼
05-Output/  figures, tables, reports
```

### Rules

- Each script reads only the previous stage's output. Nothing after 06 touches raw data.
- All lagged predictors are built in `07_panel.py`, never inside the model script.
- The minimum cell-area cut-off is applied in `07_panel.py`, not baked into the grid, so it can be varied without rebuilding anything upstream.
- Every script prints its row count and date range so a bad join is caught where it happens.

### Unit of analysis

```text
one row = grid cell × year × target month
```

Target months are August, September and October. Training runs on 2019-2024 and 2025 is held out.
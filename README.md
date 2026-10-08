# Pantanal-on-Fire
### GEOG 654: GIS and Spatial Modeling, Fall 2026
### Daylan Salmeron

## AI disclaimer

AI models were used in this project for debugging, documentation and grammar
checking. All code was written by the author, as was the research question,
the study design and the choice of data.

Where a suggestion from an AI model led to a change in the analysis, that
decision is recorded in the project notebook in `01-Doc/`, alongside the
reasoning and the evidence behind it.

## What this project does

The Brazilian Pantanal is the largest tropical wetland in the world. In 2020 about 29% of it burned, and an estimated 17 million vertebrates were killed directly by those fires. The biome covers roughly 150,000 km², so fire response has to be prioritised somewhere, and that means knowing where fire is most likely before a month starts.

This project builds and tests a model that predicts, for each 5 km cell in the Pantanal, whether fire will occur there in a given month of the dry season.

**The question:** can conditions known before a month begins (how dry the preceding months were, how hot, how much of the cell was flooded, what the land is covered in, how accessible it is) rank Pantanal locations by their chance of burning in the month ahead?

**The data:** 6,584 grid cells, August to October, 2019 to 2025. Satellite fire detections, monthly rainfall and temperature, monthly flooding, annual land cover, and roads and waterways. All public, all collected.

**The test:** a Random Forest learns from 2019 to 2024, and 2025 is held back entirely. The model is also compared against two simple alternatives: how often each cell has burned before, and whether it burned last month.

## Progress

| Script | Does | Status |
|---|---|---|
| `00_config.py` | Paths, projection, settings, input checks | Done |
| `01_grid.py` | 5 km grid cut to the Pantanal | Done, 6,584 cells, 150,880 km² |
| `02_fire.py` | Fires per cell per month | Done, 553,056 rows |
| `03_climate.py` | Rainfall and temperature per cell | Done, 632,064 rows |
| `04_water.py` | Flooded share of each cell | Next |
| `05_landcover.py` | Land cover share of each cell | |
| `06_access.py` | Distance to roads and rivers | |
| `07_panel.py` | Join everything, build the lags | |
| `08_folds.py` | Split the data for testing | |
| `09_train.py` | Random Forest | |
| `10_evaluate.py` | Metrics, baselines, uncertainty | |
| `11_experiments.py` | Sensitivity tests | |
| `12_figures.py` | Maps and plots | |

Each stage is written and committed on its own branch, and the decisions behind each one are recorded in `01-Doc/`.

## Something found along the way

Cutting a regular grid to the biome boundary leaves cells of very different sizes, from 0.002 km² to a full 25 km². Checking the fire data against cell size showed that the largest cells record fire twelve times more often than the smallest. Dividing by area cuts that gap to 2.4 times, so most of it is simply the amount of ground a cell covers rather than how fire-prone it is.

This supports applying a minimum cell size later in the pipeline rather than building it into the grid, and varying it from 0 to 100% of a full cell in the sensitivity analysis.

## Planned workflow
=======
## AI disclaimer
>>>>>>> 9b25b268fe706401fde6c4b8d9c98e0f602cd101

AI models were used in this project for debugging, documentation, grammar
checking, and for grammar review. All code was written by the author, and 
all analytical decisions, from the choice of question to
the model design and the handling of edge cases, are the author's own.

Where a suggestion from an AI model led to a change in the analysis, that
decision is recorded in the project notebook in `01-Doc/`, alongside the
reasoning and the evidence behind it.

## What this project does

The Brazilian Pantanal is the largest tropical wetland in the world. In 2020 about 29% of it burned, and an estimated 17 million vertebrates were killed directly by those fires. The biome covers roughly 150,000 km², so fire response has to be prioritised somewhere, and that means knowing where fire is most likely before a month starts.

This project builds and tests a model that predicts, for each 5 km cell in the Pantanal, whether fire will occur there in a given month of the dry season.

**The question:** can conditions known before a month begins (how dry the preceding months were, how hot, how much of the cell was flooded, what the land is covered in, how accessible it is) rank Pantanal locations by their chance of burning in the month ahead?

**The data:** 6,584 grid cells, August to October, 2019 to 2025. Satellite fire detections, monthly rainfall and temperature, monthly flooding, annual land cover, and roads and waterways. All public, all collected.

**The test:** a Random Forest learns from 2019 to 2024, and 2025 is held back entirely. The model is also compared against two simple alternatives: how often each cell has burned before, and whether it burned last month.

## Progress

| Script | Does | Status |
|---|---|---|
| `00_config.py` | Paths, projection, settings, input checks | Done |
| `01_grid.py` | 5 km grid cut to the Pantanal | Done, 6,584 cells, 150,880 km² |
| `02_fire.py` | Fires per cell per month | Done, 553,056 rows |
| `03_climate.py` | Rainfall and temperature per cell | Done, 632,064 rows |
| `04_water.py` | Flooded share of each cell | Next |
| `05_landcover.py` | Land cover share of each cell | |
| `06_access.py` | Distance to roads and rivers | |
| `07_panel.py` | Join everything, build the lags | |
| `08_folds.py` | Split the data for testing | |
| `09_train.py` | Random Forest | |
| `10_evaluate.py` | Metrics, baselines, uncertainty | |
| `11_experiments.py` | Sensitivity tests | |
| `12_figures.py` | Maps and plots | |

Each stage is written and committed on its own branch, and the decisions behind each one are recorded in `01-Doc/`.

## Something found along the way

Cutting a regular grid to the biome boundary leaves cells of very different sizes, from 0.002 km² to a full 25 km². Checking the fire data against cell size showed that the largest cells record fire twelve times more often than the smallest. Dividing by area cuts that gap to 2.4 times, so most of it is simply the amount of ground a cell covers rather than how fire-prone it is.

That is why a minimum cell size is applied later in the pipeline rather than built into the grid, and why it is varied from 0 to 100% of a full cell in the sensitivity analysis.

## Planned workflow

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
- The smallest cell size kept is decided in `07_panel.py`, not in the grid itself, so it can be changed without rebuilding everything.
- Every script prints how many rows it made and what years it covers, so a bad join shows up straight away.

### What one row means

```text
one row = one grid cell, one year, one month
```

The months being predicted are August, September and October. The model learns from 2019 to 2024, and 2025 is held back to test it.

## Repository

```text
01-Doc/        proposal, project notebook, decisions
02-Code/       the pipeline, one script per stage
03-Data/       raw data (not tracked)
04-Processed/  intermediate files (not tracked)
05-Output/     figures and tables
```

Raw and processed data are excluded from version control. `environment.yml` rebuilds the environment with `conda env create -f environment.yml`.

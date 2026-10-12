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

## How to read the commit history

Before building this pipeline I ran a preliminary version locally to work out
whether the data was usable and roughly how the project should be structured.
That earlier work is not in this repository. What is here is the rebuild: each
stage written again in order, with the decisions and checks recorded as they were
made.

So the commits move quickly, and some land within minutes of each other. The
structure was already mapped out, so the writing was fast. The repository is not a
record of typing. It is a record of reasoning: what each script decides, what was
measured before deciding it, what the checks returned, and what turned up along
the way.

The decisions are still being made here. Working through the pipeline a second
time, more carefully, surfaced things the preliminary run had missed. A few
examples:

- The MapBiomas water rasters encode the month number in the pixel value rather
  than a yes or no flag, which changes how the water share is calculated.
- Four of the eight land cover classes swing by up to eight-fold between years and
  swing against each other, because MapBiomas labels the same ground as wetland or
  water in a wet year and as grassland in a dry one. They are partly a flood
  measure, not land use, which changed how they are described and added a
  sensitivity test.
- The first version of the access script used a search box too narrow to guarantee
  the nearest road had been found, which was caught by a check and fixed.

So this repository does two things at once: it follows a process that was planned
in advance, and it reviews that process closely enough to change it.

## What this project does

The Brazilian Pantanal is the largest tropical wetland in the world. In 2020 about 29% of it burned, and an estimated 17 million vertebrates were killed directly by those fires. The biome covers roughly 150,000 km², so fire response has to be prioritised somewhere, and that means knowing where fire is most likely before a month starts.

This project builds and tests a model that predicts, for each 5 km cell in the Pantanal, whether fire will occur there in a given month of the dry season.

**The question:** can conditions known before a month begins (how dry the preceding months were, how hot, how much of the cell was flooded, what the land is covered in, how accessible it is) rank Pantanal locations by their chance of burning in the month ahead? The approach is a data-driven statistical susceptibility model, not a fire-spread simulation.

**The data:** 6,584 grid cells, August to October, 2019 to 2025. Satellite fire detections, monthly rainfall and temperature, monthly flooding, annual land cover, and roads and waterways. All public, all collected.

**The test:** a Random Forest classifier learns from 2019 to 2024, with hyperparameters
tuned by grid search inside the spatial cross-validation, and 2025 is held back entirely.
The model is compared against three baselines: how often each cell has burned before,
whether it burned last month, and a logistic regression on the same predictors.

## Progress

| Script | Does | Status |
|---|---|---|
| `00_config.py` | Paths, projection, settings, input checks | Done |
| `01_grid.py` | 5 km grid cut to the Pantanal | Done, 6,584 cells, 150,880 km² |
| `02_fire.py` | Fires per cell per month | Done, 553,056 rows |
| `03_climate.py` | Rainfall and temperature per cell | Done, 632,064 rows |
| `04_water.py` | Flooded share of each cell | DONE |
| `05_landcover.py` | Land cover share of each cell | Next|
| `06_access.py` | Distance to roads and rivers | |
| `07_panel.py` | Join everything, build the lags | |
| `08_folds.py` | Split the data for testing | |
| `09_train.py` | Random Forest classifier, tuned by grid search | |
| `10_evaluate.py` | Metrics, baselines, logistic regression, uncertainty | |
| `11_experiments.py` | Sensitivity tests | |
| `12_figures.py` | Maps and plots | |

Each stage is written and committed on its own branch, and the decisions behind each one are recorded in `01-Doc/`.

## Something found along the way

Cutting a regular grid to the biome boundary leaves cells of very different sizes, from 0.002 km² to a full 25 km². Checking the fire data against cell size showed that the largest cells record fire twelve times more often than the smallest. Dividing by area cuts that gap to 2.4 times, so most of it is simply the amount of ground a cell covers rather than how fire-prone it is.

This supports applying a minimum cell size later in the pipeline rather than building it into the grid, and varying it from 0 to 100% of a full cell in the sensitivity analysis.

## Feedback from the proposal

The proposal was reviewed in October. Three changes followed:

- Hyperparameter tuning by grid search was added to `09_train.py`, nested inside the
  spatial cross-validation so that tuning never sees the held-out blocks.
- Logistic regression was added as a third baseline, alongside historical fire frequency
  and previous-month fire.
- The problem statement and objectives were tightened to name the modelling approach
  and the specific comparisons being made.

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

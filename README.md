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

All twelve stages are built and committed.

| Script | Does | Result |
|---|---|---|
| `00_config.py` | Paths, projection, settings, palette | |
| `01_grid.py` | 5 km grid cut to the Pantanal | 6,584 cells, 150,880 km² |
| `02_fire.py` | Fires per cell per month | 553,056 rows |
| `03_climate.py` | Rainfall and temperature per cell | 632,064 rows |
| `04_water.py` | Flooded share of each cell | 632,064 rows |
| `05_landcover.py` | Land cover share of each cell | 52,672 rows, 8 classes |
| `06_access.py` | Distance to roads and rivers | 6,584 rows, 4 distances |
| `07_panel.py` | Join everything, build the lags, apply the area rule | 126,882 rows, 27 columns |
| `08_folds.py` | Spatial blocks and leave one year out | 6,042 cells, blocks at 25, 50, 75 km |
| `09_train.py` | Random Forest, tuned by grid search | 0.438 average precision, 4.0x random |
| `10_evaluate.py` | 2025 test, three baselines, uncertainty | see below |
| `11_experiments.py` | Ten sensitivity variants | result holds in all ten |
| `12_figures.py` | Maps and plots | seven figures |

Each stage is written and committed on its own branch, and the decisions behind each
one are recorded in `01-Doc/`.

## Results

**In cross-validation the model works.** Average precision is 0.438 against a fire
rate of 0.111, so four times better than ranking cells at random, and the five
spatial folds agree closely (0.391 to 0.479).

**On the held-out year it does not beat the simplest baseline.** 2025 had a fire
rate of 1.42%, the quietest year in the record. Ranking cells by how often they
burned in 2019 to 2024 caught 40.7% of 2025's fires in the top 10% of cells. The
Random Forest caught 29.8%. A paired bootstrap puts that gap at -0.100, range
-0.170 to -0.034, so it is a real difference and not noise. The forest did beat
the logistic regression and the burned-last-month baseline on every measure.

**Leave one year out explains why.** The forest beats the baseline in all four
years with a fire rate above 10%, and loses in all three quiet years. 2025 is the
quietest of all.

| Year | Share of cell-months burned | Burned before | Random Forest |
|---|---|---|---|
| 2025 (held out) | 1.4% | **0.407** | 0.298 |
| 2022 | 1.9% | **0.317** | 0.280 |
| 2023 | 3.7% | **0.328** | 0.315 |
| 2021 | 11.2% | 0.245 | **0.270** |
| 2024 | 11.3% | 0.233 | **0.253** |
| 2019 | 17.1% | 0.195 | **0.262** |
| 2020 | 21.4% | 0.186 | **0.228** |

*Share of that year's fires caught in the top 10% of ranked cells.*

So the answer to the research question is conditional. Conditions known before the
month starts do carry information about where fire will occur, but only in years
when a lot of fire occurs. In a quiet year the few fires that do happen are the
places that always burn, which is exactly what the historical baseline already
knows, so the weather has nothing left to add.

Ten sensitivity variants were run, each changing one data choice. The result holds
in all ten, with the gap running from -0.076 to -0.116.

## Things found along the way

**Cell size confounds fire detection.** Cutting a regular grid to the biome
boundary leaves cells from 0.002 km² to a full 25 km². The largest cells record
fire twelve times more often than the smallest. Dividing by area cuts that to 2.4
times, so most of it is simply the amount of ground a cell covers. The minimum cell
size is therefore applied in `07_panel.py` rather than built into the grid, and
varied in the sensitivity analysis. It turned out to matter less than expected:
cross-validation moves only from 0.437 to 0.440 across cut-offs of 50, 75 and 100%
of a full cell.

**Four land cover classes are really a flood measure.** Wetland, water, grassland
and flooded savanna swing by up to eight-fold between years and swing against one
another, because MapBiomas labels the same ground as wetland or water in a wet year
and as grassland in a dry one. Their sum is stable at 47 to 53%, which confirms it
is relabelling rather than land changing hands. They are described accordingly, and
a combined version is tested in the sensitivity analysis.

**Block size changes the score, not the answer.** Cross-validation gives 0.469 with
25 km blocks, 0.438 with 50 km and 0.412 with 75 km. Smaller blocks put training
and test cells closer together and flatter the model. The 2025 result is unaffected,
since the final model trains on all the training years either way.

**MapBiomas Water encodes the month number in the pixel value**, not a yes or no
flag, which changes how the water share is calculated.

## Feedback from the proposal

The proposal was reviewed in October. Three changes followed:

- Hyperparameter tuning by grid search was added to `09_train.py`, nested inside the
  spatial cross-validation so that tuning never sees the held-out blocks.
- Logistic regression was added as a third baseline, alongside historical fire frequency
  and previous-month fire.
- The problem statement and objectives were tightened to name the modelling approach
  and the specific comparisons being made.

## workflow

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

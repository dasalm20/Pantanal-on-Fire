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

All twelve pipeline stages are built and committed, plus three additional
testing scripts and a master script that runs everything in order.

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
| `09_train.py` | Random Forest, tuned by grid search | 0.4658 average precision, 4.0x random |
| `10_evaluate.py` | 2025 test, three baselines, uncertainty | see below |
| `11_experiments.py` | Ten sensitivity variants | see below |
| `12_figures.py` | Maps and plots | seven figures |

`02-Code/additional_testing/` holds three further scripts. `a1_panel_all.py`
and `a2_folds_all.py` rebuild the panel and the folds with no minimum cell
area, so `a3_area_sweep.py` can test cut-offs looser than the 50% baseline.

`02-Code/000_master.py` runs all fifteen scripts in order, each as its own
process so memory is released between stages, and writes a log per stage to
`05-Output/logs/`. A full run takes about 72 minutes.

Each stage is written and committed on its own branch, and the decisions behind each
one are recorded in `01-Doc/`.

## Results

**In cross-validation the model works.** Average precision is 0.4658 against a
fire rate of 0.115, so four times better than ranking cells at random, and the
five spatial folds agree closely (0.450 to 0.500).

**On the held-out year the answer depends on how the ranking is measured.**
2025 had a fire rate of 1.57%, the quietest year in the record. Across the
whole ranking the Random Forest is ahead of the burned-before baseline:
average precision 0.1100 against 0.0698, a gap of +0.0402 with a paired
bootstrap range of +0.0097 to +0.0794, so the difference is real and not
noise. On the top 10% of cells, the shortlist a fire manager would actually
work from, the two cannot be separated: the forest catches 39.8% of 2025's
fires and the baseline 41.9%, a gap of -0.0211 with a range of -0.0739 to
+0.0409. The forest beat the logistic regression and the burned-last-month
baseline on every measure.

**Leave one year out shows where the shortlist advantage comes from.** On the
top 10%, the forest wins in every year with a fire rate above 10% and loses in
every year below it, with no exceptions.

| Year | Share of cell-months burned | Burned before | Random Forest |
|---|---|---|---|
| 2025 (held out) | 1.6% | **0.419** | 0.398 |
| 2022 | 2.1% | **0.325** | 0.317 |
| 2023 | 3.9% | **0.339** | 0.306 |
| 2021 | 11.6% | 0.235 | **0.292** |
| 2024 | 11.7% | 0.235 | **0.263** |
| 2019 | 17.6% | 0.199 | **0.233** |
| 2020 | 22.1% | 0.183 | **0.236** |

*Share of that year's fires caught in the top 10% of ranked cells.*

Measured by average precision instead, the forest wins six of the seven years,
losing only 2023.

So the answer to the research question is conditional. Conditions known before
the month starts do carry information about where fire will occur, and that
information is enough to rank the whole biome better than fire history alone
in almost every year. It is only enough to improve the top 10% shortlist in
years when a lot of fire occurs. In a quiet year the few fires that happen are
the places that always burn, which is what the historical baseline already
knows, so the weather has little left to add at the top of the ranking.

Fifteen sensitivity configurations were run, ten in `11_experiments.py` and
five minimum cell areas in `a3_area_sweep.py`. The average precision gap is
positive in all fifteen, from +0.0306 to +0.0441, so the ranking advantage
does not rest on any one data choice. The top 10% gap stays near zero
throughout, from -0.0695 to +0.0023, and is negative in fourteen of the
fifteen.

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

**Block size changes the score, not the answer.** Cross-validation gives 0.5028
with 25 km blocks, 0.4658 with 50 km and 0.4161 with 75 km. Smaller blocks put
training and test cells closer together and flatter the model. The 2025 result
is identical across all three, since the final model trains on all the training
years either way.

**MapBiomas Water encodes the month number in the pixel value**, not a yes or no
It turned out to matter less than expected: cross-validation moves only from
0.4658 to 0.4704 across cut-offs of 50, 75 and 100% of a full cell, and the
full sweep from 0 to 100% in `a3_area_sweep.py` moves only from 0.4411 to
0.4542.

**Tied scores had to be broken at random.** The burned-before baseline is an
average over 18 cell-months, so it takes only 19 distinct values, and the
burned-last-month baseline is either 0 or 1. The 10% cut-off therefore falls
inside a large group of cells with identical scores, and sorting alone settles
those by row order, which made the result depend on how the table happened to
be sorted. Averaging over 20 random tie-breaks removed that. It strengthened
both baselines and narrowed the model's margin: the top 10% gap moved from
-0.0140 to -0.0211.

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

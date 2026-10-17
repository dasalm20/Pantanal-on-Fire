#----------------------------------------------------------------------------------------
# Runs the whole project in one go: the baseline, then each sensitivity run
# that needs the pipeline rebuilt from raw data. Each script runs as its own
# process, so the memory it used is released before the next one starts, and
# each run writes to its own folders so nothing is overwritten.
#----------------------------------------------------------------------------------------
# Usage:
#   conda activate pantanal
#   python 02-Code/000_master.py
#
# Expect about two and a half hours.
#----------------------------------------------------------------------------------------
# Equivalents in R:
#   subprocess.Popen     system2()
#   sys.executable       R.home()
#   Path(__file__)       here::here()
#   time.perf_counter    Sys.time()
#
# Documentation:
#   subprocess.Popen  docs.python.org/3/library/subprocess.html#subprocess.Popen
#   sys.executable    docs.python.org/3/library/sys.html#sys.executable
#   matplotlib Agg    matplotlib.org/stable/users/explain/figure/backends.html
#----------------------------------------------------------------------------------------
#---Libraries---
import os  # the environment handed to each child process
import sys  # the interpreter path, so the child uses the same environment
import subprocess  # runs each script as a separate process
import time  # how long each stage took
from datetime import datetime
from pathlib import Path

#----------------------------------------------------------------------------------------

#---Where everything sits---
# This file sits in 02-Code/, so the repo root is one level up. Taken from the
# file's own location, so it runs on any machine without editing a path.
ROOT = Path(__file__).resolve().parents[1]

# One stamp for the whole session, so the three runs group together in the logs.
STAMP = datetime.now().strftime("%Y%m%d_%H%M")

#---The pipeline, in order---
# 01 to 12 is the pipeline. a1 to a3 come after, because a1 reads what 01 to
# 06 saved and a3 reads the model saved by 09.
PIPELINE = [f"02-Code/{s}.py" for s in (
    "01_grid", "02_fire", "03_climate", "04_water", "05_landcover",
    "06_access", "07_panel", "08_folds", "09_train", "10_evaluate",
    "11_experiments", "12_figures")]

EXTRA = [f"02-Code/additional_testing/{s}.py" for s in (
    "a1_panel_all", "a2_folds_all", "a3_area_sweep")]

# The sensitivity runs stop at 11, which is where the comparison numbers are
# produced. 12 only draws the poster figures and a1 to a3 only sweep the area
# cut-off, and neither is needed for a variant.
UP_TO_11 = PIPELINE[:-1]

#---The runs---
# The baseline, then the two tests the proposal promised that cannot be done
# on a slice of the finished panel. Both need the pipeline rebuilt from raw
# data, so each gets its own folders and its own full run.
RUNS = [
    {"tag": "",             "settings": {},                       "scripts": PIPELINE + EXTRA},
    {"tag": "grid10km",     "settings": {"GRID_M": "10000"},      "scripts": UP_TO_11},
    {"tag": "conf_nominal", "settings": {"CONFIDENCE": "nominal"},"scripts": UP_TO_11},
]

#---Check every script is there before starting---
# A missing file two hours in is worse than a missing file now.
missing = sorted({s for r in RUNS for s in r["scripts"]
                  if not (ROOT / s).exists()})
if missing:
    print("missing:", ", ".join(missing))
    sys.exit(1)


#---Run one script and write its log---
# Popen returns while the script is still running, so its output can be read
# line by line and printed as it arrives rather than at the end. -u turns off
# buffering in the child so those lines actually arrive as they are produced.
# stderr is folded into stdout, so warnings and tracebacks reach the log too.
def run_script(script, env, logs):
    name = Path(script).stem
    log = logs / f"{STAMP}_{name}.log"
    start = time.perf_counter()

    print(f"\n=== {name}  started {datetime.now():%H:%M:%S} ===", flush=True)

    with open(log, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / script)],
            cwd=ROOT,                      # the scripts expect the repo root
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:           # screen and file at once
            print(line, end="", flush=True)
            f.write(line)
        code = proc.wait()                 # the process ends here, memory released

    secs = time.perf_counter() - start
    print(f"=== {name}  finished {datetime.now():%H:%M:%S},"
          f" took {int(secs // 60)}m {int(secs % 60):02d}s ===", flush=True)

    return code, log


#----------------------------------------------------------------------------------------
#---Run them---
session_start = time.perf_counter()
summary = []

for run in RUNS:
    tag = run["tag"]
    label = tag if tag else "baseline"

    logs = ROOT / (f"05-Output_{tag}" if tag else "05-Output") / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    # Positron sets MPLBACKEND to its own backend so plots appear in the pane.
    # A child process cannot import that module, and a batch run has no window
    # to draw into anyway, so the children are given Agg, which writes to file
    # and never opens a window. The run settings ride along the same way, so
    # 00_config.py picks them up without any script being edited.
    env = {**os.environ, "MPLBACKEND": "Agg"}
    for key in ("RUN_TAG", "GRID_M", "CONFIDENCE"):
        env.pop(key, None)                 # clear anything left in the session
    if tag:
        env["RUN_TAG"] = tag
    env.update(run["settings"])

    print(f"\n{'=' * 70}")
    print(f"RUN: {label}  |  {len(run['scripts'])} scripts"
          f"  |  settings {run['settings'] or 'none'}")
    print(f"{'=' * 70}", flush=True)

    start = time.perf_counter()
    failed = None

    for script in run["scripts"]:
        code, log = run_script(script, env, logs)
        if code != 0:
            failed = Path(script).stem
            print(f"\nrun {label} stopped: {failed} failed, see {log.name}")
            break                          # the rest would build on a missing file

    secs = time.perf_counter() - start
    summary.append({
        "run": label,
        "state": f"FAILED at {failed}" if failed else "ok",
        "took": f"{int(secs // 60)}m {int(secs % 60):02d}s",
        "logs": str(logs.relative_to(ROOT)),
    })
    # A failure in one run does not stop the others, since each writes to its
    # own folders and the finished ones are already saved.

#---Done---
total = time.perf_counter() - session_start

print(f"\n{'=' * 70}")
print(f"ALL RUNS, {int(total // 60)}m {int(total % 60):02d}s, stamp {STAMP}")
print(f"{'=' * 70}")
for s in summary:
    print(f"  {s['run']:<14} {s['state']:<22} {s['took']:>9}  {s['logs']}")

if any(s["state"] != "ok" for s in summary):
    sys.exit(1)
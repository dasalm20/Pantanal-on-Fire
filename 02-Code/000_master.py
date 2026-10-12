#----------------------------------------------------------------------------------------
# Runs the whole project in order, the main pipeline then the additional
# testing scripts. Each script runs as its own process, so the memory it used
# is released before the next one starts. Every script writes its own log.
#----------------------------------------------------------------------------------------
# Usage:
#   conda activate pantanal
#   python 02-Code/000_master.py
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
LOGS = ROOT / "05-Output" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

STAMP = datetime.now().strftime("%Y%m%d_%H%M")

#---The environment each script runs in---
# Positron sets MPLBACKEND to its own backend so plots appear in the pane. A
# child process cannot import that module, and a batch run has no window to
# draw into anyway, so the children are given Agg, which writes to file and
# never opens a window.
ENV = {**os.environ, "MPLBACKEND": "Agg"}

#---The order---
# 01 to 12 is the pipeline. a1 to a3 come after, because a1 reads what 01 to
# 06 saved and a3 reads the model saved by 09.
SCRIPTS = [
    "02-Code/01_grid.py",
    "02-Code/02_fire.py",
    "02-Code/03_climate.py",
    "02-Code/04_water.py",
    "02-Code/05_landcover.py",
    "02-Code/06_access.py",
    "02-Code/07_panel.py",
    "02-Code/08_folds.py",
    "02-Code/09_train.py",
    "02-Code/10_evaluate.py",
    "02-Code/11_experiments.py",
    "02-Code/12_figures.py",
    "02-Code/additional_testing/a1_panel_all.py",
    "02-Code/additional_testing/a2_folds_all.py",
    "02-Code/additional_testing/a3_area_sweep.py",
]

#---Check they are all there before starting---
# A missing file forty minutes in is worse than a missing file now.
missing = [s for s in SCRIPTS if not (ROOT / s).exists()]
if missing:
    print("missing:", ", ".join(missing))
    sys.exit(1)

#---Run them---
print("run", STAMP)
run_start = time.perf_counter()

for script in SCRIPTS:
    name = Path(script).stem
    log = LOGS / f"{STAMP}_{name}.log"
    start = time.perf_counter()

    print(f"\n=== {name}  started {datetime.now():%H:%M:%S} ===", flush=True)

    # Popen returns while the script is still running, so its output can be
    # read line by line and printed as it arrives rather than at the end.
    # -u turns off buffering in the child so those lines actually arrive as
    # they are produced. stderr is folded into stdout, so warnings and
    # tracebacks reach the log too.
    with open(log, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / script)],
            cwd=ROOT,                      # the scripts expect the repo root
            env=ENV,                       # Agg, not the editor's own backend
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

    # Stop at the first failure. Carrying on would build a later stage on a
    # file that was never written.
    if code != 0:
        print(f"\nstopped: {name} failed, see {log.name}")
        sys.exit(1)

#---Done---
total = time.perf_counter() - run_start
print(f"\nall done in {int(total // 60)}m {int(total % 60):02d}s")
print(f"logs in 05-Output/logs/ with stamp {STAMP}")
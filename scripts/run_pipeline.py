"""
Re-runs the whole data + ML pipeline from the raw public dataset:
  1. ml/build_dataset.py        -> data/processed/agrismart_sensor_dataset.csv
  2. ml/train_models.py         -> ml/models/*, experiments/exp0..exp2
  3. ml/experiments.py          -> experiments/exp3..exp6 + EXPERIMENTS_SUMMARY.md
  4. database/generate_seed.py  -> database/04_seed_data.sql (crop envelopes from the new model)
Then reload the database with  python scripts/setup_database.py

Usage:  python scripts/run_pipeline.py           (about 10-15 minutes on a laptop)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [("ml", "build_dataset.py"), ("ml", "train_models.py"), ("ml", "experiments.py"),
         ("database", "generate_seed.py")]

for folder, script in STEPS:
    t = time.time()
    print(f"\n===== {folder}/{script}", flush=True)
    r = subprocess.run([sys.executable, "-W", "ignore", script], cwd=ROOT / folder)
    if r.returncode:
        sys.exit(f"{script} failed")
    print(f"===== {script} finished in {time.time() - t:.0f}s", flush=True)
print("\nPipeline complete. Now run:  python scripts/setup_database.py")

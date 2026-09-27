"""
Script to execute headless Locust load tests and generate benchmark results (Deliverable 06)
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_locust_benchmark(host: str = "http://localhost:3000", users: int = 100, spawn_rate: int = 10, run_time: str = "30s"):
    csv_prefix = RESULTS_DIR / "load"
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(BASE_DIR / "locustfile.py"),
        "--host",
        host,
        "--users",
        str(users),
        "--spawn-rate",
        str(spawn_rate),
        "--run-time",
        run_time,
        "--headless",
        f"--csv={csv_prefix}",
    ]

    print(f"Running command: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Locust load test executed successfully.")
        print(res.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Locust execution warning/error: {e}")
        print("Stderr:", e.stderr)

    # Analyze CSV output if created
    stats_csv = RESULTS_DIR / "load_stats.csv"
    if stats_csv.exists():
        df = pd.read_csv(stats_csv)
        print("=== Locust Load Test Results Summary ===")
        print(df[["Type", "Name", "Request Count", "Failure Count", "50%", "95%", "99%", "Average Response Time"]])


if __name__ == "__main__":
    host_arg = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
    run_locust_benchmark(host=host_arg)

"""
Benchmark Harness CLI Runner (Module 4)
Runs end-to-end benchmark across all optimized model variants and updates reports/optimization_journey.md
"""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prodml.benchmark import run_full_benchmark_journey


def main():
    print("=== Starting Model Benchmark & Optimization Harness (Module 4) ===")
    df = run_full_benchmark_journey(iterations=250)
    print("\n=== Benchmark Results Journey Table ===")
    display_cols = [
        "optimization_stage",
        "size_mb",
        "size_reduction_pct",
        "p50_latency_ms",
        "p95_latency_ms",
        "speedup_factor",
        "accuracy",
        "f1_score",
    ]
    print(df[display_cols].to_string(index=False))
    print("\nFull markdown report updated at: reports/optimization_journey.md")


if __name__ == "__main__":
    main()

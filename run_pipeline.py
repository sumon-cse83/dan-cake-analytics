"""One command runs the whole framework, from raw CSV to dashboard data and report tables.

    python run_pipeline.py                 # everything (about 20-25 minutes on a laptop)
    python run_pipeline.py --steps eda forecast insights export   # only some steps

Steps run in this order:
    data      ingest, validate, clean, build SQLite warehouse
    eda       exploratory analysis, statistical tests, figures
    forecast  demand forecasting (7 ML models + baselines + Holt-Winters/SARIMA) and 12-month outlook
    classify  large-order classification (7 models)
    cluster   product and market segmentation
    anomaly   daily and transaction anomaly detection
    insights  automatic insights and recommendations
    export    dashboard data, LaTeX numbers, website data
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dancake import anomaly, classification, clustering, data, eda, export, forecasting, insights  # noqa: E402

ORDER = ["data", "eda", "forecast", "classify", "cluster", "anomaly", "insights", "export"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="*", default=ORDER, choices=ORDER)
    steps = ap.parse_args().steps
    t0 = time.time()
    d = data.run() if "data" in steps else data.load_clean()
    jobs = {"eda": lambda: eda.run(d), "forecast": lambda: forecasting.run(d), "classify": lambda: classification.run(d),
            "cluster": lambda: clustering.run(d), "anomaly": lambda: anomaly.run(d), "insights": insights.run,
            "export": lambda: export.run(d)}
    for s in ORDER[1:]:
        if s in steps:
            t = time.time()
            jobs[s]()
            print(f"    -> {s} finished in {time.time() - t:.0f}s")
    print(f"Pipeline finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

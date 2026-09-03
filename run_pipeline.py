#!/usr/bin/env python
"""
Sentinel Payment Fraud Risk Engine - Pipeline Runner

Training and Evaluation Entry Point for Sentinel:
- Executes the complete end-to-end ML pipeline
- Trains Logistic Regression and Random Forest models
- Evaluates on chronological train/val/test splits
- Optimizes decision thresholds on validation data
- Evaluates on held-out test data
- Saves production pipeline artifacts to data/processed/model/

Usage:
    python run_pipeline.py
    python run_pipeline.py --fp-cost 10.0 --fn-cost 100.0
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path to ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipeline import FraudDetectionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Sentinel Fraud Detection ML Training Pipeline."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to raw dataset directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Path to output directory for models and logs (default: data/processed)",
    )
    parser.add_argument(
        "--fp-cost",
        type=float,
        default=10.0,
        help="Assumed cost per false positive ($) [project assumption] (default: 10.0)",
    )
    parser.add_argument(
        "--fn-cost",
        type=float,
        default=100.0,
        help="Assumed cost per false negative ($) [project assumption] (default: 100.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("STARTING SENTINEL FRAUD DETECTION PIPELINE")
    print(f"  Data Directory:   {args.data_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  FP Cost Assumption: ${args.fp_cost:.2f} (customer friction / support)")
    print(f"  FN Cost Assumption: ${args.fn_cost:.2f} (chargeback loss)")
    print("=" * 75)

    pipeline = FraudDetectionPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
    )

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    pipeline.run_complete_pipeline()

    print("\n[OK] Pipeline execution successfully completed.")
    print("[OK] Model artifacts saved to: data/processed/model/")
    print("[OK] Review pipeline_summary.json for full metric breakdown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Main experiment runner for TrafCopAgent paper results.

Runs all 4 scenarios x 10 seeds = 40 experiments to produce Table 1 results.

Usage:
    # Run full main experiment (all scenarios, 10 seeds each)
    python run_main_experiments.py

    # Run specific scenario only
    python run_main_experiments.py --scenarios grid4x4

    # Custom seed count
    python run_main_experiments.py --n_seeds 5
"""

import subprocess
import sys
import os
from pathlib import Path


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = ["grid4x4", "avenue4x4", "cologne8", "fenglin"]
SEEDS = list(range(1, 11))  # 10 seeds per paper


def run_training(scenario, seed):
    """Train TrafCopAgent for one scenario-seed pair."""
    cmd = [
        sys.executable, str(BASE_DIR / "trafcop_agent/train_trafcop.py"),
        "--scenario", scenario,
        "--seed", str(seed),
    ]
    print(f"[{scenario}] Training with seed={seed}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    return result.returncode == 0


def run_evaluation(scenario, seed):
    """Evaluate trained model for one scenario-seed pair."""
    cmd = [
        sys.executable, str(BASE_DIR / "trafcop_agent/evaluate_trafcop.py"),
        "--scenario", scenario,
        "--seed", str(seed),
        "--num_eval_episodes", "32",
    ]
    print(f"[{scenario}] Evaluating with seed={seed}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    return result.returncode == 0


def run_all_main_experiments(scenarios=None, n_seeds=None):
    """Run full main experiment protocol."""
    scenarios = scenarios or SCENARIOS
    seeds = SEEDS[:n_seeds] if n_seeds else SEEDS

    results = {}

    for scenario in scenarios:
        print(f"\n{'#'*60}")
        print(f"# Running TrafCopAgent on {scenario.upper()} ({len(seeds)} seeds)")
        print(f"{'#'*60}\n")

        t_emv_list = []
        t_rev_list = []

        for seed in seeds:
            # Train (or load pretrained model)
            model_dir = BASE_DIR / "trafcop_agent" / "results" / scenario / f"seed_{seed}"
            if not model_dir.exists():
                success = run_training(scenario, seed)
                if not success:
                    print(f"  Training failed for {scenario} seed={seed}, skipping")
                    continue

            # Evaluate
            success = run_evaluation(scenario, seed)
            if success:
                # Load results from evaluation output
                result_file = model_dir / "eval_results.json"
                import json
                if result_file.exists():
                    with open(result_file) as f:
                        res = json.load(f)
                        t_emv_list.append(res.get("t_emv_mean", 0))
                        t_rev_list.append(res.get("t_rev_mean", 0))

        # Compute summary statistics for this scenario
        if t_emv_list:
            import numpy as np
            results[scenario] = {
                "t_emv_mean": float(np.mean(t_emv_list)),
                "t_emv_std": float(np.std(t_emv_list)),
                "t_rev_mean": float(np.mean(t_rev_list)),
                "t_rev_std": float(np.std(t_rev_list)),
            }

    # Save final results
    import json
    results_path = BASE_DIR / "main_experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal Results:")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to: {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs='+', default=SCENARIOS,
                        help="Scenarios to run")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of seeds per scenario")
    args = parser.parse_args()
    run_all_main_experiments(args.scenarios, args.n_seeds)

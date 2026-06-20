"""
Evaluation utilities for TrafCopAgent.

Computes T_EMV (average EMV travel time) and T_REV (average REV travel time)
matching the paper's evaluation metrics (Section "Evaluation Metrics").

    T_REV = (1/N_REV) * sum_i(t_exit^i - t_entry^i)
    T_EMV = (1/N_EMV) * sum_i(t_exit^i - t_entry^i)
"""

import numpy as np


def compute_travel_times(vehicle_data, emv_data):
    """
    Compute average travel times for REV and EMV.

    Args:
        vehicle_data: list of dicts with keys 'entry_time', 'exit_time', 'vehicle_type'
                      where vehicle_type is 'REV' or 'EMV'
        emv_data: list of dicts with keys 'entry_time', 'exit_time' for each EMV episode

    Returns:
        t_rev: float - average REV travel time
        t_emv: float - average EMV travel time
    """
    rev_times = [
        d["exit_time"] - d["entry_time"]
        for d in vehicle_data
        if d.get("vehicle_type", "REV") == "REV"
        and d.get("exit_time", 0) > 0 and d.get("entry_time", float('inf')) < float('inf')
    ]
    emv_times = [
        d["exit_time"] - d["entry_time"]
        for d in emv_data
        if d.get("exit_time", 0) > 0 and d.get("entry_time", float('inf')) < float('inf')
    ]

    t_rev = np.mean(rev_times) if rev_times else 0.0
    t_emv = np.mean(emv_times) if emv_times else 0.0
    return t_rev, t_emv


def compute_summary_stats(t_rev_list, t_emv_list):
    """
    Compute mean and standard deviation across multiple seeds/runs.

    Args:
        t_rev_list: list of per-run T_REV values
        t_emv_list: list of per-run T_EMV values

    Returns:
        stats: dict with mean and std for both metrics
    """
    return {
        "t_rev_mean": float(np.mean(t_rev_list)),
        "t_rev_std": float(np.std(t_rev_list)),
        "t_emv_mean": float(np.mean(t_emv_list)),
        "t_emv_std": float(np.std(t_emv_list)),
        "n_runs": len(t_rev_list),
    }


def format_latex_table(stats_by_scenario):
    """
    Format results as LaTeX table rows matching paper Table 1 style.

    Args:
        stats_by_scenario: dict mapping scenario name to stats dict from compute_summary_stats
    """
    lines = []
    header = "Method & "
    for i, metric in enumerate(["T_EMV", "T_REV"]):
        for scenario in stats_by_scenario:
            if i == 0:
                header += f"{scenario} & "
    lines.append(header.strip())

    # TrafCopAgent row
    t_rev_vals = []
    t_emv_vals = []
    for scenario, stats in stats_by_scenario.items():
        val_e = f"{stats['t_emv_mean']:.1f}\\pm{stats['t_emv_std']:.1f}"
        val_r = f"{stats['t_rev_mean']:.1f}\\pm{stats['t_rev_std']:.1f}"
        t_emv_vals.append(val_e)
        t_rev_vals.append(val_r)

    row = "TrafCopAgent & "
    for v in t_emv_vals + t_rev_vals:
        row += f"{v} & "
    lines.append(row.strip())
    return "\n".join(lines)

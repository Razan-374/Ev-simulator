"""Before/after comparison of Stage 1 results for matched scenario labels.

"Before" = the uncorrected 2026-11-13 run (stage1_run_20261113_160635_Apeak), which
used hospitalpower_kw, mixed internal/external sessions, arrival-sorted fleets,
randomised external load, and A_peak for both models.
"After" = a corrected run produced by run_stage1.py + merge_stage1.py.

The EV populations differ between the two (that is one of the corrections), so the
comparison is by scenario label (model, power, seed, blocked outlets, base profile, N).

Usage: py -3.13 compare_stage1_before_after.py <after_run_dir> [--seeds 0 1 ...]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import stage1_common as c

BEFORE_DIR = c.ROOT / "stage1_corrected_outputs" / "stage1_run_20261113_160635_Apeak"
KEYS = ["model", "power_kw", "seed", "blocked_outlets", "base_profile", "N_internal"]


def load_before():
    frames = []
    for path in sorted((BEFORE_DIR / "V1G").glob("hosting_capacity_results_power*kW_v1g_stage1.csv")):
        frame = pd.read_csv(path)
        frame["model"] = "V1G"
        frame["site_peak_reduction_kw"] = frame["baseline_peak_kw"] - frame["optimized_peak_kw"]
        frames.append(frame)
    for path in sorted((BEFORE_DIR / "V2G_parallel").glob("*/hosting_capacity_results_power*kW_stage1_corrected.csv")):
        frame = pd.read_csv(path)
        frame["model"] = "V2G"
        frames.append(frame)
    before = pd.concat(frames, ignore_index=True).rename(columns={"power_limit_kw": "power_kw"})
    before["service_feasible"] = before["service_feasible"].astype(str).str.lower().eq("true")
    return before.drop_duplicates(KEYS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("after_run_dir")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-values", nargs="+", type=int, default=[0, 10, 25, 50, 100])
    args = parser.parse_args()
    after = pd.read_csv(Path(args.after_run_dir) / "merged" / "results_all.csv")
    after["service_feasible"] = after["service_feasible"].astype(str).str.lower().eq("true")
    before = load_before()
    b = before[KEYS + ["baseline_peak_kw", "optimized_peak_kw", "site_peak_reduction_kw", "service_feasible",
                       "solve_class", "objective_mode", "avoidable_unmet_kwh"]].rename(columns={
        "baseline_peak_kw": "before_reference_peak_kw", "optimized_peak_kw": "before_facility_peak_kw",
        "site_peak_reduction_kw": "before_peak_reduction_kw", "service_feasible": "before_service_feasible",
        "solve_class": "before_solve_class", "objective_mode": "before_objective",
        "avoidable_unmet_kwh": "before_avoidable_unmet_kwh"})
    a = after[KEYS + ["fixed_peak_kw", "facility_peak_kw", "peak_reduction_vs_fixed_kw", "service_feasible",
                      "solve_class", "objective_mode", "avoidable_unmet_kwh", "cap_blocked_shortfall_kwh"]].rename(columns={
        "fixed_peak_kw": "after_reference_peak_kw", "facility_peak_kw": "after_facility_peak_kw",
        "peak_reduction_vs_fixed_kw": "after_peak_reduction_kw", "service_feasible": "after_service_feasible",
        "solve_class": "after_solve_class", "objective_mode": "after_objective",
        "avoidable_unmet_kwh": "after_avoidable_unmet_kwh", "cap_blocked_shortfall_kwh": "after_cap_blocked_kwh"})
    table = a.merge(b, on=KEYS, how="left")
    if args.seeds is not None:
        table = table[table["seed"].isin(args.seeds)]
    table = table[table["N_internal"].isin(args.n_values)].sort_values(KEYS)
    out = Path(args.after_run_dir) / "merged"
    table.to_csv(out / "before_after_matched_scenarios.csv", index=False)

    def capacity(frame, prefix):
        rows = []
        for keys, group in frame.groupby(["model", "power_kw", "seed", "blocked_outlets", "base_profile"]):
            group = group.sort_values("N_internal")
            ns, ok = group["N_internal"].to_numpy(), group["service_feasible"].to_numpy()
            rows.append(dict(zip(["model", "power_kw", "seed", "blocked_outlets", "base_profile"], keys),
                             **{f"{prefix}_hc": c.contiguous_capacity(ns, ok),
                                f"{prefix}_non_monotonic": " ".join(map(str, c.non_monotonic_ns(ns, ok)))}))
        return pd.DataFrame(rows)

    hc = capacity(after, "after").merge(capacity(before, "before"),
                                        on=["model", "power_kw", "seed", "blocked_outlets", "base_profile"], how="left")
    hc.to_csv(out / "before_after_hosting_capacity_by_seed.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table.to_string(index=False))
        print(hc.to_string(index=False))


if __name__ == "__main__":
    main()

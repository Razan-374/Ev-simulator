"""Merge corrected Stage 1 case outputs and build summaries.

Usage:
    py -3.13 merge_stage1.py <run_dir> [--expected-seeds 0 1 ...]

Writes to <run_dir>/merged/:
  results_all.csv                 every (model, power, seed, blocked, profile, N) row
  hosting_capacity_by_seed.csv    contiguous / any-pass capacity and non-monotonic N per case
  hosting_capacity_probabilistic.csv  HC_90/95/100 (contiguous) with any-pass and non-monotonic N
  solver_status_summary.csv       outcome counts, gaps, runtimes and validation failures
  matched_v1g_v2g.csv             V1G vs V2G at identical fleet/load/outlet/N
  seed_matrix.csv                 completed vs expected cases
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import stage1_common as c

CASE_KEYS = ["model", "power_kw", "seed", "blocked_outlets", "base_profile"]
GROUP_KEYS = ["model", "power_kw", "blocked_outlets", "available_internal_outlets", "base_profile", "objective_mode"]


def load_results(run_dir):
    files = sorted(p for p in Path(run_dir).glob("*_power*kW/*.csv"))
    if not files:
        raise SystemExit(f"No case CSV files under {run_dir}")
    frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    dup = frame.duplicated(CASE_KEYS + ["N_internal"])
    if dup.any():
        raise SystemExit(f"Duplicate result rows: {frame.loc[dup, CASE_KEYS + ['N_internal']].head()}")
    for column in ("service_feasible", "optimality_proven", "usable_solution"):
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    return frame


def per_seed_capacity(frame):
    rows = []
    for keys, group in frame.groupby(CASE_KEYS + ["objective_mode"], dropna=False):
        group = group.sort_values("N_internal")
        ns = group["N_internal"].to_numpy()
        ok = group["service_feasible"].to_numpy()
        proven = group["optimality_proven"].to_numpy()
        row = dict(zip(CASE_KEYS + ["objective_mode"], keys))
        row.update({
            "hosting_capacity_ev": c.contiguous_capacity(ns, ok),
            "hosting_capacity_any_pass": int(ns[ok].max()) if ok.any() else -1,
            "non_monotonic_pass_N": " ".join(map(str, c.non_monotonic_ns(ns, ok))),
            "hosting_capacity_proven_optimal": c.contiguous_capacity(ns, proven),
            "n_tested": len(ns), "max_n_tested": int(ns.max()),
            "time_limited_cases": int((group["solve_class"] == "time_limit_feasible").sum()),
            "infeasible_cases": int(group["solve_class"].isin(["infeasible", "time_limit_no_solution"]).sum()),
            "fixed_peak_kw": float(group["fixed_peak_kw"].dropna().iloc[0]) if group["fixed_peak_kw"].notna().any() else np.nan,
            "max_peak_reduction_vs_fixed_kw": float(group.loc[group["service_feasible"], "peak_reduction_vs_fixed_kw"].max()),
        })
        at_hc = group[group["N_internal"] == row["hosting_capacity_ev"]]
        row["peak_reduction_at_hc_kw"] = float(at_hc["peak_reduction_vs_fixed_kw"].iloc[0]) if len(at_hc) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def solver_summary(frame):
    solved = frame[frame["N_internal"] > 0]
    summary = (solved.groupby(["model", "power_kw", "base_profile", "solve_class"], dropna=False)
               .agg(cases=("N_internal", "size"), service_feasible=("service_feasible", "sum"),
                    validation_failed=("validation_passed", lambda s: int((s.astype(str).str.lower() == "false").sum())),
                    gap_abs_max=("mip_gap_abs", "max"), gap_rel_max=("mip_gap_rel", "max"),
                    runtime_mean_sec=("runtime_sec", "mean"), runtime_max_sec=("runtime_sec", "max"))
               .reset_index())
    return summary


def matched(frame):
    keys = ["power_kw", "seed", "blocked_outlets", "base_profile", "N_internal"]
    cols = keys + ["fleet_hash", "fixed_peak_kw", "facility_peak_kw", "peak_reduction_vs_fixed_kw", "service_feasible",
                   "solve_class", "avoidable_unmet_kwh", "ev_discharge_energy_kwh", "ev_net_export_peak_kw",
                   "v2v_energy_kwh", "mip_gap_abs", "objective_mode"]
    v1 = frame[frame["model"] == "V1G"][cols]
    v2 = frame[frame["model"] == "V2G"][cols]
    both = v1.merge(v2, on=keys, suffixes=("_v1g", "_v2g"))
    both["same_fleet"] = both["fleet_hash_v1g"] == both["fleet_hash_v2g"]
    both["same_fixed_load"] = np.isclose(both["fixed_peak_kw_v1g"], both["fixed_peak_kw_v2g"], equal_nan=True)
    both["v2g_minus_v1g_peak_kw"] = both["facility_peak_kw_v2g"] - both["facility_peak_kw_v1g"]
    return both


def seed_matrix(frame, expected_seeds, n_blocked=8, n_profiles=2):
    rows = []
    for (model, power), group in frame.groupby(["model", "power_kw"]):
        cases = group.drop_duplicates(CASE_KEYS)
        for seed in expected_seeds:
            done = cases[cases["seed"] == seed]
            rows.append({"model": model, "power_kw": power, "seed": seed, "cases_completed": len(done),
                         "cases_expected": n_blocked * n_profiles,
                         "complete": len(done) == n_blocked * n_profiles})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--expected-blocked", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.run_dir) / "merged"
    out.mkdir(parents=True, exist_ok=True)
    frame = load_results(args.run_dir)
    frame.to_csv(out / "results_all.csv", index=False)
    per_seed = per_seed_capacity(frame)
    per_seed.to_csv(out / "hosting_capacity_by_seed.csv", index=False)
    prob, summary = c.probabilistic_capacity(frame, GROUP_KEYS)
    peak = (frame[frame["service_feasible"]].groupby(GROUP_KEYS + ["N_internal"])["peak_reduction_vs_fixed_kw"]
            .agg(["mean", "std", "min", "max"]).add_prefix("peak_reduction_kw_").reset_index())
    prob.merge(peak, on=GROUP_KEYS + ["N_internal"], how="left").to_csv(out / "pass_probability_by_N.csv", index=False)
    summary.to_csv(out / "hosting_capacity_probabilistic.csv", index=False)
    solver_summary(frame).to_csv(out / "solver_status_summary.csv", index=False)
    pairs = matched(frame)
    pairs.to_csv(out / "matched_v1g_v2g.csv", index=False)
    matrix = seed_matrix(frame, args.expected_seeds, args.expected_blocked)
    matrix.to_csv(out / "seed_matrix.csv", index=False)
    print(f"rows={len(frame)}  validation_failed={(frame['validation_passed'].astype(str).str.lower() == 'false').sum()}")
    print(f"matched pairs={len(pairs)}  same_fleet={pairs['same_fleet'].all() if len(pairs) else 'n/a'}  "
          f"same_fixed_load={pairs['same_fixed_load'].all() if len(pairs) else 'n/a'}")
    print(solver_summary(frame).to_string(index=False))
    print(summary.to_string(index=False))
    print(matrix.groupby(["model", "power_kw"])["complete"].sum().to_string())


if __name__ == "__main__":
    main()

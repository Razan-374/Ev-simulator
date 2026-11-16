"""Investigate V2G optimality gaps: re-solve one case under several solver settings.

Usage: py -3.13 investigate_stage1_gaps.py --power-kw 22 --seed 0 --blocked 0 --profile worst_peak --n 50
           --time-limit 600 --preprocess on --out stage1_validated_outputs/gap_investigation/results.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import stage1_common as c
from stage1_v2g_model import solve_v2g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--power-kw", type=int, default=22)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blocked", type=int, default=0)
    p.add_argument("--profile", default="average")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--time-limit", type=float, default=120)
    p.add_argument("--preprocess", choices=("on", "off"), default="off")
    p.add_argument("--cycling-penalty", type=float, default=1e-3)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    profile = c.load_fixed_load_profiles("weekday")[a.profile]
    fleet, _, meta = c.ensure_fleet(c.fleet_config(a.seed, "weekday"))
    arr = c.fleet_arrays(fleet, a.n)
    avail = np.full(c.SLOTS_PER_DAY, c.INTERNAL_OUTLETS - a.blocked)
    cfg = dict(c.DEFAULT_SOLVER, time_limit_sec=a.time_limit, options=["preprocess off"] if a.preprocess == "off" else [])
    log = Path(a.out).with_name(f"cbc_p{a.power_kw}_b{a.blocked}_{a.profile}_N{a.n}_t{int(a.time_limit)}_pre{a.preprocess}.log")
    outcome, arrays = solve_v2g(arr, profile, avail, a.power_kw, cfg, cycling_penalty=a.cycling_penalty,
                                keep_log_path=log)
    row = {k: a.__dict__[k] for k in ("power_kw", "seed", "blocked", "profile", "n", "time_limit", "preprocess",
                                      "cycling_penalty")}
    row.update({k: outcome[k] for k in ("solve_class", "objective_value", "best_bound", "mip_gap_abs", "runtime_sec")})
    if arrays is not None:
        m, _ = c.evaluate_solution(arrays["charge_kw"], arrays["discharge_kw"], arr, profile, avail, a.power_kw,
                                   unmet_kwh_model=arrays["unmet_kwh"], total_load_model=arrays["total_load_kw"],
                                   allow_discharge=True)
        row.update({k: m[k] for k in ("facility_peak_kw", "peak_reduction_vs_fixed_kw", "ev_discharge_energy_kwh",
                                      "validation_passed")})
        row["tie_break_term"] = a.cycling_penalty * m["ev_discharge_energy_kwh"]
    row["finished"] = time.strftime("%H:%M:%S")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
    print(row)


if __name__ == "__main__":
    main()

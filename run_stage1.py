"""Run corrected Stage 1 hosting-capacity sweeps for one model (V1G or V2G).

Example:
    py -3.13 run_stage1.py --model v2g --power-kw 11 --seeds 0 1 --output-dir stage1_validated/run1

Each (seed, blocked outlets, base profile) case writes a CSV of per-N rows and an NPZ
with complete 15-minute power and SoC profiles. Finished cases are skipped on restart
when their stored configuration hash matches, so long runs can be resumed.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import stage1_common as c
from stage1_v1g_model import solve_v1g, OBJECTIVES as V1G_OBJECTIVES
from stage1_v2g_model import solve_v2g, OBJECTIVES as V2G_OBJECTIVES

FORMULATION_VERSION = "stage1-corrected-v1.2-soc-recursion-outcome-categories"
DEFAULT_N_VALUES =list(range(0, 101, 5))
DEFAULT_BLOCKED = [0, 2, 4, 6, 8, 10, 12, 14]
DEFAULT_PROFILES = ["average", "worst_peak"]


def EMERGENCY_STATUS(args):
    if args.model == "v1g":
        return "not_applicable_v1g_no_discharge"
    if not args.emergency_user_ids:
        return "UNVERIFIED_no_eligibility_data_all_internal_evs_treated_as_discharge_eligible"
    return "configured_ids_unverified_by_data_owner"


def outcome_category(row, outcome):
    """Mutually exclusive outcome labels separating solver status from service status."""
    solve = outcome["solve_class"]
    if solve == "baseline":
        return "baseline_N0"
    if row.get("solution_promoted_from_soft_resolve"):
        return "time_limited_service_feasible_via_soft_resolve"
    if solve == "infeasible":
        return "service_infeasible_proven"
    if solve in ("time_limit_no_solution", "other") and not row["service_feasible"]:
        return "unresolved_no_solution"
    if row.get("validation_passed") is False:
        return "constraint_violation"
    if row["service_feasible"]:
        return "optimal_service_feasible" if solve == "optimal" else "time_limited_service_feasible"
    if solve == "optimal":
        return "optimal_service_infeasible"
    if outcome.get("service_failure_proven"):
        return "time_limited_service_infeasible_proven_by_bound"
    return "time_limited_service_unresolved"


def run_case(args, fleet, fleet_meta, profile_name, profile, blocked, seed, out_dir):
    available = np.full(c.SLOTS_PER_DAY, max(0, c.INTERNAL_OUTLETS - blocked), dtype=int)
    case_config = {
        "model": args.model, "power_kw": args.power_kw, "seed": seed, "blocked_outlets": blocked,
        "base_profile": profile_name, "day_type": args.day_type, "n_values": args.n_values,
        "solver": args.solver_cfg, "objective": args.objective, "cycling_penalty": args.cycling_penalty,
        "v1g_hard_service": args.v1g_hard_service, "v2g_diagnostics": not args.no_v2g_diagnostics,
        "fleet_hash": fleet_meta["fleet_hash"],
        "profile_input_sha256": profile["input_sha256"], "site_cap_kw": c.SITE_CAP_KW,
        "formulation": FORMULATION_VERSION,
    }
    config_hash = c.stable_hash(case_config)[:12]
    stem = f"{args.model}_power{args.power_kw}kW_seed{seed}_blocked{blocked}_{profile_name}"
    csv_path, npz_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.npz"
    if csv_path.exists() and npz_path.exists():
        done = pd.read_csv(csv_path)
        if not done.empty and done["case_config_hash"].iloc[0] == config_hash:
            print(f"skip {stem} (complete)", flush=True)
            return done

    n_max = c.FLEET_SIZE
    shape = (len(args.n_values), n_max, c.SLOTS_PER_DAY)
    store = {k: np.full(shape, np.nan) for k in ("charge_kw", "discharge_kw", "soc_after_slot")}
    facility = np.full((len(args.n_values), c.SLOTS_PER_DAY), np.nan)
    statuses = np.full(len(args.n_values), "not_run", dtype="U32")
    rows = []
    for idx, n in enumerate(args.n_values):
        arr = c.fleet_arrays(fleet, n)
        started = time.perf_counter()
        if n == 0:
            outcome = {"status": "Optimal", "solve_class": "baseline", "usable_solution": True,
                       "objective_value": 0.0, "best_bound": 0.0, "mip_gap_abs": 0.0, "mip_gap_rel": 0.0,
                       "runtime_sec": 0.0}
            arrays = {"charge_kw": np.zeros((0, c.SLOTS_PER_DAY)), "discharge_kw": np.zeros((0, c.SLOTS_PER_DAY)),
                      "unmet_kwh": np.zeros(0), "total_load_kw": profile["fixed_kw"].copy()}
        elif args.model == "v1g":
            outcome, arrays = solve_v1g(arr, profile, available, args.power_kw, args.solver_cfg,
                                        args.objective, args.v1g_hard_service)
        else:
            outcome, arrays = solve_v2g(arr, profile, available, args.power_kw, args.solver_cfg,
                                        args.objective, args.cycling_penalty,
                                        diagnose_infeasible=not args.no_v2g_diagnostics)
        row = {"model": args.model.upper(), "power_kw": args.power_kw, "seed": seed, "day_type": args.day_type,
               "blocked_outlets": blocked, "available_internal_outlets": int(available[0]),
               "base_profile": profile_name, "objective_mode": args.objective, "N_internal": n,
               "case_config_hash": config_hash, "fleet_hash": fleet_meta["fleet_hash"]}
        row.update({k: v for k, v in outcome.items() if not isinstance(v, (dict, list))})
        row.setdefault("solution_source", "primary" if arrays is not None else "none")
        if arrays is not None:
            metrics, series = c.evaluate_solution(
                arrays["charge_kw"], arrays["discharge_kw"], arr, profile, available, args.power_kw,
                unmet_kwh_model=arrays["unmet_kwh"], total_load_model=arrays["total_load_kw"],
                allow_discharge=(args.model == "v2g"))
            row.update(metrics)
            # A soft-service re-solve can only rescue a case whose primary solve ended without
            # any solution at the time limit (feasibility unknown); its profiles must then meet
            # every hard constraint, which evaluate_solution checks independently. Proven
            # infeasible cases never pass.
            promoted = (not outcome["usable_solution"] and outcome["solve_class"] == "time_limit_no_solution"
                        and outcome.get("diag_solve_class") in ("optimal", "time_limit_feasible"))
            row["service_feasible"] = bool((outcome["usable_solution"] or promoted)
                                           and metrics["service_requirement_met"] and metrics["validation_passed"])
            row["solution_promoted_from_soft_resolve"] = bool(promoted and row["service_feasible"])
            store["charge_kw"][idx, :n] = arrays["charge_kw"]
            store["discharge_kw"][idx, :n] = arrays["discharge_kw"]
            store["soc_after_slot"][idx, :n] = series["soc"]
            facility[idx] = series["facility_kw"]
        else:
            row["service_feasible"] = False
            row["validation_passed"] = np.nan
        row["optimality_proven"] = bool(row["service_feasible"] and outcome["solve_class"] in ("optimal", "baseline"))
        row["outcome_category"] = outcome_category(row, outcome)
        row["emergency_protection_status"] = EMERGENCY_STATUS(args)
        statuses[idx] = outcome["solve_class"]
        rows.append(row)
        print(f"{stem} N={n:>3} {outcome['solve_class']:<22} service={row['service_feasible']!s:<5} "
              f"peak={row.get('facility_peak_kw', float('nan')):8.1f} gap={outcome.get('mip_gap_abs', float('nan')):.2f} "
              f"valid={row.get('validation_passed')} t={time.perf_counter() - started:5.1f}s", flush=True)

    mask = c.window_mask(fleet["arrival_slot"].to_numpy(), fleet["dwell_slots"].to_numpy())
    np.savez_compressed(
        npz_path, n_internal=np.asarray(args.n_values), slot_start_minute=np.arange(c.SLOTS_PER_DAY) * c.SLOT_MINUTES,
        charge_power_kw=store["charge_kw"], discharge_power_kw=store["discharge_kw"],
        net_power_kw=store["charge_kw"] - store["discharge_kw"], soc_after_slot=store["soc_after_slot"],
        facility_load_kw=facility, building_load_kw=profile["building_kw"], external_load_kw=profile["external_kw"],
        fixed_load_kw=profile["fixed_kw"], ev_headroom_kw=profile["headroom_kw"],
        fixed_excess_kw=profile["fixed_excess_kw"], available_internal_outlets=available, window_mask=mask,
        initial_soc=fleet["starting_soc"].to_numpy(), arrival_slot=fleet["arrival_slot"].to_numpy(),
        dwell_slots=fleet["dwell_slots"].to_numpy(), energy_required_kwh=fleet["energy_required_kwh"].to_numpy(),
        emergency_vehicle=fleet["emergency_vehicle"].to_numpy(), session_id=fleet["session_id"].to_numpy(str),
        solve_status=statuses, model=args.model.upper(), power_kw=args.power_kw, seed=seed, blocked=blocked,
        base_profile=profile_name, profile_source=profile["source"], objective_mode=args.objective,
        charge_efficiency=c.CHARGE_EFFICIENCY, discharge_efficiency=c.DISCHARGE_EFFICIENCY,
        battery_capacity_kwh=c.BATTERY_CAPACITY_KWH, soc_min=c.SOC_MIN, soc_max=c.SOC_MAX,
        discharge_reserve=c.V2G_DISCHARGE_RESERVE_SOC, site_cap_kw=c.SITE_CAP_KW,
        fleet_hash=fleet_meta["fleet_hash"], case_config=json.dumps(case_config))
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)   # written last: its presence marks the case complete
    return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("v1g", "v2g"), required=True)
    parser.add_argument("--power-kw", type=int, choices=c.POWER_CONFIGURATIONS_KW, default=11)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--day-type", choices=("weekday", "friday", "weekend"), default="weekday")
    parser.add_argument("--blocked", nargs="+", type=int, default=DEFAULT_BLOCKED)
    parser.add_argument("--profiles", nargs="+", choices=DEFAULT_PROFILES, default=DEFAULT_PROFILES)
    parser.add_argument("--n-values", nargs="+", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument("--objective", default=None,
                        help="Defaults to the model's intended objective (V1G: D_service_first_peak, V2G: A_peak).")
    parser.add_argument("--cycling-penalty", type=float, default=1e-3, help="V2G discharge tie-breaker per kWh.")
    parser.add_argument("--v1g-hard-service", action="store_true",
                        help="Diagnostic: also enforce V1G service as a hard constraint.")
    parser.add_argument("--no-v2g-diagnostics", action="store_true",
                        help="Skip the soft-service diagnostic re-solve of service-infeasible V2G cases.")
    parser.add_argument("--time-limit", type=float, default=c.DEFAULT_SOLVER["time_limit_sec"])
    parser.add_argument("--gap-rel", type=float, default=c.DEFAULT_SOLVER["gap_rel"])
    parser.add_argument("--gap-abs", type=float, default=c.DEFAULT_SOLVER["gap_abs"])
    parser.add_argument("--emergency-user-ids", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fleet-dir", default=str(c.DEFAULT_FLEET_DIR))
    args = parser.parse_args(argv)
    args.objective = args.objective or ("D_service_first_peak" if args.model == "v1g" else "A_peak")
    allowed = V1G_OBJECTIVES if args.model == "v1g" else V2G_OBJECTIVES
    if args.objective not in allowed:
        parser.error(f"--objective for {args.model} must be one of {allowed}")
    if 0 not in args.n_values:
        args.n_values = [0] + args.n_values
    args.n_values = sorted(set(args.n_values))
    if max(args.n_values) > c.FLEET_SIZE:
        parser.error(f"N may not exceed the fleet size {c.FLEET_SIZE}")
    args.solver_cfg = dict(c.DEFAULT_SOLVER, time_limit_sec=args.time_limit, gap_rel=args.gap_rel, gap_abs=args.gap_abs)
    return args


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.output_dir).resolve() / f"{args.model.upper()}_power{args.power_kw}kW"
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = c.load_fixed_load_profiles(args.day_type)
    if args.model == "v2g" and not args.emergency_user_ids:
        print("WARNING: no emergency-vehicle eligibility data supplied; V2G emergency discharge protection is "
              "NOT verified and every internal EV is treated as discharge-eligible.", flush=True)
    manifest = {"argv": sys.argv, "args": {k: v for k, v in vars(args).items()}, "assumptions": c.ASSUMPTIONS,
                "profiles": {k: {"source": v["source"], "fixed_peak_kw": float(v["fixed_kw"].max()),
                                 "fixed_excess_slots": int((v["fixed_excess_kw"] > 0).sum())}
                             for k, v in profiles.items()},
                "emergency_protection_status": EMERGENCY_STATUS(args), "python": sys.version, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "fleets": {}}
    for seed in args.seeds:
        config = c.fleet_config(seed, args.day_type, emergency_user_ids=args.emergency_user_ids)
        fleet, fleet_path, fleet_meta = c.ensure_fleet(config, args.fleet_dir)
        manifest["fleets"][str(seed)] = {"path": str(fleet_path), "fleet_hash": fleet_meta["fleet_hash"],
                                         **fleet_meta["info"]}
        for blocked in args.blocked:
            for profile_name in args.profiles:
                run_case(args, fleet, fleet_meta, profile_name, profiles[profile_name], blocked, seed, out_dir)
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest_path = out_dir / f"manifest_seeds{'-'.join(map(str, args.seeds))}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()

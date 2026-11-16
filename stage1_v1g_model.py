"""Corrected Stage 1 V1G (unidirectional managed charging) MILP.

Ported from ``V1G_optimization`` in ``sensitivity analysis with peak limits 11kw.py``
(the original script is left unchanged as the pre-correction reference).

Intended objective (preserved): **service first** -- deliver every EV's energy
requirement as far as physically possible, then minimise the peak of the managed EV
charging load (objective ``D_service_first_peak``: LEXI_M * sum(unmet) + peak_ev_kw).
Unmet energy is a soft, dominantly penalised slack so that resource shortages are
measured (avoidable unmet energy) instead of turning the whole case infeasible.

Corrections versus the original script:
* facility load = building (powerhospitalalone) + reconstructed external charging +
  managed EV load (no hospitalpower_kw / powerev_kw);
* headroom = max(0, 2000 kW - building - external) instead of ignoring external load;
* fleet, fixed load and solver settings shared with V2G through ``stage1_common``;
* hard-coded debugging removed; outcomes classified from PuLP solution status.
"""

import numpy as np
import pulp as lp

import stage1_common as c

OBJECTIVES = ("D_service_first_peak", "A_peak", "B_peak_above_baseline")


def build_v1g_model(fleet_arr, profile, available_outlets, power_kw, objective="D_service_first_peak",
                    hard_service=False, soc_form="recursive"):
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown V1G objective {objective}")
    n = len(fleet_arr["energy_kwh"])
    vehicles, slots = range(n), range(c.SLOTS_PER_DAY)
    building, external = profile["building_kw"], profile["external_kw"]
    headroom = profile["headroom_kw"]
    unavoidable = c.unavoidable_shortfall_kwh(fleet_arr, power_kw)

    model = lp.LpProblem("Stage1_V1G", lp.LpMinimize)
    p = lp.LpVariable.dicts("charge_kw", ((i, t) for i in vehicles for t in slots), lowBound=0.0, upBound=power_kw)
    s = lp.LpVariable.dicts("charge_state", ((i, t) for i in vehicles for t in slots), cat="Binary")
    unmet = lp.LpVariable.dicts("unmet_kwh", vehicles, lowBound=0.0)
    total_load = lp.LpVariable.dicts("facility_kw", slots)
    total_ev = lp.LpVariable.dicts("ev_kw", slots, lowBound=0.0)
    peak_ev = lp.LpVariable("peak_ev_kw", lowBound=0.0)
    max_load = lp.LpVariable("max_facility_kw")

    for t in slots:
        model += total_ev[t] == lp.lpSum(p[i, t] for i in vehicles)
        model += total_load[t] == total_ev[t] + float(building[t]) + float(external[t])
        model += total_ev[t] <= peak_ev
        model += total_ev[t] <= float(headroom[t])            # controllable EV headroom only
        model += total_load[t] <= max_load
        model += lp.lpSum(s[i, t] for i in vehicles) <= int(available_outlets[t])

    for i in vehicles:
        window = c.window_slots(fleet_arr["arrival_slot"][i], fleet_arr["dwell_slots"][i])
        in_window = set(window)
        for t in slots:
            if t in in_window:
                model += p[i, t] <= power_kw * s[i, t]
            else:
                model += p[i, t] == 0
                model += s[i, t] == 0
        model += lp.lpSum(p[i, t] * c.SLOT_HOURS for t in window) + unmet[i] == float(fleet_arr["energy_kwh"][i])
        # SoC recursion (equivalent to the original cumulative-sum upper-bound constraints).
        previous = float(fleet_arr["soc0"][i])
        for k, t in enumerate(window):
            if soc_form == "cumulative":      # original formulation, kept for equivalence tests
                model += (float(fleet_arr["soc0"][i]) + lp.lpSum(p[i, window[j]] for j in range(k + 1))
                          * c.CHARGE_EFFICIENCY * c.SLOT_HOURS / c.BATTERY_CAPACITY_KWH) <= c.SOC_MAX
                continue
            soc_after = lp.LpVariable(f"soc_{i}_{t}", lowBound=c.SOC_MIN, upBound=c.SOC_MAX)
            model += soc_after == previous + p[i, t] * (c.CHARGE_EFFICIENCY * c.SLOT_HOURS / c.BATTERY_CAPACITY_KWH)
            previous = soc_after
        if hard_service:
            model += unmet[i] <= float(unavoidable[i]) + 1e-6

    if objective == "D_service_first_peak":
        model += c.LEXI_M * lp.lpSum(unmet.values()) + peak_ev
    elif objective == "A_peak":
        model += max_load + c.LEXI_M * lp.lpSum(unmet.values())
    else:
        increase = lp.LpVariable("peak_increase_kw", lowBound=0.0)
        for t in slots:
            model += increase >= total_load[t] - float(building[t] + external[t])
        model += increase + c.LEXI_M * lp.lpSum(unmet.values())
    return model, {"p": p, "s": s, "unmet": unmet, "total_load": total_load, "peak_ev": peak_ev}


def solve_v1g(fleet_arr, profile, available_outlets, power_kw, solver_cfg=None,
              objective="D_service_first_peak", hard_service=False, keep_log_path=None, soc_form="recursive"):
    n = len(fleet_arr["energy_kwh"])
    model, var = build_v1g_model(fleet_arr, profile, available_outlets, power_kw, objective, hard_service, soc_form)
    outcome = c.solve_model(model, solver_cfg, keep_log_path)
    outcome.update(model="V1G", objective_mode=objective, hard_service=hard_service, solution_source="primary")
    # Under the service-first objective (LEXI_M * unmet + peak_ev) the best bound implies
    # unmet >= (bound - max possible peak_ev) / LEXI_M, so a time-limited run can still
    # *prove* a service failure.
    if objective == "D_service_first_peak" and np.isfinite(outcome.get("best_bound", np.nan)):
        unavoidable = float(c.unavoidable_shortfall_kwh(fleet_arr, power_kw).sum())
        peak_upper = min(n * power_kw, float(np.max(profile["headroom_kw"])))
        outcome["unmet_lower_bound_kwh"] = max(0.0, (outcome["best_bound"] - peak_upper) / c.LEXI_M)
        outcome["service_failure_proven"] = bool(outcome["unmet_lower_bound_kwh"]
                                                 > unavoidable + c.SERVICE_TOLERANCE_KWH)
    if not outcome["usable_solution"]:
        return outcome, None
    charge = np.array([[var["p"][i, t].varValue or 0.0 for t in range(c.SLOTS_PER_DAY)] for i in range(n)])
    arrays = {
        "charge_kw": charge,
        "discharge_kw": np.zeros_like(charge),
        "unmet_kwh": np.array([var["unmet"][i].varValue or 0.0 for i in range(n)]),
        "total_load_kw": np.array([var["total_load"][t].varValue or 0.0 for t in range(c.SLOTS_PER_DAY)]),
    }
    outcome["managed_peak_obj_kw"] = float(var["peak_ev"].varValue or 0.0)
    return outcome, arrays

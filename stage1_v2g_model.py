"""Corrected Stage 1 V2G (bidirectional) MILP.

Ported from ``V1G_optimization`` in
``sensetivity analysis with peak limits v2g - correct soc11.py`` (the original script
is left unchanged as the pre-correction reference).

Intended objective (preserved): **facility peak minimisation** (``A_peak``: minimise
max_t facility load) subject to enforced charging service -- each EV's unmet energy may
not exceed its physically unavoidable shortfall (window/battery limited). If the
service requirement cannot be met the case is reported infeasible.

A secondary tie-breaker ``cycling_penalty * discharged kWh`` (default 1e-3 per kWh,
i.e. 1000 kWh of discharge is worth 1 kW of peak) removes arbitrary charge/discharge
cycling among the many alternative optima of a pure min-max objective. Set it to 0 to
reproduce the pure peak objective.

Physical constraints: charge/discharge are separate non-negative variables with binary
modes (no simultaneous charging and discharging per EV); asymmetric efficiencies 0.90
and 0.95; SoC within [0, 1]; discharge only while SoC after the slot stays >= 50 %;
emergency vehicles cannot discharge; outlet count limits active (charging or
discharging) EVs; per-EV power limits are enforced individually, so the net site
headroom constraint cannot be used to exceed any charger, battery or service limit.
"""

import numpy as np
import pulp as lp

import stage1_common as c

OBJECTIVES = ("A_peak", "B_peak_above_baseline")


def build_v2g_model(fleet_arr, profile, available_outlets, power_kw, objective="A_peak",
                    cycling_penalty=1e-3, reserve_soc=c.V2G_DISCHARGE_RESERVE_SOC, soft_service=False,
                    soc_form="recursive"):
    """``soft_service`` is used only for the diagnostic re-solve of a service-infeasible
    case: the unmet slack is then penalised (LEXI_M) instead of bounded."""
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown V2G objective {objective}")
    n = len(fleet_arr["energy_kwh"])
    vehicles, slots = range(n), range(c.SLOTS_PER_DAY)
    building, external = profile["building_kw"], profile["external_kw"]
    headroom = profile["headroom_kw"]
    unavoidable = c.unavoidable_shortfall_kwh(fleet_arr, power_kw)
    eta_c, eta_d = c.CHARGE_EFFICIENCY, c.DISCHARGE_EFFICIENCY

    model = lp.LpProblem("Stage1_V2G", lp.LpMinimize)
    ch = lp.LpVariable.dicts("charge_kw", ((i, t) for i in vehicles for t in slots), lowBound=0.0, upBound=power_kw)
    dis = lp.LpVariable.dicts("discharge_kw", ((i, t) for i in vehicles for t in slots), lowBound=0.0, upBound=power_kw)
    u = lp.LpVariable.dicts("charge_state", ((i, t) for i in vehicles for t in slots), cat="Binary")
    w = lp.LpVariable.dicts("discharge_state", ((i, t) for i in vehicles for t in slots), cat="Binary")
    unmet = lp.LpVariable.dicts("unmet_kwh", vehicles, lowBound=0.0)
    total_load = lp.LpVariable.dicts("facility_kw", slots)
    total_ev = lp.LpVariable.dicts("ev_net_kw", slots)
    max_load = lp.LpVariable("max_facility_kw")

    for t in slots:
        model += total_ev[t] == lp.lpSum(ch[i, t] - dis[i, t] for i in vehicles)
        model += total_load[t] == total_ev[t] + float(building[t]) + float(external[t])
        model += total_load[t] <= max_load
        model += total_ev[t] <= float(headroom[t])            # net import limit of controllable EVs
        model += lp.lpSum(u[i, t] + w[i, t] for i in vehicles) <= int(available_outlets[t])

    for i in vehicles:
        window = c.window_slots(fleet_arr["arrival_slot"][i], fleet_arr["dwell_slots"][i])
        in_window = set(window)
        for t in slots:
            if t in in_window:
                model += u[i, t] + w[i, t] <= 1
                model += ch[i, t] <= power_kw * u[i, t]
                model += dis[i, t] <= power_kw * w[i, t]
                if fleet_arr["emergency"][i]:
                    model += w[i, t] == 0
            else:
                for var in (ch, dis, u, w):
                    model += var[i, t] == 0
        # SoC after each window slot via a recursion (equivalent to the original
        # cumulative-sum form, but O(window) instead of O(window^2) terms).
        previous = float(fleet_arr["soc0"][i])
        for k, t in enumerate(window):
            if soc_form == "cumulative":      # original formulation, kept for equivalence tests
                soc_after = float(fleet_arr["soc0"][i]) + lp.lpSum(
                    ch[i, window[j]] * eta_c - dis[i, window[j]] * (1.0 / eta_d) for j in range(k + 1)
                ) * c.SLOT_HOURS / c.BATTERY_CAPACITY_KWH
                model += soc_after <= c.SOC_MAX
                model += soc_after >= c.SOC_MIN
            else:
                soc_after = lp.LpVariable(f"soc_{i}_{t}", lowBound=c.SOC_MIN, upBound=c.SOC_MAX)
                model += soc_after == previous + (ch[i, t] * eta_c - dis[i, t] * (1.0 / eta_d)) \
                    * (c.SLOT_HOURS / c.BATTERY_CAPACITY_KWH)
                previous = soc_after
            model += soc_after >= reserve_soc - (1 - w[i, t])
        delivered = lp.lpSum((ch[i, t] - dis[i, t] * (1.0 / (eta_c * eta_d))) * c.SLOT_HOURS for t in window)
        model += delivered + unmet[i] == float(fleet_arr["energy_kwh"][i])
        if not soft_service:
            model += unmet[i] <= float(unavoidable[i]) + 1e-6  # enforced charging service

    tie_break = cycling_penalty * lp.lpSum(dis[i, t] * c.SLOT_HOURS for i in vehicles for t in slots)
    if soft_service:
        tie_break = tie_break + c.LEXI_M * lp.lpSum(unmet.values())
    if objective == "A_peak":
        model += max_load + tie_break
    else:
        increase = lp.LpVariable("peak_increase_kw", lowBound=0.0)
        for t in slots:
            model += increase >= total_load[t] - float(building[t] + external[t])
        model += increase + tie_break
    return model, {"ch": ch, "dis": dis, "unmet": unmet, "total_load": total_load, "max_load": max_load}


def solve_v2g(fleet_arr, profile, available_outlets, power_kw, solver_cfg=None, objective="A_peak",
              cycling_penalty=1e-3, keep_log_path=None, diagnose_infeasible=False, soc_form="recursive"):
    """Solve with enforced service. If no service-feasible solution is found and
    ``diagnose_infeasible`` is set, re-solve with penalised (soft) service.

    * Primary ``infeasible``: the re-solve only quantifies the shortfall; the case fails.
    * Primary ``time_limit_no_solution``: feasibility is unknown. If the re-solve reaches
      zero avoidable unmet energy, its solution satisfies every hard constraint and is a
      valid (not proven-optimal) service-feasible solution; the runner checks this
      independently. ``solution_source`` records which solve produced the profiles."""
    n = len(fleet_arr["energy_kwh"])
    model, var = build_v2g_model(fleet_arr, profile, available_outlets, power_kw, objective, cycling_penalty,
                                 soc_form=soc_form)
    outcome = c.solve_model(model, solver_cfg, keep_log_path)
    outcome.update(model="V2G", objective_mode=objective, cycling_penalty=cycling_penalty, solution_source="primary")
    if not outcome["usable_solution"]:
        if not diagnose_infeasible:
            return outcome, None
        model, var = build_v2g_model(fleet_arr, profile, available_outlets, power_kw, objective,
                                     cycling_penalty, soft_service=True)
        diag = c.solve_model(model, solver_cfg)
        outcome.update({f"diag_{k}": v for k, v in diag.items()
                        if k in ("solve_class", "runtime_sec", "mip_gap_abs", "objective_value", "best_bound")})
        if not diag["usable_solution"]:
            return outcome, None
        outcome["solution_source"] = "soft_service_diagnostic"
    grid = range(c.SLOTS_PER_DAY)
    arrays = {
        "charge_kw": np.array([[var["ch"][i, t].varValue or 0.0 for t in grid] for i in range(n)]),
        "discharge_kw": np.array([[var["dis"][i, t].varValue or 0.0 for t in grid] for i in range(n)]),
        "unmet_kwh": np.array([var["unmet"][i].varValue or 0.0 for i in range(n)]),
        "total_load_kw": np.array([var["total_load"][t].varValue or 0.0 for t in grid]),
    }
    outcome["max_load_obj_kw"] = float(var["max_load"].varValue or 0.0)
    return outcome, arrays

"""Shared, model-independent Stage 1 components for the corrected V1G/V2G studies.

This module deliberately contains no optimisation model. The two MILPs live in
``stage1_v1g_model.py`` and ``stage1_v2g_model.py`` and keep their own objectives.
What is shared here is everything that must be identical for a matched comparison:

* fixed facility load (provisional building-only load + reconstructed external
  charging), taken from the existing Stage 2 preparation outputs;
* the internal (controllable) fleet realisation for each seed;
* site headroom, solver configuration and solver-outcome classification;
* post-solve evaluation, physical validation and hosting-capacity metrics.

Assumptions that remain unverified are recorded in ``ASSUMPTIONS`` and written to
every run manifest.
"""

import hashlib
import io
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pulp as lp

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------
# Validated Stage 1 configuration
# --------------------------------------------------------------------------------------
SLOT_MINUTES = 15
SLOTS_PER_DAY = 96
SLOT_HOURS = SLOT_MINUTES / 60.0

CHARGE_EFFICIENCY = 0.90
DISCHARGE_EFFICIENCY = 0.95
BATTERY_CAPACITY_KWH = 60.0
SOC_MIN = 0.0
SOC_MAX = 1.0
V2G_DISCHARGE_RESERVE_SOC = 0.50
SITE_CAP_KW = 2000.0
INTERNAL_OUTLETS = 17
EXTERNAL_OUTLETS = 5
POWER_CONFIGURATIONS_KW = (11, 22)

DWELL_MIN_MINUTES = 10
OPERATIONAL_DWELL_CAP_MINUTES = 24 * 60
FLEET_SIZE = 100
MIN_DAY_TYPE_POOL = 30
FLEET_SAMPLING_VERSION = "stage1-corrected-v1:internal-only,random-order,without-replacement"

SERVICE_TOLERANCE_KWH = 1e-3
POWER_TOLERANCE_KW = 1e-3
SOC_TOLERANCE = 1e-5
LEXI_M = 1e4

FIXED_LOAD_CSV = ROOT / "stage2_data" / "shared_inputs" / "stage2_fixed_load.csv"
INTERNAL_SESSIONS_CSV = ROOT / "stage2_data" / "shared_inputs" / "stage2_internal_sessions.csv"
DEFAULT_FLEET_DIR = ROOT / "stage1_fleets"

ASSUMPTIONS = [
    "Building load is the provisional powerhospitalalone series; its meter definition is unverified.",
    "External/public charging is the Stage 2 reconstruction (11 kW from recorded start until recorded energy, "
    "capped at recorded stop). It is assumed, not measured; 186 historical sessions are 11 kW-infeasible.",
    "powerev_kw and hospitalpower_kw are never added to the facility load.",
    "The hospital clock is provisionally treated as UTC interval starts (as in Stage 2); session arrival "
    "minutes use the UTC timestamps.",
    "Internal sessions are the Stage 2 'internal' records overlapping the hospital grid; recorded stop is "
    "treated as departure although it may be charging end.",
    "Single representative day: sessions are wrapped cyclically onto 96 slots and dwell is capped at 24 h.",
    "Outlets constrain simultaneously active (charging or discharging) EVs, not physical plug occupancy.",
    "Arrival SoC = 1 - eta_c * E / 60 kWh; energy targets are charger-side kWh, clipped at 60 kWh.",
    "Emergency-vehicle identities are not available; no vehicle is flagged unless IDs are supplied.",
    "External charging uses the 5 public outlets and does not block the 17 internal outlets in Stage 1.",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def day_type_of(timestamp):
    weekday = pd.Timestamp(timestamp).day_of_week
    if weekday == 4:
        return "friday"
    if weekday >= 5:
        return "weekend"
    return "weekday"


# --------------------------------------------------------------------------------------
# 1. Fixed facility load from the Stage 2 preparation
# --------------------------------------------------------------------------------------
def load_fixed_load_profiles(day_type, path=FIXED_LOAD_CSV):
    """Return representative-day profiles of building, external and fixed load.

    ``average``: slot-wise mean over all complete days of ``day_type``.
    ``worst_peak``: the complete day of ``day_type`` with the highest fixed-load
    (building + external) peak; building and external are taken from the same day.
    """
    frame = pd.read_csv(path, parse_dates=["timestamp_nominal"])
    required = {"timestamp_nominal", "building_load_kw_provisional",
                "external_charging_kw_assumed_11kw", "fixed_load_kw_provisional"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    identity = (frame["fixed_load_kw_provisional"] - frame["building_load_kw_provisional"]
                - frame["external_charging_kw_assumed_11kw"]).abs().max()
    if identity > 1e-6:
        raise ValueError(f"Stage 2 fixed load identity violated by {identity} kW")

    frame = frame.dropna(subset=["building_load_kw_provisional", "fixed_load_kw_provisional"])
    frame["date"] = frame["timestamp_nominal"].dt.normalize()
    frame["slot"] = (frame["timestamp_nominal"].dt.hour * 60 + frame["timestamp_nominal"].dt.minute) // SLOT_MINUTES
    counts = frame.groupby("date")["slot"].nunique()
    complete = counts[counts == SLOTS_PER_DAY].index
    frame = frame[frame["date"].isin(complete)]
    frame = frame[frame["date"].map(day_type_of) == day_type]
    if frame.empty:
        raise ValueError(f"No complete {day_type} days in {path}")

    def pivot(column):
        return frame.pivot_table(index="date", columns="slot", values=column, aggfunc="first").reindex(
            columns=range(SLOTS_PER_DAY))

    building = pivot("building_load_kw_provisional")
    external = pivot("external_charging_kw_assumed_11kw")
    fixed = building + external
    worst_date = fixed.max(axis=1).idxmax()
    profiles = {
        "average": {
            "building_kw": building.mean(axis=0).to_numpy(float),
            "external_kw": external.mean(axis=0).to_numpy(float),
            "source": f"mean of {len(building)} complete {day_type} days",
            "source_dates": [str(d.date()) for d in building.index],
        },
        "worst_peak": {
            "building_kw": building.loc[worst_date].to_numpy(float),
            "external_kw": external.loc[worst_date].to_numpy(float),
            "source": f"{day_type} day with highest fixed-load peak: {worst_date.date()}",
            "source_dates": [str(worst_date.date())],
        },
    }
    for profile in profiles.values():
        profile["fixed_kw"] = profile["building_kw"] + profile["external_kw"]
        headroom, excess = site_headroom(profile["fixed_kw"])
        profile["headroom_kw"] = headroom
        profile["fixed_excess_kw"] = excess
        profile["input_sha256"] = sha256_file(path)
    return profiles


def site_headroom(fixed_kw, site_cap_kw=SITE_CAP_KW):
    """Controllable EV headroom and uncontrollable fixed excess above the site cap.

    The cap is applied to the *net* import of the controllable EVs on top of the fixed
    (building + external) demand. Where the fixed demand alone exceeds the cap, the
    excess is uncontrollable: it is reported, and EVs may not add import in that slot
    (V2G may still export), but it does not make the scenario infeasible by itself.
    """
    fixed_kw = np.asarray(fixed_kw, dtype=float)
    return np.maximum(0.0, site_cap_kw - fixed_kw), np.maximum(0.0, fixed_kw - site_cap_kw)


# --------------------------------------------------------------------------------------
# 2./3. Internal (controllable) fleet sampling
# --------------------------------------------------------------------------------------
def load_internal_pool(day_type, path=INTERNAL_SESSIONS_CSV, min_pool=MIN_DAY_TYPE_POOL):
    """Internal sessions only, classified by the documented ``ev_type`` (user_category)."""
    frame = pd.read_csv(path, dtype={"recorded_user_id": str})
    categories = frame["user_category"].astype(str).str.strip().str.lower()
    if not categories.eq("internal").all():
        raise ValueError("Internal session input contains non-internal records")
    frame = frame[frame["input_status"].astype(str).eq("pending")].copy()
    before = len(frame)
    frame = frame.drop_duplicates(subset=["session_id"])
    frame = frame.drop_duplicates(subset=["evse_id", "recorded_start_utc", "recorded_stop_utc", "recorded_energy_kwh"])
    duplicates_removed = before - len(frame)
    start = pd.to_datetime(frame["recorded_start_utc"], utc=True)
    stop = pd.to_datetime(frame["recorded_stop_utc"], utc=True)
    frame["arrival_minute"] = start.dt.hour * 60 + start.dt.minute
    frame["dwell_minutes_recorded"] = (stop - start).dt.total_seconds() / 60.0
    frame["energy_recorded_kwh"] = pd.to_numeric(frame["recorded_energy_kwh"], errors="raise")
    frame["day_type"] = start.dt.tz_convert(None).map(day_type_of)
    frame = frame[(frame["dwell_minutes_recorded"] > 0) & (frame["energy_recorded_kwh"] > 0)]
    selected = frame[frame["day_type"] == day_type]
    pool_scope = day_type
    if len(selected) < min_pool:
        selected, pool_scope = frame, "all"
    selected = selected.sort_values("session_id", kind="stable").reset_index(drop=True)
    return selected, {"pool_scope": pool_scope, "pool_size": len(selected),
                      "duplicates_removed": int(duplicates_removed)}


def fleet_config(seed, day_type, n_internal=FLEET_SIZE, battery_kwh=BATTERY_CAPACITY_KWH,
                 dwell_cap_minutes=OPERATIONAL_DWELL_CAP_MINUTES, dwell_min_minutes=DWELL_MIN_MINUTES,
                 emergency_user_ids=(), sessions_path=INTERNAL_SESSIONS_CSV):
    return {
        "sampling_version": FLEET_SAMPLING_VERSION,
        "seed": int(seed),
        "day_type": day_type,
        "n_internal": int(n_internal),
        "battery_capacity_kwh": float(battery_kwh),
        "charge_efficiency": CHARGE_EFFICIENCY,
        "dwell_cap_minutes": None if dwell_cap_minutes is None else float(dwell_cap_minutes),
        "dwell_min_minutes": float(dwell_min_minutes),
        "slot_minutes": SLOT_MINUTES,
        "emergency_user_ids": sorted(str(x) for x in emergency_user_ids),
        "sessions_file": os.path.relpath(sessions_path, ROOT),
        "sessions_sha256": sha256_file(sessions_path),
    }


def _fleet_frame_hash(frame):
    return hashlib.sha256(frame.drop(columns=["fleet_hash"], errors="ignore")
                          .to_csv(index=False).encode("utf-8")).hexdigest()


def build_fleet(config):
    """Sample one fleet realisation. Rows are kept intact (arrival, dwell, energy
    stay jointly correlated) and kept in random draw order, so the first N rows form
    nested random subsets."""
    pool, pool_info = load_internal_pool(config["day_type"], ROOT / config["sessions_file"])
    rng = np.random.default_rng(10000 + config["seed"])
    n = config["n_internal"]
    replace = len(pool) < n
    idx = rng.choice(len(pool), size=n, replace=replace)
    rows = pool.iloc[idx].reset_index(drop=True)
    battery = config["battery_capacity_kwh"]
    dwell = np.maximum(rows["dwell_minutes_recorded"].to_numpy(float), config["dwell_min_minutes"])
    if config["dwell_cap_minutes"] is not None:
        dwell = np.minimum(dwell, config["dwell_cap_minutes"])
    energy_recorded = rows["energy_recorded_kwh"].to_numpy(float)
    energy = np.clip(energy_recorded, 0.1, battery)
    emergency_ids = set(config["emergency_user_ids"])
    fleet = pd.DataFrame({
        "order": np.arange(n),
        "session_id": rows["session_id"].to_numpy(str),
        "source_user_id": rows["recorded_user_id"].astype(str).to_numpy(),
        "evse_id": rows["evse_id"].astype(str).to_numpy(),
        "session_day_type": rows["day_type"].to_numpy(str),
        "arrival_minute": rows["arrival_minute"].to_numpy(int),
        "dwell_minutes_recorded": rows["dwell_minutes_recorded"].to_numpy(float),
        "dwell_minutes": dwell,
        "energy_recorded_kwh": energy_recorded,
        "energy_required_kwh": energy,
        "energy_clipped": ~np.isclose(energy, energy_recorded),
        "emergency_vehicle": rows["recorded_user_id"].astype(str).isin(emergency_ids).to_numpy(),
    })
    fleet["arrival_slot"] = (fleet["arrival_minute"] // SLOT_MINUTES).astype(int) % SLOTS_PER_DAY
    fleet["dwell_slots"] = np.clip(np.round(fleet["dwell_minutes"] / SLOT_MINUTES), 1, SLOTS_PER_DAY).astype(int)
    fleet["starting_soc"] = 1.0 - CHARGE_EFFICIENCY * fleet["energy_required_kwh"] / battery
    fleet["fleet_hash"] = _fleet_frame_hash(fleet)
    info = dict(pool_info, sampled_with_replacement=bool(replace),
                duplicate_sessions_in_fleet=int(fleet["session_id"].duplicated().sum()),
                energy_clipped_count=int(fleet["energy_clipped"].sum()))
    return fleet, info


def ensure_fleet(config, fleet_dir=DEFAULT_FLEET_DIR):
    """Load or create the cached fleet. The cache key covers every configuration item,
    so a changed day type, battery, dwell cap, data file or sampling rule never reuses a
    stale file. Cached files are verified against their own content hash."""
    fleet_dir = Path(fleet_dir)
    fleet_dir.mkdir(parents=True, exist_ok=True)
    key = stable_hash(config)[:16]
    stem = f"internal_fleet_{config['day_type']}_seed{config['seed']}_n{config['n_internal']}_{key}"
    csv_path, meta_path = fleet_dir / f"{stem}.csv", fleet_dir / f"{stem}.json"
    dtypes = {"session_id": str, "source_user_id": str, "evse_id": str}
    lock_path = fleet_dir / f"{stem}.lock"
    deadline = time.time() + 300
    while not meta_path.exists():
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:            # another process is creating this fleet
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for {lock_path}")
            time.sleep(0.5)
            continue
        try:
            os.close(fd)
            if not meta_path.exists():
                fleet, info = build_fleet(config)
                text = fleet.drop(columns=["fleet_hash"]).to_csv(index=False)
                reloaded = pd.read_csv(io.StringIO(text), dtype=dtypes)   # hash the on-disk representation
                fleet_hash = _fleet_frame_hash(reloaded)
                reloaded["fleet_hash"] = fleet_hash
                tmp_csv = csv_path.with_suffix(f".tmp{os.getpid()}")
                reloaded.to_csv(tmp_csv, index=False)
                os.replace(tmp_csv, csv_path)
                meta = {"config": config, "cache_key": key, "fleet_hash": fleet_hash, "info": info}
                tmp_meta = meta_path.with_suffix(f".tmp{os.getpid()}")
                tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                os.replace(tmp_meta, meta_path)     # metadata last: marks the cache complete
        finally:
            os.remove(lock_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fleet = pd.read_csv(csv_path, dtype=dtypes)
    if meta.get("config") == config and _fleet_frame_hash(fleet) == meta.get("fleet_hash") \
            and fleet["fleet_hash"].iloc[0] == meta.get("fleet_hash"):
        return fleet, csv_path, meta
    raise RuntimeError(f"Cached fleet {csv_path} failed verification; remove it explicitly to rebuild.")


def fleet_arrays(fleet, n):
    sub = fleet.iloc[:n]
    return {
        "arrival_slot": sub["arrival_slot"].to_numpy(int),
        "dwell_slots": sub["dwell_slots"].to_numpy(int),
        "energy_kwh": sub["energy_required_kwh"].to_numpy(float),
        "soc0": sub["starting_soc"].to_numpy(float),
        "emergency": sub["emergency_vehicle"].to_numpy(bool),
        "session_id": sub["session_id"].to_numpy(str),
    }


def window_slots(arrival_slot, dwell_slots):
    dwell_slots = int(min(max(int(dwell_slots), 0), SLOTS_PER_DAY))
    return [(int(arrival_slot) + k) % SLOTS_PER_DAY for k in range(dwell_slots)]


def window_mask(arrival_slots, dwell_slots):
    mask = np.zeros((len(arrival_slots), SLOTS_PER_DAY), dtype=bool)
    for i, (a, d) in enumerate(zip(arrival_slots, dwell_slots)):
        mask[i, window_slots(a, d)] = True
    return mask


def max_deliverable_kwh(n_window_slots, soc0, power_kw, battery_kwh=BATTERY_CAPACITY_KWH,
                        charge_efficiency=CHARGE_EFFICIENCY, soc_max=SOC_MAX):
    """Charger-side energy deliverable by one EV ignoring other EVs (time and battery limits)."""
    time_limited = max(0, int(n_window_slots)) * power_kw * SLOT_HOURS
    battery_limited = max(0.0, (soc_max - soc0) * battery_kwh / charge_efficiency)
    return min(time_limited, battery_limited)


def unavoidable_shortfall_kwh(fleet_arr, power_kw):
    out = np.zeros(len(fleet_arr["energy_kwh"]))
    for i in range(len(out)):
        n_slots = len(window_slots(fleet_arr["arrival_slot"][i], fleet_arr["dwell_slots"][i]))
        out[i] = max(0.0, fleet_arr["energy_kwh"][i] - max_deliverable_kwh(n_slots, fleet_arr["soc0"][i], power_kw))
    return out


def cap_blocked_shortfall_kwh(fleet_arr, headroom_kw, power_kw):
    """Per-EV energy that could not be delivered even if the EV were alone on site,
    because the site cap leaves too little EV headroom in its window (beyond the
    time/battery-unavoidable shortfall). Exact for V1G; for V2G an energy transfer from
    another EV's discharge could in principle bypass it, so it is an indicator there."""
    unavoidable = unavoidable_shortfall_kwh(fleet_arr, power_kw)
    out = np.zeros(len(unavoidable))
    for i in range(len(out)):
        window = window_slots(fleet_arr["arrival_slot"][i], fleet_arr["dwell_slots"][i])
        cap_limited = sum(min(power_kw, headroom_kw[t]) for t in window) * SLOT_HOURS
        battery_limited = max(0.0, (SOC_MAX - fleet_arr["soc0"][i]) * BATTERY_CAPACITY_KWH / CHARGE_EFFICIENCY)
        deliverable = min(cap_limited, battery_limited)
        out[i] = max(0.0, fleet_arr["energy_kwh"][i] - deliverable - unavoidable[i])
    return out


# --------------------------------------------------------------------------------------
# Solver configuration and outcome classification
# --------------------------------------------------------------------------------------
DEFAULT_SOLVER = {"time_limit_sec": 120, "gap_rel": 1e-3, "gap_abs": 1.0, "options": ["preprocess off"]}

_LOG_NUMBER = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def parse_cbc_log(text):
    info = {}
    for key, label in (("objective_log", "Objective value:"), ("best_bound", "Lower bound:"),
                       ("gap_log", "Gap:"), ("wallclock_sec", "Time (Wallclock seconds):")):
        match = re.search(re.escape(label) + r"\s*" + _LOG_NUMBER, text)
        if match:
            info[key] = float(match.group(1))
    if "Result - Optimal solution found" in text:
        info["log_result"] = "optimal"
    elif "Result - Stopped on time limit" in text or "Exiting on maximum time" in text:
        info["log_result"] = "time_limit"
    elif "Result - Problem proven infeasible" in text or "Problem is infeasible" in text:
        info["log_result"] = "infeasible"
    else:
        info["log_result"] = "unknown"
    return info


def solve_model(model, solver_cfg=None, keep_log_path=None):
    cfg = dict(DEFAULT_SOLVER, **(solver_cfg or {}))
    fd, log_path = tempfile.mkstemp(prefix="cbc_stage1_", suffix=".log")
    os.close(fd)
    solver = lp.PULP_CBC_CMD(msg=False, timeLimit=cfg["time_limit_sec"], gapRel=cfg["gap_rel"],
                             gapAbs=cfg["gap_abs"], options=list(cfg["options"]), logPath=log_path)
    started = time.perf_counter()
    status = model.solve(solver)
    runtime = time.perf_counter() - started
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
        if keep_log_path:
            Path(keep_log_path).write_text(text, encoding="utf-8")
    finally:
        try:
            os.remove(log_path)
        except OSError:
            pass
    info = parse_cbc_log(text)
    sol_status = int(model.sol_status)
    has_solution = sol_status in (lp.LpSolutionOptimal, lp.LpSolutionIntegerFeasible)
    if info["log_result"] == "infeasible" or sol_status == lp.LpSolutionInfeasible:
        solve_class = "infeasible"
    elif info["log_result"] == "time_limit":
        solve_class = "time_limit_feasible" if has_solution else "time_limit_no_solution"
    elif has_solution and sol_status == lp.LpSolutionOptimal:
        solve_class = "optimal"
    elif has_solution:
        solve_class = "time_limit_feasible"
    else:
        solve_class = "other"
    objective = lp.value(model.objective) if has_solution else np.nan
    bound = info.get("best_bound", np.nan)
    if solve_class == "optimal" and not np.isfinite(bound):
        bound = objective
    gap_abs = objective - bound if np.isfinite(objective) and np.isfinite(bound) else np.nan
    gap_rel = gap_abs / max(abs(objective), 1e-9) if np.isfinite(gap_abs) else np.nan
    return {
        "status": lp.LpStatus[status],
        "sol_status": sol_status,
        "solve_class": solve_class,
        "usable_solution": bool(has_solution and solve_class in ("optimal", "time_limit_feasible")),
        "objective_value": objective,
        "best_bound": bound,
        "mip_gap_abs": gap_abs,
        "mip_gap_rel": gap_rel,
        "runtime_sec": runtime,
        "time_limit_sec": cfg["time_limit_sec"],
        "gap_rel_setting": cfg["gap_rel"],
        "gap_abs_setting": cfg["gap_abs"],
    }


# --------------------------------------------------------------------------------------
# Evaluation and physical validation (identical for V1G and V2G)
# --------------------------------------------------------------------------------------
def soc_trajectory(charge_kw, discharge_kw, fleet_arr, battery_kwh=BATTERY_CAPACITY_KWH):
    n = len(fleet_arr["energy_kwh"])
    soc = np.full((n, SLOTS_PER_DAY), np.nan)
    for i in range(n):
        current = float(fleet_arr["soc0"][i])
        for t in window_slots(fleet_arr["arrival_slot"][i], fleet_arr["dwell_slots"][i]):
            current += (charge_kw[i, t] * CHARGE_EFFICIENCY - discharge_kw[i, t] / DISCHARGE_EFFICIENCY) \
                * SLOT_HOURS / battery_kwh
            soc[i, t] = current
    return soc


def evaluate_solution(charge_kw, discharge_kw, fleet_arr, profile, available_outlets, power_kw,
                      unmet_kwh_model=None, total_load_model=None, allow_discharge=False,
                      reserve_soc=V2G_DISCHARGE_RESERVE_SOC):
    """Recompute every reported quantity from the decision profiles and check physics."""
    n = len(fleet_arr["energy_kwh"])
    charge_kw = np.asarray(charge_kw, float).reshape(n, SLOTS_PER_DAY)
    discharge_kw = np.asarray(discharge_kw, float).reshape(n, SLOTS_PER_DAY)
    mask = window_mask(fleet_arr["arrival_slot"], fleet_arr["dwell_slots"])
    building, external = profile["building_kw"], profile["external_kw"]
    fixed = building + external
    net_ev = (charge_kw - discharge_kw).sum(axis=0)
    facility = building + external + net_ev
    soc = soc_trajectory(charge_kw, discharge_kw, fleet_arr)
    delivered = ((charge_kw - discharge_kw / (CHARGE_EFFICIENCY * DISCHARGE_EFFICIENCY)) * SLOT_HOURS).sum(axis=1)
    unmet = np.maximum(0.0, fleet_arr["energy_kwh"] - delivered)
    unavoidable = unavoidable_shortfall_kwh(fleet_arr, power_kw)
    avoidable = np.maximum(0.0, unmet - unavoidable)
    cap_blocked = cap_blocked_shortfall_kwh(fleet_arr, profile["headroom_kw"], power_kw)
    active = ((charge_kw > POWER_TOLERANCE_KW) | (discharge_kw > POWER_TOLERANCE_KW)).sum(axis=0)
    gross_charge = charge_kw.sum(axis=0)
    gross_discharge = discharge_kw.sum(axis=0)
    fixed_peak = float(np.max(fixed))
    facility_peak = float(np.max(facility))
    final_soc = np.array([soc[i, window_slots(fleet_arr["arrival_slot"][i], fleet_arr["dwell_slots"][i])[-1]]
                          if mask[i].any() else fleet_arr["soc0"][i] for i in range(n)])
    discharge_mask = discharge_kw > POWER_TOLERANCE_KW

    checks = {
        "power_bounds": max(0.0, -charge_kw.min(initial=0), -discharge_kw.min(initial=0),
                            charge_kw.max(initial=0) - power_kw, discharge_kw.max(initial=0) - power_kw),
        "outside_window_power": float(np.abs(np.where(mask, 0.0, charge_kw + discharge_kw)).max(initial=0)),
        "simultaneous_charge_discharge": float(np.minimum(charge_kw, discharge_kw).max(initial=0)),
        "discharge_not_allowed": 0.0 if allow_discharge else float(discharge_kw.max(initial=0)),
        "emergency_discharge": float(discharge_kw[fleet_arr["emergency"]].max(initial=0)),
        "outlet_availability": float(np.max(active - np.asarray(available_outlets), initial=0)),
        "site_headroom": float(np.max(net_ev - profile["headroom_kw"], initial=0)),
        "soc_upper": float(np.nanmax(soc - SOC_MAX, initial=0)) if n else 0.0,
        "soc_lower": float(np.nanmax(SOC_MIN - soc, initial=0)) if n else 0.0,
        "discharge_reserve": float(np.max(np.where(discharge_mask, reserve_soc - np.nan_to_num(soc, nan=1.0), 0.0),
                                          initial=0)) if n else 0.0,
        "final_soc_consistency": float(np.max(np.abs(final_soc - (fleet_arr["soc0"] + CHARGE_EFFICIENCY
                                                                   * delivered / BATTERY_CAPACITY_KWH)),
                                              initial=0)) if n else 0.0,
        "unmet_model_consistency": float(np.max(np.abs(np.asarray(unmet_kwh_model) - unmet), initial=0))
        if unmet_kwh_model is not None and n else 0.0,
        "power_balance": float(np.max(np.abs(np.asarray(total_load_model) - facility), initial=0))
        if total_load_model is not None else 0.0,
    }
    tolerances = {k: POWER_TOLERANCE_KW for k in checks}
    tolerances.update(soc_upper=SOC_TOLERANCE, soc_lower=SOC_TOLERANCE, discharge_reserve=SOC_TOLERANCE,
                      final_soc_consistency=SOC_TOLERANCE, unmet_model_consistency=SERVICE_TOLERANCE_KWH,
                      power_balance=1e-2, outlet_availability=0.0)
    failed = [k for k, v in checks.items() if not v <= tolerances[k]]

    metrics = {
        "fixed_peak_kw": fixed_peak,
        "building_peak_kw": float(np.max(building)),
        "facility_peak_kw": facility_peak,
        "peak_reduction_vs_fixed_kw": fixed_peak - facility_peak,
        "ev_added_peak_kw": facility_peak - fixed_peak,
        "fixed_excess_slots": int(np.sum(profile["fixed_excess_kw"] > 0)),
        "fixed_excess_max_kw": float(np.max(profile["fixed_excess_kw"])),
        "facility_over_cap_max_kw": float(max(0.0, facility_peak - SITE_CAP_KW)),
        "ev_net_import_peak_kw": float(np.max(net_ev, initial=0)),
        "ev_net_export_peak_kw": float(max(0.0, -np.min(net_ev, initial=0))),
        "ev_charge_energy_kwh": float(gross_charge.sum() * SLOT_HOURS),
        "ev_discharge_energy_kwh": float(gross_discharge.sum() * SLOT_HOURS),
        "ev_net_export_energy_kwh": float(np.maximum(-net_ev, 0).sum() * SLOT_HOURS),
        "v2v_energy_kwh": float(np.minimum(gross_charge, gross_discharge).sum() * SLOT_HOURS),
        "max_active_evs": int(active.max(initial=0)),
        "required_total_kwh": float(fleet_arr["energy_kwh"].sum()),
        "delivered_total_kwh": float(delivered.sum()),
        "total_unmet_kwh": float(unmet.sum()),
        "unavoidable_unmet_kwh": float(unavoidable.sum()),
        "avoidable_unmet_kwh": float(avoidable.sum()),
        "cap_blocked_shortfall_kwh": float(cap_blocked.sum()),
        "cap_blocked_ev_count": int(np.sum(cap_blocked > SERVICE_TOLERANCE_KWH)),
        "unmet_ev_count": int(np.sum(unmet > SERVICE_TOLERANCE_KWH)),
        "window_infeasible_ev_count": int(np.sum(unavoidable > SERVICE_TOLERANCE_KWH)),
        "served_fraction": float(delivered.sum() / fleet_arr["energy_kwh"].sum()) if n else 1.0,
        "soc_min_observed": float(np.nanmin(soc)) if n and mask.any() else np.nan,
        "soc_max_observed": float(np.nanmax(soc)) if n and mask.any() else np.nan,
        "service_requirement_met": bool(avoidable.sum() <= SERVICE_TOLERANCE_KWH),
        "validation_passed": not failed,
        "validation_failures": ";".join(failed),
    }
    metrics.update({f"check_{k}": v for k, v in checks.items()})
    series = {"net_ev_kw": net_ev, "facility_kw": facility, "soc": soc}
    return metrics, series


# --------------------------------------------------------------------------------------
# Hosting capacity
# --------------------------------------------------------------------------------------
def contiguous_capacity(n_values, passed):
    """Largest N such that every tested N' <= N passed (N=0 always counts)."""
    order = np.argsort(n_values)
    capacity = -1
    for idx in order:
        if not passed[idx]:
            break
        capacity = int(n_values[idx])
    return capacity


def non_monotonic_ns(n_values, passed):
    order = np.argsort(n_values)
    ns, flags = np.asarray(n_values)[order], np.asarray(passed, bool)[order]
    first_fail = np.argmax(~flags) if (~flags).any() else None
    if first_fail is None:
        return []
    return [int(n) for n, ok in zip(ns[first_fail:], flags[first_fail:]) if ok]


def probabilistic_capacity(df, group_cols, pass_col="service_feasible", thresholds=(0.90, 0.95, 1.00)):
    prob = (df.groupby(group_cols + ["N_internal"], dropna=False)
              .agg(seeds_tested=("seed", "nunique"), pass_count=(pass_col, "sum"),
                   pass_probability=(pass_col, "mean"))
              .reset_index())
    rows = []
    for keys, group in prob.groupby(group_cols, dropna=False):
        group = group.sort_values("N_internal")
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        ns, probs = group["N_internal"].to_numpy(), group["pass_probability"].to_numpy()
        for threshold in thresholds:
            pct = int(round(threshold * 100))
            ok = probs >= threshold - 1e-12
            row[f"HC_{pct}"] = contiguous_capacity(ns, ok)
            row[f"HC_{pct}_any_pass"] = int(ns[ok].max()) if ok.any() else -1
            row[f"HC_{pct}_non_monotonic_N"] = " ".join(map(str, non_monotonic_ns(ns, ok)))
        row["seeds_tested_min"] = int(group["seeds_tested"].min())
        row["max_n_tested"] = int(ns.max())
        rows.append(row)
    return prob, pd.DataFrame(rows)

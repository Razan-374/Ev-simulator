#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pulp as lp
import re
from matplotlib import pyplot as plt
from queue import PriorityQueue
import pandas as pd

SLOT_MINUTES = 15
SLOTS_PER_DAY = int(1440 / SLOT_MINUTES)  # 96 slots
SLOT_HOURS = SLOT_MINUTES / 60.0
UNMET_ENERGY_PENALTY_PER_KWH = 1000.0

# Dwell scenario controls for single-day planning vs realism studies.
# "operational": cap dwell at 24h for strict day-ahead planning.
# "realism": keep long overstays (no upper cap).
DWELL_SCENARIO = "operational"  # options: "operational", "realism"
DWELL_MIN_MINUTES = 10
OPERATIONAL_DWELL_CAP_MINUTES = 24 * 60
ALLOW_FULL_INTERNAL_BLOCK = False  # when False, keep at least one internal charger available if any exist
BEHAVIOR_SCENARIO = "workplace"  # options: "hospital", "workplace", "residential"
BASELINE_COMPARISON_SEED = 42
MONTE_CARLO_RUNS = 30

BEHAVIOR_SCENARIO_PARAMS = {
    "workplace": {
        "arrival_mean_h": 8.5,
        "arrival_std_h": 1.0,
        "arrival_min_h": 6.0,
        "arrival_max_h": 11.0,
        "dwell_mean_h": 8.0,
        "dwell_std_h": 1.5,
        "dwell_min_h": 4.0,
        "dwell_max_h": 10.0,
        "energy_mean_kwh": 15.0,
        "energy_std_kwh": 5.0,
        "energy_min_kwh": 5.0,
        "energy_max_kwh": 30.0,
    },
    "residential": {
        "arrival_mean_h": 18.5,
        "arrival_std_h": 1.5,
        "arrival_min_h": 16.0,
        "arrival_max_h": 22.0,
        "dwell_mean_h": 12.0,
        "dwell_std_h": 2.0,
        "dwell_min_h": 8.0,
        "dwell_max_h": 16.0,
        "energy_mean_kwh": 20.0,
        "energy_std_kwh": 6.0,
        "energy_min_kwh": 8.0,
        "energy_max_kwh": 40.0,
    },
}

INTERNAL_DAILY_COUNT_SCENARIOS = {
    "hospital": {
        "counts": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "probs": np.array([0.1009, 0.1374, 0.1304, 0.1617, 0.1496, 0.1478, 0.0974, 0.0556, 0.0139, 0.0052]),
    },
}

SCENARIO_BASELINE_INTERNAL_COUNTS = {
    # Main strategy-comparison assumption:
    # keep outlet/session availability fixed across scenarios so differences
    # reflect charging-session behavior rather than infrastructure scale.
    # Reduced charger availability is examined separately in the sensitivity study.
    "hospital": 15,
    "workplace": 15,
    "residential": 15,
}

SCENARIO_RULES = {
    "hospital": {
        "weekday_only": False,
        "include_external_sessions": False,
        "external_as_fixed_load": False,
        "blocked_fraction_range": None,
    },
    "workplace": {
        "weekday_only": True,
        "include_external_sessions": False,
        "external_as_fixed_load": False,
        # Keep baseline comparison outlet availability aligned with hospital.
        "blocked_fraction_range": None,
    },
    "residential": {
        "weekday_only": True,
        "include_external_sessions": False,
        "external_as_fixed_load": False,
        # Keep baseline comparison outlet availability aligned with hospital.
        "blocked_fraction_range": None,
    },
}

# Behavior-driven samplers
# ----------------------
# Defaults can be replaced with histogram/KDE fits from your dataset.
arrival_histogram = np.ones(1440)  # probability mass across minutes of day
cluster_weights = np.array([1.0])  # mixture weights for demand/dwell clusters
cluster_dwell_params = [
    {"mean_minutes": 8 * 60, "sigma": 0.35},  # lognormal fallback
]
cluster_energy_params = [
    {"mean_kwh": 22.0, "sigma": 0.4},  # lognormal fallback
]
use_energy_targets = True  # switch SOC constraints off and use energy requirements instead
cluster_ids = list(range(len(cluster_weights)))
arrival_histograms = {}
dwell_samples_by_cluster = {}
energy_samples_by_cluster = {}
dwell_energy_pairs_by_cluster = {}
internal_triplets_by_cluster = {}
internal_triplets_by_cluster_and_daytype = {}
external_arrival_minutes = np.array([])
external_dwell_minutes = np.array([])
external_energy_kwh = np.array([])
external_dwell_energy_pairs = np.empty((0, 2))
external_triplets_by_type = {
    "weekday": np.empty((0, 3)),
    "friday": np.empty((0, 3)),
    "weekend": np.empty((0, 3)),
    "all": np.empty((0, 3)),
}
base_Load = np.zeros(SLOTS_PER_DAY)
available_internal_chargers_by_slot = np.zeros(SLOTS_PER_DAY, dtype=int)
simulation_day_start = pd.Timestamp.today().normalize()

def get_active_dwell_cap_minutes():
    if DWELL_SCENARIO == "operational":
        return OPERATIONAL_DWELL_CAP_MINUTES
    return None

def sample_internal_daily_count():
    if BEHAVIOR_SCENARIO in SCENARIO_BASELINE_INTERNAL_COUNTS:
        return int(SCENARIO_BASELINE_INTERNAL_COUNTS[BEHAVIOR_SCENARIO])

    scenario_name = BEHAVIOR_SCENARIO if BEHAVIOR_SCENARIO in INTERNAL_DAILY_COUNT_SCENARIOS else "hospital"
    scenario_dist = INTERNAL_DAILY_COUNT_SCENARIOS[scenario_name]
    counts = np.asarray(scenario_dist["counts"], dtype=int)
    probs = np.asarray(scenario_dist["probs"], dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(counts, p=probs))

# Empirical all-day blocked internal outlets by day type (midnight prebooking proxy).
k_dist_by_type = {
    "weekday": {0: 0.0016, 1: 0.0016, 2: 0.0049, 3: 0.0296, 4: 0.1151, 5: 0.2878,
                6: 0.3059, 7: 0.1661, 8: 0.0691, 9: 0.0148, 10: 0.0033},
    "friday":  {0: 0.0066, 3: 0.0464, 4: 0.1060, 5: 0.2649, 6: 0.3245,
                7: 0.1722, 8: 0.0662, 9: 0.0066, 10: 0.0066},
    "weekend": {2: 0.0033, 3: 0.0298, 4: 0.1192, 5: 0.2682, 6: 0.3344,
                7: 0.1755, 8: 0.0530, 9: 0.0166},
}

def _day_type(day_start):
    wd = pd.Timestamp(day_start).weekday()  # Mon=0 ... Sun=6
    if wd == 4:
        return "friday"
    if wd >= 5:
        return "weekend"
    return "weekday"

def _sample_clipped_normal(n, mean, std, lower, upper):
    return np.clip(np.random.normal(mean, std, n), lower, upper)

def sample_behavior_triplets(n, scenario):
    if scenario == "hospital":
        raise ValueError("Hospital behavior should use the empirical internal triplet pools.")
    if scenario not in BEHAVIOR_SCENARIO_PARAMS:
        raise ValueError(
            f"Unsupported behavior scenario '{scenario}'. "
            f"Choose from {['hospital'] + list(BEHAVIOR_SCENARIO_PARAMS.keys())}."
        )

    p = BEHAVIOR_SCENARIO_PARAMS[scenario]
    arrival_hours = _sample_clipped_normal(
        n, p["arrival_mean_h"], p["arrival_std_h"], p["arrival_min_h"], p["arrival_max_h"]
    )
    dwell_hours = _sample_clipped_normal(
        n, p["dwell_mean_h"], p["dwell_std_h"], p["dwell_min_h"], p["dwell_max_h"]
    )
    energy_kwh = _sample_clipped_normal(
        n, p["energy_mean_kwh"], p["energy_std_kwh"], p["energy_min_kwh"], p["energy_max_kwh"]
    )

    triplets = np.column_stack([
        np.round(arrival_hours * 60.0),
        np.round(dwell_hours * 60.0),
        energy_kwh,
    ]).astype(float)

    max_dwell_min = get_active_dwell_cap_minutes()
    triplets[:, 0] = np.clip(triplets[:, 0], 0, 1439)
    triplets[:, 1] = np.maximum(triplets[:, 1], DWELL_MIN_MINUTES)
    if max_dwell_min is not None:
        triplets[:, 1] = np.minimum(triplets[:, 1], max_dwell_min)
    triplets[:, 2] = np.maximum(triplets[:, 2], 0.1)
    return triplets

def sample_k_blocked(day_start, rng=None):
    rng = rng or np.random.default_rng()
    rules = SCENARIO_RULES.get(BEHAVIOR_SCENARIO, SCENARIO_RULES["hospital"])
    blocked_fraction_range = rules.get("blocked_fraction_range")
    if blocked_fraction_range is not None:
        return None
    dtype = _day_type(day_start)
    dist = k_dist_by_type[dtype]

    k_vals = np.array(list(dist.keys()), dtype=int)
    probs = np.array(list(dist.values()), dtype=float)
    probs = probs / probs.sum()
    return int(rng.choice(k_vals, p=probs))

def build_available_internal_chargers_all_day(day_start, n_internal_chargers, slots_per_day, rng=None):
    """Sample blocked internal outlets once and apply for all slots in the day."""
    rng = rng or np.random.default_rng()
    rules = SCENARIO_RULES.get(BEHAVIOR_SCENARIO, SCENARIO_RULES["hospital"])
    blocked_fraction_range = rules.get("blocked_fraction_range")
    if blocked_fraction_range is not None:
        blocked_fraction = float(rng.uniform(blocked_fraction_range[0], blocked_fraction_range[1]))
        k = int(np.round(n_internal_chargers * blocked_fraction))
    else:
        k = sample_k_blocked(day_start, rng=rng)
    if ALLOW_FULL_INTERNAL_BLOCK:
        k = min(k, n_internal_chargers)
    else:
        # Keep at least one outlet available whenever internal chargers exist.
        k = min(k, max(n_internal_chargers - 1, 0))
    available = np.full(slots_per_day, n_internal_chargers - k, dtype=int)
    available[available < 0] = 0
    return available, k

def resolve_simulation_day_start(day_start):
    ts = pd.Timestamp(day_start).normalize()
    rules = SCENARIO_RULES.get(BEHAVIOR_SCENARIO, SCENARIO_RULES["hospital"])
    if not rules.get("weekday_only", False):
        return ts
    while ts.weekday() >= 5:
        ts = ts + pd.Timedelta(days=1)
    return ts

# --- Load session data (adjust path if needed)
try:
    df = pd.read_csv("session_behaviour_clusters.csv", parse_dates=["arrival_time"])
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")
    df = df.dropna(subset=["arrival_time"])

    # Required columns: arrival_time (timestamp), parking_duration (hours), kWhDelivered (float), cluster (int)
    df["minute_of_day"] = df["arrival_time"].dt.hour * 60 + df["arrival_time"].dt.minute
    df["dwell_minutes"] = df["parking_duration"].astype(float) * 60.0
    df["energy_kwh"] = np.maximum(df["kWhDelivered"].astype(float), 0.1)

    bins = np.arange(0, 1441, SLOT_MINUTES)  # 96 bins of 15 minutes
    arrival_histograms = {}
    dwell_samples_by_cluster = {}
    energy_samples_by_cluster = {}
    dwell_energy_pairs_by_cluster = {}
    internal_triplets_by_cluster = {}
    internal_triplets_by_cluster_and_daytype = {}
    cluster_ids = sorted(df["cluster"].unique())
    cluster_weights = []

    for c in cluster_ids:
        sub = df[df["cluster"] == c]
        cluster_weights.append(len(sub))

        # arrival distribution
        minutes = sub["minute_of_day"].values
        hist, _ = np.histogram(minutes, bins=bins)
        if hist.sum() == 0:
            arr_15 = np.ones(len(bins) - 1, dtype=float)
        else:
            arr_15 = hist.astype(float) / hist.sum()

        arr_1440 = np.repeat(arr_15, 15)
        arr_1440 = arr_1440 / arr_1440.sum()
        arrival_histograms[c] = arr_1440

        # empirical dwell & energy pools
        dwell_samples_by_cluster[c] = sub["dwell_minutes"].values.astype(float)
        energy_samples_by_cluster[c] = sub["energy_kwh"].values.astype(float)
        dwell_energy_pairs_by_cluster[c] = sub[["dwell_minutes", "energy_kwh"]].to_numpy(dtype=float)
        all_triplets = sub[["minute_of_day", "dwell_minutes", "energy_kwh"]].to_numpy(dtype=float)
        internal_triplets_by_cluster[c] = all_triplets

        # Day-type split for internal sessions; keep cluster-based sampling with per-day fallback.
        day_type = sub["arrival_time"].dt.weekday.map(
            lambda wd: "friday" if wd == 4 else ("weekend" if wd >= 5 else "weekday")
        )
        internal_triplets_by_cluster_and_daytype[c] = {"all": all_triplets}
        for dt in ("weekday", "friday", "weekend"):
            dt_sub = sub.loc[day_type == dt, ["minute_of_day", "dwell_minutes", "energy_kwh"]]
            internal_triplets_by_cluster_and_daytype[c][dt] = (
                dt_sub.to_numpy(dtype=float) if not dt_sub.empty else np.empty((0, 3))
            )

    cluster_weights = np.array(cluster_weights, dtype=float)
    cluster_weights = cluster_weights / cluster_weights.sum()
except FileNotFoundError:
    # fall back to synthetic defaults
    pass

# --- Load external sessions (unmanaged-only)
try:
    df_ext = pd.read_csv(
        "external.csv",
        parse_dates=["chargingStart ISO-8601", "chargingStop ISO-8601"],
    )
    if not df_ext.empty:
        df_ext["chargingStart ISO-8601"] = pd.to_datetime(df_ext["chargingStart ISO-8601"], errors="coerce")
        df_ext["chargingStop ISO-8601"] = pd.to_datetime(df_ext["chargingStop ISO-8601"], errors="coerce")

# Drop sessions with invalid dates
        df_ext = df_ext.dropna(subset=["chargingStart ISO-8601", "chargingStop ISO-8601"])
        df_ext["minute_of_day"] = (
            df_ext["chargingStart ISO-8601"].dt.hour * 60
            + df_ext["chargingStart ISO-8601"].dt.minute
        )
        if "chargingDuration[hh:mm:ss]" in df_ext.columns:
            df_ext["dwell_minutes"] = pd.to_timedelta(
                df_ext["chargingDuration[hh:mm:ss]"]
            ).dt.total_seconds() / 60
        else:
            df_ext["dwell_minutes"] = (
                df_ext["chargingStop ISO-8601"] - df_ext["chargingStart ISO-8601"]
            ).dt.total_seconds() / 60
        # Remove noisy/extreme dwell times using the selected dwell scenario.
        min_dwell_min = DWELL_MIN_MINUTES
        max_dwell_min = get_active_dwell_cap_minutes()
        df_ext["dwell_minutes"] = pd.to_numeric(df_ext["dwell_minutes"], errors="coerce")
        df_ext = df_ext.dropna(subset=["dwell_minutes"])
        df_ext = df_ext[df_ext["dwell_minutes"] >= min_dwell_min]
        if max_dwell_min is not None:
            df_ext = df_ext[df_ext["dwell_minutes"] <= max_dwell_min]
        df_ext["energy_kwh"] = pd.to_numeric(df_ext["energyConsumptionInKWh"], errors="coerce")
        df_ext = df_ext.dropna(subset=["energy_kwh"])
        df_ext["energy_kwh"] = np.maximum(df_ext["energy_kwh"], 0.1)

        external_arrival_minutes = df_ext["minute_of_day"].to_numpy(dtype=int)
        external_dwell_minutes = df_ext["dwell_minutes"].to_numpy(float)
        external_energy_kwh = df_ext["energy_kwh"].to_numpy(float)
        external_dwell_energy_pairs = df_ext[["dwell_minutes", "energy_kwh"]].to_numpy(dtype=float)

        ext_day_type = df_ext["chargingStart ISO-8601"].dt.weekday.map(
            lambda wd: "friday" if wd == 4 else ("weekend" if wd >= 5 else "weekday")
        )
        external_triplets_by_type["all"] = df_ext[["minute_of_day", "dwell_minutes", "energy_kwh"]].to_numpy(dtype=float)
        for dt in ("weekday", "friday", "weekend"):
            sub = df_ext.loc[ext_day_type == dt, ["minute_of_day", "dwell_minutes", "energy_kwh"]]
            external_triplets_by_type[dt] = sub.to_numpy(dtype=float) if not sub.empty else np.empty((0, 3))
except FileNotFoundError:
    pass

def _normalize_probs(hist, size):
    hist = np.asarray(hist, dtype=float).flatten()
    if hist.size != size:
        hist = np.resize(hist, size)
    hist[hist < 0] = 0
    if hist.sum() == 0:
        return np.ones(size) / size
    return hist / hist.sum()

def sample_cluster(n, weights=None):
    base_weights = weights if weights is not None else cluster_weights
    probs = _normalize_probs(base_weights, len(base_weights))
    choices = np.array(cluster_ids) if len(cluster_ids) == len(probs) else np.arange(len(probs))
    return np.random.choice(choices, size=n, p=probs)

def sample_arrivals(n, cluster_ids_arr=None):
    out = np.zeros(n, dtype=int)
    clusters = cluster_ids_arr if cluster_ids_arr is not None else np.zeros(n, dtype=int)
    base_probs = _normalize_probs(arrival_histogram, 1440)
    for c in np.unique(clusters):
        mask = clusters == c
        probs = arrival_histograms.get(c, base_probs)
        out[mask] = np.random.choice(np.arange(1440), size=mask.sum(), p=probs)
    return out

def _sample_lognormal(size, mean, sigma, min_value):
    sampled = np.random.lognormal(mean=np.log(max(mean, min_value)), sigma=sigma, size=size)
    return np.maximum(sampled, min_value)

def sample_dwell(n, cluster_ids=None):
    dwell = np.zeros(n)
    clusters = cluster_ids if cluster_ids is not None else np.zeros(n, dtype=int)
    fallback = cluster_dwell_params[0] if cluster_dwell_params else {"mean_minutes": 480, "sigma": 0.35}
    for c in np.unique(clusters):
        mask = clusters == c
        if not np.any(mask):
            continue
        if c in dwell_samples_by_cluster and dwell_samples_by_cluster[c].size > 0:
            pool = dwell_samples_by_cluster[c]
            dwell[mask] = np.random.choice(pool, size=mask.sum(), replace=True)
        else:
            dwell[mask] = _sample_lognormal(mask.sum(), fallback.get("mean_minutes", 480), fallback.get("sigma", 0.35), 1)
    return np.int_(np.round(dwell))

def sample_energy(n, cluster_ids=None):
    energy = np.zeros(n)
    clusters = cluster_ids if cluster_ids is not None else np.zeros(n, dtype=int)
    fallback = cluster_energy_params[0] if cluster_energy_params else {"mean_kwh": 20, "sigma": 0.4}
    for c in np.unique(clusters):
        mask = clusters == c
        if not np.any(mask):
            continue
        if c in energy_samples_by_cluster and energy_samples_by_cluster[c].size > 0:
            pool = energy_samples_by_cluster[c]
            energy[mask] = np.random.choice(pool, size=mask.sum(), replace=True)
        else:
            energy[mask] = _sample_lognormal(mask.sum(), fallback.get("mean_kwh", 20), fallback.get("sigma", 0.4), 0.1)
    return energy

def sample_dwell_energy(n, cluster_ids=None):
    dwell = np.zeros(n)
    energy = np.zeros(n)
    clusters = cluster_ids if cluster_ids is not None else np.zeros(n, dtype=int)
    fallback_dwell = cluster_dwell_params[0] if cluster_dwell_params else {"mean_minutes": 480, "sigma": 0.35}
    fallback_energy = cluster_energy_params[0] if cluster_energy_params else {"mean_kwh": 20, "sigma": 0.4}
    for c in np.unique(clusters):
        mask = clusters == c
        if not np.any(mask):
            continue
        count = mask.sum()
        if c in dwell_energy_pairs_by_cluster and dwell_energy_pairs_by_cluster[c].shape[0] > 0:
            pool = dwell_energy_pairs_by_cluster[c]
            idx = np.random.choice(pool.shape[0], size=count, replace=True)
            sampled = pool[idx]
            dwell[mask] = sampled[:, 0]
            energy[mask] = sampled[:, 1]
        else:
            dwell[mask] = _sample_lognormal(count, fallback_dwell.get("mean_minutes", 480), fallback_dwell.get("sigma", 0.35), 1)
            energy[mask] = _sample_lognormal(count, fallback_energy.get("mean_kwh", 20), fallback_energy.get("sigma", 0.4), 0.1)
    return np.int_(np.round(dwell)), energy

def sample_internal_triplets(n, cluster_ids_arr=None, day_start=None, min_pool=30):
    if BEHAVIOR_SCENARIO != "hospital":
        return sample_behavior_triplets(n, BEHAVIOR_SCENARIO)

    clusters = cluster_ids_arr if cluster_ids_arr is not None else sample_cluster(n)
    triplets = np.zeros((n, 3), dtype=float)  # [arrival_minute, dwell_minutes, energy_kwh]
    max_dwell_min = get_active_dwell_cap_minutes()
    if day_start is None:
        day_start = simulation_day_start
    dt = _day_type(day_start)
    for c in np.unique(clusters):
        mask = clusters == c
        if not np.any(mask):
            continue
        count = mask.sum()
        pool_by_dt = internal_triplets_by_cluster_and_daytype.get(c, {})
        pool = pool_by_dt.get(dt, np.empty((0, 3)))
        if pool.shape[0] < min_pool:
            pool = pool_by_dt.get("all", pool)
        if pool.shape[0] == 0:
            pool = internal_triplets_by_cluster.get(c, np.empty((0, 3)))
        if pool.shape[0] > 0:
            idx = np.random.choice(pool.shape[0], size=count, replace=True)
            sampled = pool[idx]
            sampled[:, 0] = np.clip(sampled[:, 0], 0, 1439)
            sampled[:, 1] = np.maximum(sampled[:, 1], DWELL_MIN_MINUTES)
            if max_dwell_min is not None:
                sampled[:, 1] = np.minimum(sampled[:, 1], max_dwell_min)
            sampled[:, 2] = np.maximum(sampled[:, 2], 0.1)
            triplets[mask] = sampled
        else:
            # Fallback keeps old behavior when a cluster pool is empty.
            arr = sample_arrivals(count)
            dwell, energy = sample_dwell_energy(count)
            triplets[mask, 0] = arr
            triplets[mask, 1] = np.maximum(dwell, DWELL_MIN_MINUTES)
            if max_dwell_min is not None:
                triplets[mask, 1] = np.minimum(triplets[mask, 1], max_dwell_min)
            triplets[mask, 2] = np.maximum(energy, 0.1)
    return triplets

# External-only samplers (used for unmanaged externals)
def sample_external_arrivals(n):
    if external_arrival_minutes.size == 0:
        return sample_arrivals(n)
    return np.random.choice(external_arrival_minutes, size=n, replace=True)

def sample_external_dwell(n):
    if external_dwell_minutes.size == 0:
        return sample_dwell(n)
    return np.random.choice(external_dwell_minutes, size=n, replace=True)

def sample_external_energy(n):
    if external_energy_kwh.size == 0:
        return sample_energy(n)
    return np.random.choice(external_energy_kwh, size=n, replace=True)

def sample_external_dwell_energy(n):
    if external_dwell_energy_pairs.shape[0] == 0:
        dwell = sample_external_dwell(n)
        energy = sample_external_energy(n)
        return dwell, energy
    idx = np.random.choice(external_dwell_energy_pairs.shape[0], size=n, replace=True)
    sampled = external_dwell_energy_pairs[idx]
    return sampled[:, 0], sampled[:, 1]

def sample_external_triplets(day_start, n, min_pool=50):
    dt = _day_type(day_start)
    pool = external_triplets_by_type.get(dt, np.empty((0, 3)))
    if pool.shape[0] < min_pool:
        pool = external_triplets_by_type.get("all", pool)
    if pool.shape[0] == 0:
        arrival = sample_external_arrivals(n).astype(float)
        dwell, energy = sample_external_dwell_energy(n)
        sampled = np.column_stack([arrival, dwell, energy])
        return sampled, dt
    idx = np.random.choice(pool.shape[0], size=n, replace=True)
    return pool[idx], dt

def build_fixed_external_load(day_start, n_sessions, power, n_chargers):
    if n_sessions <= 0 or n_chargers <= 0 or power <= 0:
        return np.zeros(SLOTS_PER_DAY, dtype=float)

    sampled_triplets, _ = sample_external_triplets(day_start, n_sessions)
    sampled_triplets = sampled_triplets[np.argsort(sampled_triplets[:, 0])]
    arrival_time = sampled_triplets[:, 0].astype(int)
    dwell_minutes = sampled_triplets[:, 1]
    energy_required_kwh = sampled_triplets[:, 2]
    energy_minutes = np.maximum(0, (energy_required_kwh * 60) / (power * charge_efficiency))

    starting_time = np.zeros(n_sessions)
    end_time = np.zeros(n_sessions)
    active_chargers = min(n_chargers, n_sessions)
    starting_time[:active_chargers] = arrival_time[:active_chargers]
    end_time[:active_chargers] = np.int_(np.around(starting_time[:active_chargers] + dwell_minutes[:active_chargers]))

    q = PriorityQueue()
    for i in range(active_chargers):
        q.put(end_time[i])
    for i in range(active_chargers, len(arrival_time)):
        non_available = [j for j in q.queue if j > arrival_time[i]]
        if len(non_available) == active_chargers:
            starting_time[i] = np.int_(min(non_available)) + 1
            q.get()
        else:
            starting_time[i] = np.int_(arrival_time[i])
            q.get()
        end_time[i] = np.int_(np.around(starting_time[i] + dwell_minutes[i]))
        q.put(end_time[i])

    charge_end = np.int_(np.around(np.minimum(starting_time + dwell_minutes, starting_time + energy_minutes)))
    starting_time = [int(i) for i in time24(starting_time)]
    charge_end = [int(i) for i in time24(charge_end)]

    fixed_load_min = np.zeros(Maxtime_Interval, dtype=float)
    for i in range(n_sessions):
        if charge_end[i] == -1 or starting_time[i] == -1:
            continue
        if charge_end[i] > starting_time[i]:
            iterator_time = np.arange(starting_time[i] - 1, charge_end[i], 1)
            for t in iterator_time:
                fixed_load_min[t] += power
        else:
            iterator_time = np.arange(starting_time[i] - 1, Maxtime_Interval, 1)
            for t in iterator_time:
                fixed_load_min[t] += power
            iterator_time = np.arange(0, charge_end[i], 1)
            for t in iterator_time:
                fixed_load_min[t] += power

    fixed_load_slots = np.zeros(SLOTS_PER_DAY, dtype=float)
    for t in range(SLOTS_PER_DAY):
        fixed_load_slots[t] = np.average(fixed_load_min[t * SLOT_MINUTES:(t + 1) * SLOT_MINUTES])
    return fixed_load_slots

v1g_energy_required_kwh = None
v1g_dwell_slots = None

def user_definedProb(Charge_Freq,values):
    totalFreq = sum(Charge_Freq)
    Charge_Prob = [x / totalFreq for x in Charge_Freq]
    Charge_Prob_new = np.zeros(1440)
    values_interval = np.int_([x * 60 for x in values])
    j = 0
    for j in range(24): #changed 16 into 24
        if j < 23: #changed 15 into 23
            index_bottom = values_interval[j]
            index_upper = values_interval[j + 1]
            Charge_Prob_new[index_bottom:index_upper] = Charge_Prob[j]
        if j == 23: #changed 16 into 23
            index_bottom = values_interval[0]
            Charge_Prob_new[0:index_bottom] = Charge_Prob[j]
            index_upper = values_interval[j]
            if index_upper < 1440:
                Charge_Prob_new[index_upper:1440] = Charge_Prob[j]
    totalFreq = sum(Charge_Prob_new)
    Charge_Prob_new2 = [x / totalFreq for x in Charge_Prob_new]
    return Charge_Prob_new2

def getArrivalTime(nVech): # starting time is defined by users as an input
    if useCase ==1: #residential
        ###default value and user defined input, e.g., (20,2) to (19, 2.5)
        arrivalTime = np.random.normal(20, 1, nVech)
        return arrivalTime
    if useCase==2:  #office
        ####default value
        arrivalTime = np.random.lognormal(2.32274, 0.301833, nVech)
        return arrivalTime
        #### user defined input: e.g., (9,2) to (8.5, 1.5)
        arrivalTime = np.random.normal(9, 2, nVech)
        return arrivalTime
    if useCase==3: #public
        ####fixed, don't allow for changes
        Arrival_Freq = [407.1, 439.5, 502.7, 557.2, 664.4, 874.9, 1109.4, 1773.9,
                       1974.6, 2073.7, 2080.1, 2098.3, 2110.6, 2113.4, 2116.8, 2110.1,
                       2104.5, 2086.1, 2082.2, 2079.9, 2055.3, 1944.7, 1606.2, 367.1]
        values = [1, 2, 3, 4,  5, 6, 7, 8,
                  9, 10, 11, 12, 13, 14, 15, 16,
                  17, 18, 19, 20, 21, 22, 23,24]
        values_new = range(1440)
        Charge_Prob_new2 = user_definedProb(Arrival_Freq, values)
        arrivalTime = np.random.choice(values_new, nVech, p=list(Charge_Prob_new2))
        return arrivalTime / 60
    if useCase==4: #user defined
        #user defined: option 1-- 24 elements of Arrival Freq
        Arrival_Freq = [507, 422, 401, 420, 508,634, 898, 1390, 1913, 2187,2363,2363,
                        2139, 1939, 1821, 1621, 1521, 1335, 1311, 603,600,563, 503,503]
        #Arrival_Freq = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                0, 0, 0, 0, 0, 0, 24, 12,7,1, 1,1]
        values=range(24)
        values_new = range(1440)
        Charge_Prob_new2=user_definedProb(Arrival_Freq,values)
        #arrivalTime = np.random.choice(values_new, nVech, p=list(Charge_Prob_new2))
        #return arrivalTime / 60

        #user defined: option 2--e.g, (18,1.5) to (17, 2)
        arrivalTime=np.random.normal(18,2, nVech)
        return arrivalTime


def getStartingSOC(isSlow, n_Veh, energy_required_kwh=None, battery_capacity_kwh=None): # SOC is defined by charging frequencies
    if (energy_required_kwh is None) != (battery_capacity_kwh is None):
        raise ValueError("Provide both energy_required_kwh and battery_capacity_kwh, or neither.")

    # If energy demand is provided, derive arrival SOC directly:
    # SOC_start = 1 - (energy_required / battery_capacity)
    if energy_required_kwh is not None and battery_capacity_kwh is not None:
        energy_required = np.asarray(energy_required_kwh, dtype=float).reshape(-1)
        if energy_required.size != n_Veh:
            raise ValueError(
                f"n_Veh ({n_Veh}) must match len(energy_required_kwh) ({energy_required.size})."
            )

        battery_cap = np.asarray(battery_capacity_kwh, dtype=float)
        if battery_cap.size == 1:
            battery_cap = np.full(energy_required.size, float(battery_cap))
        else:
            battery_cap = battery_cap.reshape(-1)
            if battery_cap.size != energy_required.size:
                raise ValueError(
                    "battery_capacity_kwh must be scalar or same length as energy_required_kwh."
                )
        battery_cap = np.maximum(battery_cap, 1e-9)
        startingSOC = 1.0 - (energy_required / battery_cap)
        return np.clip(startingSOC, 0.0, 1.0)

    # Keep useCase 4 reachable regardless of slow/fast mode.
    if useCase==4: #user defined starting SOC: SOC is a value between 0 and 1. e.g., (0.6, 0.2) to (0.2, 0.3)
        startingSOC=np.random.normal(0.4,0.2, n_Veh)
        return np.clip(startingSOC, 0.0, 1.0)

    if isSlow == 1: # slow charge: one charge per day
        dist_list = np.random.lognormal(3.67017251945698, 0.532230403897875, n_Veh)
        startingSOC = 1 - dist_list / batteryRange
        return np.clip(startingSOC, 0.0, 1.0)

    if isSlow > 1: # fast charge: multiple days per charge
        if useCase==1 or useCase==2: # resiential & office: 'mileages traveled' approach
            dist_list = np.random.lognormal(3.67017251945698, 0.532230403897875, n_Veh)
            for i in range(n_Veh):
                dist_list[i] = dist_list[i] * Charge_frequency
                if dist_list[i] > batteryRange * 0.8:  #cannot be below 20%SOC
                    while True:
                        dist_list[i] = np.random.lognormal(3.67017251945698, 0.532230403897875, 1)
                        dist_list[i] = dist_list[i] * Charge_frequency
                        if dist_list[i] <= batteryRange * 0.8:
                            break
            startingSOC = 1 - dist_list / batteryRange
            return np.clip(startingSOC, 0.0, 1.0)
        if useCase==3: # commercial & public: 'predefined starting SOC':
            startingSOC = np.random.beta(1.37722, 3.42153, n_Veh)
            return np.clip(startingSOC, 0.0, 1.0)

    # Fallback if no branch matched.
    return np.full(n_Veh, 0.5, dtype=float)

def time24(time):
    condlist = [time< 2160, time >=2160]
    choicelist= [time, -1]
    time=np.select(condlist, choicelist)

    condlist = [time < 1440, time >= 1440]
    choicelist = [time, time - 1440]
    time = np.select(condlist, choicelist)
    return time


def Unmanaged_charge(n_Veh, power, nChargers):
    if n_Veh <= 0 or nChargers <= 0 or power <= 0:
        zero_profile = np.zeros(Maxtime_Interval)
        zero_slots = np.zeros(SLOTS_PER_DAY)
        return zero_profile, zero_profile, zero_profile, 0.0, 0.0, 0.0

    iterator=range(n_iter)
    Load_maxtrix = np.zeros((n_iter, Maxtime_Interval))
    wait_count_list = np.zeros(n_iter)
    uncharged_count_list = np.zeros(n_iter)

    for it in iterator:
        eLoad=np.zeros(Maxtime_Interval)

        clusters = sample_cluster(n_Veh)

        # Use external triplets for unmanaged fleet (arrival, dwell, kWh) from same sampled rows.
        sampled_triplets, _ = sample_external_triplets(simulation_day_start, n_Veh)
        sampled_triplets = sampled_triplets[np.argsort(sampled_triplets[:, 0])]
        arrivalTime = sampled_triplets[:, 0].astype(int)
        dwell_minutes = sampled_triplets[:, 1]
        energy_required_kwh = sampled_triplets[:, 2]
        if it == 0:
            print("iter0 arrival sample:", arrivalTime[:10])
            print("iter0 dwell sample (min):", np.round(dwell_minutes[:10], 1))
            print("iter0 energy sample (kWh):", np.round(energy_required_kwh[:10], 2))
        energy_minutes = np.maximum(0, (energy_required_kwh * 60) / (power * charge_efficiency))

        # calculate the charging starting time based on the queuing size, that is, limited number of chargers
        startingTime = np.zeros(n_Veh)
        endTime = np.zeros(n_Veh)
        startingTime[:nChargers] = arrivalTime[:nChargers]
        endTime[:nChargers] = np.int_(
            np.around(startingTime[:nChargers] + dwell_minutes[:nChargers]))

        q = PriorityQueue()
        for i in range(0, nChargers):  # initiate the queue
            q.put(endTime[i])
        for i in range(nChargers, len(arrivalTime)):
            non_available = [j for j in q.queue if j > arrivalTime[i]]
            # print('que status', q.queue)
            if len(non_available) == nChargers:
                startingTime[i] = np.int_(min(non_available)) + 1
                q.get()
            else:
                startingTime[i] = np.int_(arrivalTime[i])
                q.get()
            endTime[i] = np.int_(np.around(startingTime[i] + dwell_minutes[i]))
            q.put(endTime[i])

        waitTime = startingTime - arrivalTime
        charge_end = np.int_(np.around(np.minimum(startingTime + dwell_minutes, startingTime + energy_minutes)))
        startingTime = time24(startingTime)
        endTime = time24(endTime)
        charge_end = time24(charge_end)
        startingTime = [int(i) for i in startingTime]
        endTime = [int(i) for i in endTime]
        charge_end = [int(i) for i in charge_end]
        for i in range(0, len(waitTime)):
            if endTime[i] == -1:
                waitTime[i] = 0
        avgwait = np.mean(waitTime)
        unChargedveh = endTime.count(-1)
        waited_veh = int(np.sum(waitTime > 0))
        wait_count_list[it] = waited_veh
        uncharged_count_list[it] = unChargedveh
        # print('arrival time', arrivalTime)
        # print('starting time', startingTime)
        # print('end time', endTime)
        # print('wait time', waitTime)
        # print('average wait time for slow chargers is %5.2f hours' %avgwaitSlow)
        # print('%4d vehicles uncharged due to lack of slow chargers' %unChargedveh_slow)

        for c in range(n_Veh):
            if charge_end[c] ==-1 or startingTime[c] ==-1:
                break
            if charge_end[c]>startingTime[c]:
                iterator_time=np.arange(startingTime[c]-1,charge_end[c],1)
                for t in iterator_time:
                    eLoad[t]= eLoad[t]+power
            else:
                iterator_time=np.arange(startingTime[c]-1, Maxtime_Interval,1)
                for t in iterator_time:
                    eLoad[t]=eLoad[t]+power
                iterator_time=np.arange(0,charge_end[c],1)
                for t in iterator_time:
                    eLoad[t]=eLoad[t]+power
        Load_maxtrix[it]=eLoad
        if it == 0:
            print("iter0 eLoad min/max:", float(np.min(eLoad)), float(np.max(eLoad)))
            print("iter0 eLoad first 60 min:", eLoad[:60])

    Load_avg=Load_maxtrix.mean(0)
    Load_var=Load_maxtrix.std(0)
    Load_upper=Load_avg+Load_var*2
    Load_bottom=np.maximum(Load_avg-Load_var*2, 0)
    print("Load_avg min/max:", float(np.min(Load_avg)), float(np.max(Load_avg)))
    print("Load_avg std across day:", float(np.std(Load_avg)))
    avg_waited_veh = float(np.mean(wait_count_list))
    avg_uncharged_veh = float(np.mean(uncharged_count_list))

    return Load_avg, Load_bottom, Load_upper, avgwait, avg_waited_veh, avg_uncharged_veh

def Unmanaged_usecase(useCase):
    avgwaitSlow = -1
    avgwaitFast = -1
    avgwaitGen = -1
    avgWaitedSlow = -1
    avgWaitedFast = -1
    avgWaitedGen = -1
    avgUnchargedSlow = -1
    avgUnchargedFast = -1
    avgUnchargedGen = -1

    if useCase==1:
        Load_avg, Load_bottom, Load_upper, avgwait, avgwaited, avguncharged = Unmanaged_charge(n_slowVeh, power_Slow, nslow_Chargers)
        avgwaitSlow=avgwait
        avgWaitedSlow = avgwaited
        avgUnchargedSlow = avguncharged
        iterator_time=range(SLOTS_PER_DAY)
        for t in iterator_time:
            slow_Load_1h_avg[t]=np.average(Load_avg[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])
        slow_Load_1h_upper=np.zeros(SLOTS_PER_DAY)
        for t in iterator_time:
            slow_Load_1h_upper[t]=np.average(Load_upper[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])
        slow_Load_1h_bottom=np.zeros(SLOTS_PER_DAY)
        for t in iterator_time:
            slow_Load_1h_bottom[t]=np.average(Load_bottom[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])

        Load_avg, Load_bottom, Load_upper, avgwait, avgwaited, avguncharged = Unmanaged_charge(n_fastVeh, power_Fast, nfast_Chargers)
        avgwaitFast = avgwait
        avgWaitedFast = avgwaited
        avgUnchargedFast = avguncharged
        iterator_time=range(SLOTS_PER_DAY)
        for t in iterator_time:
            fast_Load_1h_avg[t]=np.average(Load_avg[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])
        fast_Load_1h_upper=np.zeros(SLOTS_PER_DAY)
        for t in iterator_time:
            fast_Load_1h_upper[t]=np.average(Load_upper[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])
        fast_Load_1h_bottom=np.zeros(SLOTS_PER_DAY)
        for t in iterator_time:
            fast_Load_1h_bottom[t]=np.average(Load_bottom[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])

    if useCase==2 or useCase==3:
        Load_avg, Load_bottom, Load_upper, avgwait, avgwaited, avguncharged = Unmanaged_charge(n_fastVeh, power_Fast, nfast_Chargers)
        avgwaitFast = avgwait
        avgWaitedFast = avgwaited
        avgUnchargedFast = avguncharged
        iterator_time=range(SLOTS_PER_DAY)
        for t in iterator_time:
            fast_Load_1h_avg[t]=np.average(Load_avg[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])
        fast_Load_1h_upper=np.zeros(SLOTS_PER_DAY)
        for t in iterator_time:
            fast_Load_1h_upper[t]=np.average(Load_upper[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])
        fast_Load_1h_bottom=np.zeros(SLOTS_PER_DAY)
        for t in iterator_time:
            fast_Load_1h_bottom[t]=np.average(Load_bottom[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])

    if useCase==4:
        Load_avg, Load_bottom, Load_upper, avgwait, avgwaited, avguncharged = Unmanaged_charge(n_V0G_Veh, power_General, ngen_Chargers) #please change into n_fastVeh
        avgwaitGen = avgwait
        avgWaitedGen = avgwaited
        avgUnchargedGen = avguncharged
        iterator_time=range(SLOTS_PER_DAY)
        for t in iterator_time:
            gen_Load_1h_avg[t]=np.average(Load_avg[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])
        gen_Load_1h_upper=np.zeros(SLOTS_PER_DAY)
        for t in iterator_time:
            gen_Load_1h_upper[t]=np.average(Load_upper[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])
        gen_Load_1h_bottom=np.zeros(SLOTS_PER_DAY)
        for t in iterator_time:
            gen_Load_1h_bottom[t]=np.average(Load_bottom[t*SLOT_MINUTES:(t+1)*SLOT_MINUTES])

    return (
        avgwaitSlow, avgwaitFast, avgwaitGen,
        avgWaitedSlow, avgWaitedFast, avgWaitedGen,
        avgUnchargedSlow, avgUnchargedFast, avgUnchargedGen
    )

def initiate_V1G(n_V1G_Veh): # initiate the initial state (including SOC, parking duration) for managed charging vehicles
    global v1g_energy_required_kwh, v1g_dwell_slots
    clusters = sample_cluster(n_V1G_Veh)
    sampled_triplets = sample_internal_triplets(
        n_V1G_Veh,
        clusters,
        day_start=simulation_day_start,
    )
    sampled_triplets = sampled_triplets[np.argsort(sampled_triplets[:, 0])]
    arrival_minutes = sampled_triplets[:, 0].astype(int)
    dwell_minutes = sampled_triplets[:, 1]
    v1g_energy_required_kwh = sampled_triplets[:, 2]

    arrival_slots = np.int_(np.floor(arrival_minutes / SLOT_MINUTES))
    dwell_slots = np.int_(np.round(dwell_minutes / SLOT_MINUTES))
    v1g_dwell_slots = dwell_slots.copy()
    v1g_startingTime = arrival_slots % SLOTS_PER_DAY
    v1g_endingTime = (arrival_slots + dwell_slots) % SLOTS_PER_DAY

    # process startingSOC and travel distance for each V1G vehicle
   # v1g_startingSOC = getStartingSOC(2, n_V1G_Veh)
    v1g_startingSOC = getStartingSOC(
        isSlow,
        n_V1G_Veh,
        energy_required_kwh=v1g_energy_required_kwh,
        battery_capacity_kwh=batteryCapacity,
    ) ## derive SOC from required energy when available

    totalsoc=sum(v1g_startingSOC)
    print('total v1g soc:', totalsoc)
    #v1g_distance = np.random.lognormal(3.67017251945698, 0.532230403897875, n_V1G_Veh)
    #v1g_startingSOC = sorted(v1g_startingSOC)
    #v1g_distance = sorted(v1g_distance)
    #for i in range(n_V1G_Veh):
    #    startSOC_temp = 1 - v1g_distance[i] / batteryRange
    #    if startSOC_temp < v1g_startingSOC[i]:
    #        v1g_startingSOC[i] = startSOC_temp

    return v1g_startingTime, v1g_endingTime, v1g_startingSOC

def diagnose_v1g_samples(start_slots, end_slots, start_soc, energy_required_kwh, battery_capacity_kwh, dwell_slots=None):
    start_slots = np.asarray(start_slots, dtype=int)
    end_slots = np.asarray(end_slots, dtype=int)
    start_soc = np.asarray(start_soc, dtype=float)
    energy_required_kwh = np.asarray(energy_required_kwh, dtype=float)
    battery_capacity_kwh = float(battery_capacity_kwh)

    if start_slots.size == 0:
        print("V1G diagnostics: no managed EVs to diagnose.")
        return

    if dwell_slots is not None:
        window_slots = np.asarray(dwell_slots, dtype=int).reshape(-1)
    else:
        window_slots = np.where(
            end_slots > start_slots,
            end_slots - start_slots,
            np.where(end_slots < start_slots, (SLOTS_PER_DAY - start_slots) + end_slots, 0),
        )
    derived_soc = np.clip(1.0 - (energy_required_kwh / max(battery_capacity_kwh, 1e-9)), 0.0, 1.0)
    soc_mismatch = np.abs(start_soc - derived_soc) > 1e-6
    near_zero_soc = start_soc <= 0.02
    over_capacity_energy = energy_required_kwh > battery_capacity_kwh + 1e-9
    zero_window = window_slots == 0
    full_day_window = window_slots >= SLOTS_PER_DAY

    print(
        "V1G diagnostics:",
        f"N={start_soc.size}, SOC[min/mean/max]=({start_soc.min():.3f}/{start_soc.mean():.3f}/{start_soc.max():.3f}),",
        f"energy[min/mean/max]=({energy_required_kwh.min():.2f}/{energy_required_kwh.mean():.2f}/{energy_required_kwh.max():.2f}) kWh"
    )

    if np.any(near_zero_soc):
        idx = np.where(near_zero_soc)[0][:10]
        print(f"WARNING: near-zero arrival SOC EVs: count={near_zero_soc.sum()}, sample_idx={idx.tolist()}")
    if np.any(over_capacity_energy):
        idx = np.where(over_capacity_energy)[0][:10]
        print(f"WARNING: energy_required > battery capacity: count={over_capacity_energy.sum()}, sample_idx={idx.tolist()}")
    if np.any(zero_window):
        idx = np.where(zero_window)[0][:10]
        print(f"WARNING: zero-window EVs (dwell_slots==0): count={zero_window.sum()}, sample_idx={idx.tolist()}")
    if np.any(full_day_window):
        idx = np.where(full_day_window)[0][:10]
        print(f"INFO: full-day window EVs (dwell_slots>=SLOTS_PER_DAY): count={full_day_window.sum()}, sample_idx={idx.tolist()}")
    if np.any(soc_mismatch):
        idx = np.where(soc_mismatch)[0][:10]
        print(f"WARNING: SOC mismatch vs derived energy formula: count={soc_mismatch.sum()}, sample_idx={idx.tolist()}")

def V1G_optimization(objective_mode="spread", alpha=0.05): #this is the function for managed charging

    # initiate the problem statement
    model=lp.LpProblem('Minimize_Distribution_level_Peak_Valley_Difference', lp.LpMinimize)
    LEXI_M = 1e4  # dominant penalty to approximate lexicographic service-first objective

    if objective_mode == "D_service_first_peak" and (not use_energy_targets or v1g_energy_required_kwh is None):
        raise ValueError("D_service_first_peak requires use_energy_targets=True and valid v1g_energy_required_kwh.")

    #define optimization variables
    veh_V1G=range(n_V1G_Veh)
    time_Interval=range(SLOTS_PER_DAY)

    chargeprofiles=lp.LpVariable.dicts('charging_profiles', ((i,j) for i in veh_V1G for j in time_Interval), lowBound=0.0, upBound=power_V1GUppper)
    chargestates=lp.LpVariable.dicts('charging_states', ((i,j) for i in veh_V1G for j in time_Interval), cat='Binary')
    total_load=lp.LpVariable.dicts('total_load', time_Interval,lowBound=0.0)
    total_ev_load=lp.LpVariable.dicts('total_ev_load', time_Interval, lowBound=0.0)
    max_load=lp.LpVariable('max_load', lowBound=0)
    min_load=lp.LpVariable('min_load', lowBound=0)
    peak_ev_kw = lp.LpVariable('peak_ev_kw', lowBound=0.0)
    unmet_energy = lp.LpVariable.dicts('unmet_energy', veh_V1G, lowBound=0.0)

    def build_window_slots(temp_start, temp_end, dwell_slots_i):
        if dwell_slots_i <= 0:
            return []
        if dwell_slots_i >= SLOTS_PER_DAY:
            return list(range(SLOTS_PER_DAY))
        if temp_start < temp_end:
            return list(range(temp_start, temp_end))
        return list(range(temp_start, SLOTS_PER_DAY)) + list(range(0, temp_end))

    # Feasibility diagnostics (informational, does not change model)
    if use_energy_targets and v1g_energy_required_kwh is not None:
        infeasible_evs = []
        for i in veh_V1G:
            temp_start = v1g_startingTime[i]
            temp_end = v1g_endingTime[i]
            dwell_slots_i = int(v1g_dwell_slots[i]) if v1g_dwell_slots is not None else (
                (SLOTS_PER_DAY - temp_start) + temp_end if temp_start >= temp_end else temp_end - temp_start
            )
            n_slots = len(build_window_slots(temp_start, temp_end, dwell_slots_i))
            emax = n_slots * power_V1GUppper * charge_efficiency * SLOT_HOURS
            if v1g_energy_required_kwh[i] > emax + 1e-9:
                infeasible_evs.append((i, v1g_energy_required_kwh[i], emax, n_slots))
        if infeasible_evs:
            print("WARNING: Per-EV window infeasibility detected:")
            for ev_id, req_kwh, max_kwh, n_slots in infeasible_evs[:10]:
                print(f"  EV {ev_id}: required={req_kwh:.2f} kWh, max={max_kwh:.2f} kWh, window_slots={n_slots}")
            if len(infeasible_evs) > 10:
                print(f"  ... and {len(infeasible_evs) - 10} more EVs")

        e_total_req = float(np.sum(v1g_energy_required_kwh))
        e_total_max = float(np.sum(available_internal_chargers_by_slot)) * power_V1GUppper * charge_efficiency * SLOT_HOURS
        if e_total_req > e_total_max + 1e-9:
            print("WARNING: System-level energy infeasibility detected:")
            print(f"  Total required={e_total_req:.2f} kWh > total max deliverable={e_total_max:.2f} kWh")

    # define constraints
    for t in time_Interval: # constraint 1 & 2: to identify the max and min loads
        model += total_load[t] <= max_load
        model += total_load[t] >= min_load

    for t in time_Interval: # constraint 3: calculate the total load at each time interval t
        if opt_includeSolar == 0:
            model += lp.lpSum( [chargeprofiles[i,t]] for i in veh_V1G) + base_Load[t] + unmanaged_Load[t] == total_load[t]
        else:
            model += lp.lpSum([chargeprofiles[i, t]] for i in veh_V1G) + base_Load[t] + unmanaged_Load[t] - solarCurve[t] == total_load[t]

    # constraint 3a: managed EV load aggregation and managed EV peak tracker
    for t in time_Interval:
        model += total_ev_load[t] == lp.lpSum(chargeprofiles[i, t] for i in veh_V1G)
        model += total_ev_load[t] <= peak_ev_kw

    # constraint 3aa: hard site headroom limit for managed EV load
    SITE_CAP_KW = 2000.0
    for t in time_Interval:
        if opt_includeSolar == 0:
            headroom_t = max(0.0, SITE_CAP_KW - base_Load[t])
        else:
            headroom_t = max(0.0, SITE_CAP_KW - (base_Load[t] - solarCurve[t]))
        model += total_ev_load[t] <= headroom_t

    # constraint 3b: site charger availability by slot
    for t in time_Interval:
        model += lp.lpSum([chargestates[i, t] for i in veh_V1G]) <= available_internal_chargers_by_slot[t]

    for i in veh_V1G: # constraint 4: constraint on charging powers for each EV: only optimize the charge profile between start and end charging time
        temp_start=v1g_startingTime[i]
        temp_end=v1g_endingTime[i]
        dwell_slots_i = int(v1g_dwell_slots[i]) if v1g_dwell_slots is not None else (
            (SLOTS_PER_DAY - temp_start) + temp_end if temp_start >= temp_end else temp_end - temp_start
        )
        window = build_window_slots(temp_start, temp_end, dwell_slots_i)
        window_set = set(window)
        for t in time_Interval:
            if t in window_set:
                model += chargeprofiles[i,t] <= chargestates[i,t] * power_V1GUppper
                model += chargeprofiles[i,t] >= chargestates[i,t] * power_V1GLower
            else:
                model += chargeprofiles[i,t] == 0
                model += chargestates[i,t] ==0

    for i in veh_V1G: # constraint 5: energy target (or SOC fallback)
        temp_start=v1g_startingTime[i]
        temp_end=v1g_endingTime[i]
        dwell_slots_i = int(v1g_dwell_slots[i]) if v1g_dwell_slots is not None else (
            (SLOTS_PER_DAY - temp_start) + temp_end if temp_start >= temp_end else temp_end - temp_start
        )
        window = build_window_slots(temp_start, temp_end, dwell_slots_i)
        if use_energy_targets and v1g_energy_required_kwh is not None:
            if len(window) == 0:
                if v1g_energy_required_kwh[i] > 1e-9:
                    print(
                        f"WARNING: EV {i} has empty charging window but positive required energy "
                        f"({v1g_energy_required_kwh[i]:.2f} kWh)."
                    )
                model += unmet_energy[i] == v1g_energy_required_kwh[i]
            else:
                delivered_i = lp.lpSum(chargeprofiles[i, tn] * charge_efficiency * SLOT_HOURS for tn in window)
                model += delivered_i + unmet_energy[i] == v1g_energy_required_kwh[i]
        else:
            temp_startSOC=v1g_startingSOC[i]
            if temp_start >= temp_end:
                for t in range(temp_start+1, SLOTS_PER_DAY, 1):
                    temp_timer = range (temp_start, t, 1)
                    model += temp_startSOC + lp.lpSum( [chargeprofiles[i,tn] *charge_efficiency* SLOT_HOURS/batteryCapacity] for tn in temp_timer) <=1
                for t in range (0, temp_end+1, 1):
                    temp_timer = range (0, t, 1)
                    model += temp_startSOC + lp.lpSum( [chargeprofiles[i,tn] *charge_efficiency* SLOT_HOURS/batteryCapacity] for tn in range(temp_start, SLOTS_PER_DAY,1)) \
                             + lp.lpSum( [chargeprofiles[i,tn] *charge_efficiency* SLOT_HOURS/batteryCapacity] for tn in temp_timer) <=1
                if end_SOC ==2:
                    model += temp_startSOC + lp.lpSum([chargeprofiles[i, tn] *charge_efficiency * SLOT_HOURS/ batteryCapacity] for tn in range(temp_start, SLOTS_PER_DAY, 1)) + lp.lpSum([chargeprofiles[i, tn] *charge_efficiency * SLOT_HOURS/ batteryCapacity] for tn in range(0, temp_end)) ==1
            if temp_start < temp_end:
                 for t in range (temp_start+1, temp_end+1, 1):
                    temp_timer = range (temp_start, t, 1)
                    model += temp_startSOC + lp.lpSum( [chargeprofiles[i,tn] *charge_efficiency* SLOT_HOURS/batteryCapacity] for tn in temp_timer) <=1
                 if end_SOC ==2:
                    model += temp_startSOC + lp.lpSum([chargeprofiles[i, tn] * charge_efficiency * SLOT_HOURS/ batteryCapacity] for tn in range(temp_start, temp_end, 1)) ==1

                #for t in time_Interval:  # constraint 6: number of chargers available at time interval t
        #model += lp.lpSum([chargestates[i, t]] for i in veh_V1G) <= 400

    unmet_penalty_term = UNMET_ENERGY_PENALTY_PER_KWH * lp.lpSum(unmet_energy[i] for i in veh_V1G)

    # define objective function (choose one mode)
    if objective_mode == "A_peak":
        model += max_load + unmet_penalty_term, "obj_peak"
    elif objective_mode == "B_peak_above_baseline":
        baseline = [
            base_Load[t] + unmanaged_Load[t] - (solarCurve[t] if opt_includeSolar else 0)
            for t in time_Interval
        ]
        peak_increase = lp.LpVariable('peak_increase', lowBound=0)
        for t in time_Interval:
            model += peak_increase >= total_load[t] - baseline[t]
        model += peak_increase + unmet_penalty_term, "obj_peak_increase"
    elif objective_mode == "C_spread_plus_peak":
        model += (max_load - min_load) + alpha * max_load + unmet_penalty_term, "obj_spread_plus_peak"
    elif objective_mode == "D_service_first_peak":
        model += LEXI_M * lp.lpSum(unmet_energy[i] for i in veh_V1G) + peak_ev_kw, "obj_service_first_peak"
    else:  # default: minimize spread
        model += max_load - min_load + unmet_penalty_term, "obj_spread"

    #print(model)
    status=model.solve()

    results = {
        "status": lp.LpStatus[status],
        "max_load": lp.value(max_load),
        "min_load": lp.value(min_load),
        "objective": lp.value(model.objective),
        "objective_name": getattr(model.objective, "name", "n/a"),
    }
    if objective_mode == "B_peak_above_baseline":
        results["peak_increase"] = lp.value(peak_increase)
    unmet_total_kwh = float(sum((lp.value(unmet_energy[i]) or 0.0) for i in veh_V1G))
    peak_ev_val = float(lp.value(peak_ev_kw) or 0.0)
    required_total_kwh = float(np.sum(v1g_energy_required_kwh)) if v1g_energy_required_kwh is not None else np.nan
    served_total_kwh = required_total_kwh - unmet_total_kwh if not np.isnan(required_total_kwh) else np.nan
    served_fraction = (served_total_kwh / required_total_kwh) if (not np.isnan(required_total_kwh) and required_total_kwh > 0) else np.nan
    results["peak_ev_kw"] = peak_ev_val
    results["unmet_total_kwh"] = unmet_total_kwh
    results["required_total_kwh"] = required_total_kwh
    results["served_total_kwh"] = served_total_kwh
    results["served_fraction"] = served_fraction

    return chargeprofiles, total_load, results


def simulate_unmanaged_internal():
    """Charge the frozen internal EV sessions immediately on arrival without optimization.

    The same internal sessions used by the managed objectives are scheduled in a
    first-come-first-served manner, subject only to slot charger availability.
    This represents true plug-and-charge behavior and does not respond to the
    hospital load or site headroom constraint.
    """
    internal_load = np.zeros(SLOTS_PER_DAY, dtype=float)
    required = np.array(v1g_energy_required_kwh, dtype=float).copy()
    remaining = required.copy()
    arrival = np.array(v1g_startingTime, dtype=int)
    dwell_slots = np.array(v1g_dwell_slots, dtype=int)

    active_by_slot = [[] for _ in range(SLOTS_PER_DAY)]
    for ev_id in range(len(required)):
        start = int(arrival[ev_id])
        slots_i = int(dwell_slots[ev_id])
        if slots_i <= 0:
            continue
        window = []
        if slots_i >= SLOTS_PER_DAY:
            window = list(range(SLOTS_PER_DAY))
        else:
            end = (start + slots_i) % SLOTS_PER_DAY
            if start < end:
                window = list(range(start, end))
            else:
                window = list(range(start, SLOTS_PER_DAY)) + list(range(0, end))
        for t in window:
            active_by_slot[t].append(ev_id)

    # Earlier-arriving EVs keep priority when more EVs are present than chargers.
    order_key = np.lexsort((np.arange(len(required)), arrival))
    priority = {int(ev_id): rank for rank, ev_id in enumerate(order_key)}

    for t in range(SLOTS_PER_DAY):
        chargers_left = int(available_internal_chargers_by_slot[t])
        if chargers_left <= 0:
            continue

        active = [ev_id for ev_id in active_by_slot[t] if remaining[ev_id] > 1e-9]
        active.sort(key=lambda ev_id: priority[int(ev_id)])
        for ev_id in active[:chargers_left]:
            power_t = min(power_V1GUppper, remaining[ev_id] / max(charge_efficiency * SLOT_HOURS, 1e-9))
            if power_t <= 1e-9:
                continue
            delivered = power_t * charge_efficiency * SLOT_HOURS
            internal_load[t] += power_t
            remaining[ev_id] = max(0.0, remaining[ev_id] - delivered)

    total_unmet_kwh = float(np.sum(remaining))
    required_total_kwh = float(np.sum(required))
    served_total_kwh = required_total_kwh - total_unmet_kwh
    served_fraction = served_total_kwh / required_total_kwh if required_total_kwh > 0 else np.nan
    total_site_load = base_Load + unmanaged_Load + internal_load - (solarCurve if opt_includeSolar else 0)

    return internal_load, total_site_load, {
        "status": "Simulated",
        "required_total_kwh": required_total_kwh,
        "served_total_kwh": served_total_kwh,
        "unmet_total_kwh": total_unmet_kwh,
        "served_fraction": served_fraction,
        "peak_ev_kw": float(np.max(internal_load)) if internal_load.size else 0.0,
    }


def loadAnalysis(chargeprofiles, total_load, objective_mode=None, results=None, alpha=0.05): # synthesize of the results
    opt_EVprofiles = np.zeros((n_V1G_Veh, SLOTS_PER_DAY))
    for i in chargeprofiles.items():
        name = i[1].name
        index = re.findall(r'\d+', name)
        index = [int(i) for i in index]
        veh_index = index[0]
        time_index = index[1]
        opt_EVprofiles[veh_index][time_index] = i[1].varValue

    opt_EVload = np.zeros(SLOTS_PER_DAY)
    for i in range(SLOTS_PER_DAY):
        opt_EVload[i] = sum(row[i] for row in opt_EVprofiles)

    opt_Totalload = np.zeros(SLOTS_PER_DAY)
    for i in total_load.items():
        name = i[1].name
        index = re.findall(r'\d+', name)
        index = [int(i) for i in index]
        time_index = index[0]
        opt_Totalload[time_index] = i[1].varValue

    # show the results
    capacity_Line = np.full(SLOTS_PER_DAY, transformer_capacity)
    efficiency_Line = np.full(SLOTS_PER_DAY, maxEfficiency * transformer_capacity)

    x = np.arange(0, SLOTS_PER_DAY, 1) * SLOT_HOURS
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.set_xlabel('Time of a day: hour')
    ax1.set_ylabel('Load: kW')
    ax1.fill_between(x, unmanaged_Load, color='whitesmoke', label='EV Load: Unmanaged Charging')
    ax1.fill_between(x, opt_EVload, color='lightcyan', label='EV Load: Managed Charging')
    EV_load = opt_EVload + unmanaged_Load
    ax1.plot(x, EV_load, color='orange', lw=3, label='EV Load: Total')
    if opt_includeSolar==1:
        ax1.plot(x, solarCurve, color='limegreen', linestyle=':', label='Solar output')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_ylim([-1000, 4500])

    ax2.set_xlabel('Time of a day: hour')
    ax2.set_ylabel('Load: kW')
    ax2.set_ylim([-1000, 4500])
    if opt_includeSolar==0:
        ax2.plot(x, opt_Totalload, color='gray', lw=3, label='Total load')
        ax2.fill_between(x, base_Load, opt_Totalload, color='orange', label='EV Load')
        ax2.fill_between(x, base_Load, color='lemonchiffon', label='Base Load')
    else:
        Load_beforeSolar=EV_load + base_Load
        ax2.fill_between(x, opt_Totalload, Load_beforeSolar, alpha=0.5, color='limegreen', label='Solar Consumption')
        base_Load1 = np.minimum(base_Load, opt_Totalload)
        ax2.fill_between(x, base_Load1, opt_Totalload, color='orange', label='EV Load(Remaining)')
        ax2.fill_between(x, base_Load, color='lemonchiffon', alpha=0.4, label='Base Load')
        ax2.plot(x, opt_Totalload, color='black', lw=3, label='Total Load')

    ax2.plot(x, capacity_Line, 'r:', label='Rated capacity')
    ax2.plot(x, efficiency_Line, 'y:', label='80% of Rated capacity')
    ax2.legend(fontsize=8, loc='upper left')

    max_Load = float(np.max(opt_Totalload))
    min_Load = float(np.min(opt_Totalload))
    maxIndex = int(np.argmax(opt_Totalload))
    peak_Valley_Diff = max_Load - min_Load
    energy_cost = np.sum(np.maximum(opt_Totalload, 0)) * SLOT_HOURS * 0.212
    demand_charge = np.max(opt_Totalload) * 0.409

    solver_status = results.get("status", "Unknown") if results else "Unknown"
    objective_value = results.get("objective", np.nan) if results else np.nan
    peak_base = float(np.max(base_Load))
    peak_base_unmanaged = float(np.max(base_Load + unmanaged_Load))
    delta_vs_base = max_Load - peak_base
    delta_vs_base_unmanaged = max_Load - peak_base_unmanaged
    base_peak_slot = int(np.argmax(base_Load))
    internal_ev_at_hospital_peak = float(opt_EVload[base_peak_slot])
    rated_kw = float(transformer_capacity)
    limit80_kw = float(maxEfficiency * transformer_capacity)
    utilization = max_Load / rated_kw if rated_kw > 0 else 0.0
    headroom80 = limit80_kw - max_Load
    headroom100 = rated_kw - max_Load
    exceed80 = max(0.0, max_Load - limit80_kw)
    exceed100 = max(0.0, max_Load - rated_kw)
    ev_at_peak = float(opt_EVload[maxIndex])

    print('Solver status:', solver_status)
    print('Objective mode:', objective_mode, '| alpha:', alpha)
    if results:
        print('Solved objective name:', results.get("objective_name", "n/a"))
    print('Objective value: {:.3f}'.format(objective_value))
    if solver_status != 'Optimal':
        print('Not optimal -> skip KPI reporting')
        return EV_load,opt_Totalload

    print('Optimized daily peak load: {:.1f} kW'.format(max_Load))
    print('Maximum aggregated EV charging power: {:.1f} kW'.format(float(np.max(opt_EVload))))
    print('Peak increase vs base load only: {:+.1f} kW'.format(delta_vs_base))
    print('Peak increase vs base + unmanaged EVs: {:+.1f} kW'.format(delta_vs_base_unmanaged))
    print('Managed internal EV power at hospital highest-load moment: {:+.1f} kW'.format(internal_ev_at_hospital_peak))
    print('Peak valley difference: {:.1f} kW'.format(peak_Valley_Diff))
    print('Transformer utilization (at peak): {:.0%}'.format(utilization))
    print('Headroom to 80% limit: {:.0f} kW'.format(headroom80))
    print('Headroom to rated capacity: {:.0f} kW'.format(headroom100))
    print('Capacity exceedance (80% / 100%): {:.0f} kW / {:.0f} kW'.format(exceed80, exceed100))
    print('Managed EV power at system peak slot: {:.1f} kW (slot {})'.format(ev_at_peak, maxIndex))

    if objective_mode == "A_peak":
        print('Objective note: minimizing absolute system peak load.')
    elif objective_mode == "B_peak_above_baseline":
        baseline = base_Load + unmanaged_Load - (solarCurve if opt_includeSolar else 0)
        baseline_peak = float(np.max(baseline))
        peak_increase_measured = float(np.max(opt_Totalload - baseline))
        peak_increase_obj = results.get("peak_increase", np.nan) if results else np.nan
        print('Baseline peak (no managed): {:.1f} kW'.format(baseline_peak))
        print('Optimized peak (with managed): {:.1f} kW'.format(max_Load))
        print('Peak increase (objective B): {:.3f} kW'.format(peak_increase_obj))
        print('Peak increase (measured max(total-baseline)): {:.3f} kW'.format(peak_increase_measured))
    elif objective_mode == "C_spread_plus_peak":
        spread_term = max_Load - min_Load
        peak_term = alpha * max_Load
        obj_c = spread_term + peak_term
        print('Max load: {:.3f} kW'.format(max_Load))
        print('Min load: {:.3f} kW'.format(min_Load))
        print('Spread term (max-min): {:.3f} kW'.format(spread_term))
        print('Peak penalty term (alpha*max): {:.3f}'.format(peak_term))
        print('Objective C value (spread + alpha*max): {:.3f}'.format(obj_c))
    elif objective_mode == "D_service_first_peak":
        unmet_total_kwh = results.get("unmet_total_kwh", np.nan) if results else np.nan
        peak_ev_obj = results.get("peak_ev_kw", np.nan) if results else np.nan
        served_total_kwh = results.get("served_total_kwh", np.nan) if results else np.nan
        served_fraction = results.get("served_fraction", np.nan) if results else np.nan
        print('Total unmet charging energy (objective D): {:.3f} kWh'.format(unmet_total_kwh))
        print('Managed EV peak (objective D variable): {:.3f} kW'.format(peak_ev_obj))
        print('Managed EV peak (measured max managed EV load): {:.3f} kW'.format(float(np.max(opt_EVload))))
        print('Total served charging energy (objective D): {:.3f} kWh'.format(served_total_kwh))
        print('Served fraction (objective D): {:.2%}'.format(served_fraction) if not np.isnan(served_fraction) else 'Served fraction (objective D): nan')

    print('Energy cost (EUR/day): {:.2f}'.format(energy_cost))
    print('Demand charge (EUR/day): {:.2f}'.format(demand_charge))

    return EV_load,opt_Totalload

def writeCSV(evLoad, opTotal_load):
    df = pd.DataFrame({'EV load': evLoad,'Total load': opTotal_load})
    output_path = 'C:\\Users\\razan\\Desktop\\ev-simulator-main\\workplace.csv'
    df.to_csv(output_path, index=False)


if __name__ == '__main__':

    useCase= 4 # 1:residential, 2:office, 3: commercial, 4: user defined
    active_cap = get_active_dwell_cap_minutes()
    if active_cap is None:
        print("Dwell scenario:", DWELL_SCENARIO, "| upper dwell cap: none")
    else:
        print("Dwell scenario:", DWELL_SCENARIO, "| upper dwell cap (min):", active_cap)
    print("Behavior scenario:", BEHAVIOR_SCENARIO)
    scenario_rules = SCENARIO_RULES.get(BEHAVIOR_SCENARIO, SCENARIO_RULES["hospital"])
    batteryCapacity = 60
    energyEfficiency = 90  # energy consumption per 100km travelled

    isManaged=1 #unmanaged charging=0 or managed charging=1
    power_V1GLower = 0 #lower power limit for managed charging
    power_V1GUppper= 22 #upper power limit for managed charging
    # site hardware counts
    internal_chargers_available = 15  # managed/internal chargers
    public_chargers_available = 4     # unmanaged/public chargers

    # Daily internal/external counts (managed = internal)
    n_internal_today = sample_internal_daily_count()
    n_external_today = 15 if scenario_rules["include_external_sessions"] else 0
    n_vehicles = n_internal_today
    percentV1G = n_internal_today / n_vehicles if n_vehicles > 0 else 0
    if BEHAVIOR_SCENARIO in SCENARIO_BASELINE_INTERNAL_COUNTS:
        print("Configured internal EV sessions today:", n_internal_today)
    else:
        print("Sampled internal EV sessions today:", n_internal_today)
    print("Sampled external EV sessions today:", n_external_today)

    if useCase==1: #residential
        percentFast = 0  # out of vehicles for Unmanaged Charging, how many vehicles are fast-charging vehicles
        nslow_Chargers = 800  # unmanaged chargers: number of level 1 chargers
        nfast_Chargers = 800 #unmanaged chargers: number of level 2 and DCF chargers
        power_Slow = 7
        power_Fast = 7 #divide line between fast charging power and slow charging power is: 10kW

    if useCase==2 or useCase==3: #workplace or public charging stations: fast charging only
        percentFast = 1  # fixed, not an input
        nslow_Chargers = 0  # fixed, not an input
        nfast_Chargers = 400  # unmanaged chargers: number of level 2 and DCF chargers
        power_Fast = 10

    if useCase==4: #other case
        ngen_Chargers= public_chargers_available
        power_General = 7
        nslow_Chargers= 0 #dummy
        nfast_Chargers= 0 #dummy
        percentFast= 0 # all unmanaged EVs treated as slow in useCase 4

    if useCase==1 or useCase ==2:
        Charge_frequency = 3 #to determine startingSOC of fast charging vehicles

    transformer_capacity=4000

    # Optional: override base load from hospital CSV.
    base_Load_from_file = None
    base_load_profiles = {}
    hospital_load_file = "hospital_data_with_weekday_weekend.csv"
    base_load_specific_date = None  # e.g., "2025-01-15" when base_load_mode == "specific_date"
    try:
        df_load = pd.read_csv(hospital_load_file, parse_dates=["Timestamp"])
        df_load = df_load.sort_values("Timestamp")
        df_load["slot"] = df_load["Timestamp"].dt.hour * (60 // SLOT_MINUTES) + (df_load["Timestamp"].dt.minute // SLOT_MINUTES)
        df_load["date"] = df_load["Timestamp"].dt.date

        slot_avg = df_load.groupby("slot")["hospitalpower_kw"].mean()
        base_load_profiles["average"] = slot_avg.reindex(range(SLOTS_PER_DAY)).to_numpy()

        slot_by_day = df_load.groupby(["date", "slot"])["hospitalpower_kw"].mean().unstack(fill_value=0)
        worst_day = slot_by_day.max(axis=1).idxmax()
        base_load_profiles["worst_peak"] = slot_by_day.loc[worst_day].reindex(range(SLOTS_PER_DAY), fill_value=0).to_numpy()

        if base_load_specific_date:
            target_date = pd.to_datetime(base_load_specific_date).date()
            df_day = df_load[df_load["Timestamp"].dt.date == target_date]
            if not df_day.empty:
                slot_avg = df_day.groupby("slot")["hospitalpower_kw"].mean()
                base_load_profiles["specific_date"] = slot_avg.reindex(range(SLOTS_PER_DAY)).fillna(0).to_numpy()
    except FileNotFoundError:
        base_load_profiles = {}

    if useCase==1:
        base_Load = np.array([1578.7, 1414.7, 1290.1, 1258.6, 1199.2, 1279.8, 1327.9, 1378, 1492.4, 1666.8, 1738.2, 1497.9, 1433.9,
                     1446.8, 1463.8, 1434.5, 1523.9, 1651.1, 1727.2, 1922, 2162.6, 2192.6, 1944.4, 1762.9])
    if useCase==2:
        base_Load =np.array([1043.071385, 1009.268146, 980.293941, 985.1229752, 973.0503897, 1011.682663, 1294.181163, 1675.674865, 2087.350029,2290.169466, 2391.579184, 2125.982303,2237.050089, 1812.095081, 1746.903119,1733.623275, 1717.928914, 1800.022495,1558.570785, 1302.631973, 1154.139172,981.5011995, 874.0551888,	802.8269344])
    if useCase==3:
        base_Load=np.array([726.1660166,503.4268145,449.1001799,443.6675164,449.1001799,488.9397119,467.2090581,599.4038691,918.1201258,2147.712957,3290.383173,3451.552189,3536.663916,3507.689711,3324.790041,3047.724204,3054.967756,3281.328733,3348.331583,3422.577984,3409.901769,3248.732753,2692.790191,1774.670066])
    if useCase==4:
        base_Load=np.zeros(SLOTS_PER_DAY) #user defined

    # expand 24-element base load to 96 slots by repeating each hour into four 15-min slots
    if base_Load.size == 24:
        base_Load = np.repeat(base_Load, 60 // SLOT_MINUTES)
    if not base_load_profiles:
        base_load_profiles["average"] = base_Load.copy()
        base_load_profiles["worst_peak"] = base_Load.copy()

    #for managed charging
    parking_Duration=np.zeros(24) + 8 # user can redefine the duration in hours, like from 8 hours (here) to 10 hours
    #parking_Duration = [9.8, 9, 9.3, 8.8, 8.5, 7.3, 7.4, 7.7, 6.8, 5.4, 5, 5.2, 5.1, 5.2, 5, 6.1, 7.2, 9.1, 9.8, 9.8,
    #                    10.7, 9.4, 10.1, 8.8]
    opt_includeSolar=0 #0 doesn't include solar, 1 includes solar; solarCurve can be user definedr
    solarCurve=np.array([0,	0,	0,	0,	0,	0.867388228,	12.39391599,	40.83306752,	77.55910021,	112.4685535,	136.0694384,	149.133645,	156.6184249,	153.7845818,	135.7031852,	104.389254,	72.90418373,	37.78768372,	11.03414567,	0.813459333,	0,	0,	0,	0])
    solarCurve=solarCurve * 8
    if solarCurve.size == 24:
        solarCurve = np.repeat(solarCurve, 60 // SLOT_MINUTES)

    ###################fixed parameters
    end_SOC=2 # for managed charging 1 is to cover the travel distance, 2 is to charge fully
    charge_efficiency=0.9 #charging power loss
    maxEfficiency=0.8 #maximual capacity of transformer
    n_iter=25   #number of iterations for the unmanaged model
    Maxtime_Interval=1440 #time interval for unmanaged model
    isSlow=1

    batteryRange = batteryCapacity / energyEfficiency * 100
    n_V1G_Veh_base = int(np.round(n_vehicles * percentV1G))
    n_V0G_Veh_base = int(np.round(n_vehicles * (1 - percentV1G)))
    # Use physical internal charger count for site availability (not EV count).
    nV1G_Chargers = internal_chargers_available
    nfast_base = nfast_Chargers
    nslow_base = nslow_Chargers
    ngen_base = ngen_Chargers if "ngen_Chargers" in globals() else None

    # Comparison mode: run unmanaged + selected managed objectives on repeated frozen scenarios.
    COMPARE_UNMANAGED_AND_OBJECTIVES = True
    comparison_day_start = pd.Timestamp("2023-09-27")
    fixed_k_blocked = 0  # fixed baseline-comparison assumption across scenarios
    comparison_day_start = resolve_simulation_day_start(comparison_day_start)

    if COMPARE_UNMANAGED_AND_OBJECTIVES:
        scenario_defs = [
            ("unmanaged_only", 0, None),
            ("managed_A_peak", 1, "A_peak"),
            ("managed_B_peak_above_baseline", 1, "B_peak_above_baseline"),
            ("managed_D_service_first_peak", 1, "D_service_first_peak"),
            ("managed_spread", 1, "spread"),
        ]
    else:
        scenario_defs = [("single_run", isManaged, "D_service_first_peak" if isManaged == 1 else None)]

    comparison_rows = []
    for run_idx in range(MONTE_CARLO_RUNS):
        comparison_seed = BASELINE_COMPARISON_SEED + run_idx

        # ------------------------------
        # Frozen scenario generation (once per Monte Carlo run)
        # ------------------------------
        np.random.seed(comparison_seed)
        n_V1G_Veh = n_V1G_Veh_base
        n_V0G_Veh = 0
        day_start = comparison_day_start
        simulation_day_start = day_start
        if fixed_k_blocked is None:
            available_internal_chargers_by_slot, k_blocked = build_available_internal_chargers_all_day(
                day_start=day_start,
                n_internal_chargers=nV1G_Chargers,
                slots_per_day=SLOTS_PER_DAY,
            )
        else:
            k_blocked = int(max(0, min(fixed_k_blocked, nV1G_Chargers - 1)))
            available_internal_chargers_by_slot = np.full(SLOTS_PER_DAY, nV1G_Chargers - k_blocked, dtype=int)

        nfast_Chargers = nfast_base
        nslow_Chargers = nslow_base
        if ngen_base is not None:
            ngen_Chargers = ngen_base
        n_fastVeh = 0
        n_slowVeh = 0
        if nfast_Chargers > n_fastVeh:
            nfast_Chargers = n_fastVeh
        if nslow_Chargers > n_slowVeh:
            nslow_Chargers = n_slowVeh
        isSlow = 2 if percentFast > 0 else 1

        slow_Load_1h_avg = np.zeros(SLOTS_PER_DAY)
        fast_Load_1h_avg = np.zeros(SLOTS_PER_DAY)
        gen_Load_1h_avg = np.zeros(SLOTS_PER_DAY)
        unmanaged_Load = np.zeros(SLOTS_PER_DAY)
        (
            avgwaitSlow, avgwaitFast, avgwaitGen,
            avgWaitedSlow, avgWaitedFast, avgWaitedGen,
            avgUnchargedSlow, avgUnchargedFast, avgUnchargedGen
        ) = Unmanaged_usecase(useCase)
        unmanaged_Load = slow_Load_1h_avg + fast_Load_1h_avg + gen_Load_1h_avg
        if scenario_rules["external_as_fixed_load"]:
            external_fixed_load = build_fixed_external_load(
                day_start=simulation_day_start,
                n_sessions=n_external_today,
                power=power_General,
                n_chargers=public_chargers_available,
            )
            unmanaged_Load = np.zeros(SLOTS_PER_DAY)
        else:
            external_fixed_load = np.zeros(SLOTS_PER_DAY)

        v1g_startingTime, v1g_endingTime, v1g_startingSOC = initiate_V1G(n_V1G_Veh)
        diagnose_v1g_samples(
            v1g_startingTime,
            v1g_endingTime,
            v1g_startingSOC,
            v1g_energy_required_kwh,
            batteryCapacity,
            dwell_slots=v1g_dwell_slots,
        )

        fixed_available_internal = np.array(available_internal_chargers_by_slot, dtype=int).copy()
        fixed_unmanaged_Load = np.array(unmanaged_Load, dtype=float).copy()
        fixed_v1g_startingTime = np.array(v1g_startingTime, dtype=int).copy()
        fixed_v1g_endingTime = np.array(v1g_endingTime, dtype=int).copy()
        fixed_v1g_startingSOC = np.array(v1g_startingSOC, dtype=float).copy()
        fixed_v1g_energy_required_kwh = np.array(v1g_energy_required_kwh, dtype=float).copy()
        fixed_v1g_dwell_slots = np.array(v1g_dwell_slots, dtype=int).copy()

        print("\nFrozen scenario checks:")
        print("run_idx =", int(run_idx), "| seed =", int(comparison_seed))
        print("k_blocked =", int(k_blocked))
        print("sum(unmanaged_Load) =", float(np.sum(fixed_unmanaged_Load)))
        print("sum(v1g_energy_required_kwh) =", float(np.sum(fixed_v1g_energy_required_kwh)))
        print("v1g_startingTime[:5] =", fixed_v1g_startingTime[:5].tolist())
        print("v1g_endingTime[:5] =", fixed_v1g_endingTime[:5].tolist())

        # ------------------------------
        # Objective comparison (reuse frozen EV/unmanaged scenario within each run)
        # ------------------------------
        for base_profile_name in ["average", "worst_peak"]:
            if base_profile_name not in base_load_profiles:
                continue
            base_Load = np.array(base_load_profiles[base_profile_name], dtype=float).copy()
            base_Load = base_Load + external_fixed_load
            print("\n" + "#" * 72)
            print("Run:", int(run_idx), "| Seed:", int(comparison_seed), "| Base load profile:", base_profile_name)

            for scenario_name, scenario_is_managed, scenario_objective_mode in scenario_defs:
                print("\n" + "=" * 72)
                print("Scenario:", scenario_name)

                n_V1G_Veh = n_V1G_Veh_base
                n_V0G_Veh = 0
                simulation_day_start = day_start
                available_internal_chargers_by_slot = fixed_available_internal.copy()
                unmanaged_Load = fixed_unmanaged_Load.copy()
                v1g_startingTime = fixed_v1g_startingTime.copy()
                v1g_endingTime = fixed_v1g_endingTime.copy()
                v1g_startingSOC = fixed_v1g_startingSOC.copy()
                v1g_energy_required_kwh = fixed_v1g_energy_required_kwh.copy()
                v1g_dwell_slots = fixed_v1g_dwell_slots.copy()

                print("Midnight blocked internal outlets (k:):", int(k_blocked), "| day type:", _day_type(day_start))
                print("avail chargers min/max:", int(np.min(available_internal_chargers_by_slot)), int(np.max(available_internal_chargers_by_slot)))

                if scenario_is_managed == 1:
                    print('start managed charging')
                    print("power_V1GUppper:", power_V1GUppper, "eff:", charge_efficiency, "SLOT_HOURS:", SLOT_HOURS)
                    chargeprofiles, total_load, results = V1G_optimization(objective_mode=scenario_objective_mode)
                    EV_load, opTotal_load = loadAnalysis(
                        chargeprofiles,
                        total_load,
                        objective_mode=scenario_objective_mode,
                        results=results,
                    )
                    objective_value = float(results.get("objective", np.nan))
                    solver_status = results.get("status", "Unknown")
                    unmet_total_kwh = float(results.get("unmet_total_kwh", np.nan))
                    managed_peak_obj_kw = float(results.get("peak_ev_kw", np.nan))
                    required_total_kwh = float(results.get("required_total_kwh", np.nan))
                    served_total_kwh = float(results.get("served_total_kwh", np.nan))
                    served_fraction = float(results.get("served_fraction", np.nan))
                    internal_ev_profile = np.maximum(EV_load - unmanaged_Load, 0)
                else:
                    internal_unmanaged_load, opTotal_load, results = simulate_unmanaged_internal()
                    EV_load = unmanaged_Load + internal_unmanaged_load
                    objective_value = np.nan
                    solver_status = results.get("status", "Simulated")
                    unmet_total_kwh = float(results.get("unmet_total_kwh", np.nan))
                    managed_peak_obj_kw = float(results.get("peak_ev_kw", np.nan))
                    required_total_kwh = float(results.get("required_total_kwh", np.nan))
                    served_total_kwh = float(results.get("served_total_kwh", np.nan))
                    served_fraction = float(results.get("served_fraction", np.nan))
                    internal_ev_profile = internal_unmanaged_load

                baseline_series = base_Load - (solarCurve if opt_includeSolar else 0)
                peak_total_kw = float(np.max(opTotal_load))
                baseline_peak_kw = float(np.max(baseline_series))
                peak_increase_kw = float(peak_total_kw - baseline_peak_kw)
                baseline_peak_slot = int(np.argmax(base_Load))
                internal_ev_at_hospital_peak_kw = float(internal_ev_profile[baseline_peak_slot])
                managed_peak_kw = float(np.max(np.maximum(EV_load - unmanaged_Load, 0)))
                energy_cost_eur = float(np.sum(np.maximum(opTotal_load, 0)) * SLOT_HOURS * 0.212)
                demand_charge_eur = float(np.max(opTotal_load) * 0.409)

                comparison_rows.append({
                    "run_idx": int(run_idx),
                    "seed": int(comparison_seed),
                    "base_profile": base_profile_name,
                    "scenario": scenario_name,
                    "isManaged": int(scenario_is_managed),
                    "objective_mode": scenario_objective_mode if scenario_objective_mode is not None else "unmanaged",
                    "status": solver_status,
                    "objective_value": objective_value,
                    "day_start": str(day_start.date()),
                    "day_type": _day_type(day_start),
                    "k_blocked": int(k_blocked),
                    "avail_internal": int(np.min(available_internal_chargers_by_slot)),
                    "n_internal_ev": int(n_V1G_Veh),
                    "n_external_ev": int(n_V0G_Veh),
                    "peak_total_kw": peak_total_kw,
                    "baseline_peak_kw": baseline_peak_kw,
                    "peak_increase_kw": peak_increase_kw,
                    "internal_ev_at_hospital_peak_kw": internal_ev_at_hospital_peak_kw,
                    "managed_peak_kw": managed_peak_kw,
                    "unmet_total_kwh": unmet_total_kwh,
                    "managed_peak_obj_kw": managed_peak_obj_kw,
                    "required_total_kwh": required_total_kwh,
                    "served_total_kwh": served_total_kwh,
                    "served_fraction": served_fraction,
                    "energy_cost_eur": energy_cost_eur,
                    "demand_charge_eur": demand_charge_eur,
                })

                if scenario_name == "single_run" and run_idx == 0:
                    writeCSV(EV_load, opTotal_load)

    df_comp = pd.DataFrame(comparison_rows)
    comp_path = "C:\\Users\\razan\\Desktop\\ev-simulator-main\\objective_comparison_same_setup.csv"
    df_comp.to_csv(comp_path, index=False)
    summary_metrics = [
        "k_blocked",
        "avail_internal",
        "peak_total_kw",
        "baseline_peak_kw",
        "peak_increase_kw",
        "internal_ev_at_hospital_peak_kw",
        "managed_peak_kw",
        "unmet_total_kwh",
        "managed_peak_obj_kw",
        "required_total_kwh",
        "served_total_kwh",
        "served_fraction",
        "energy_cost_eur",
        "demand_charge_eur",
    ]
    df_comp_summary = (
        df_comp.groupby(["base_profile", "scenario", "objective_mode"])[summary_metrics]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    df_comp_summary.columns = [
        "_".join([str(level) for level in col if str(level)])
        for col in df_comp_summary.columns.to_flat_index()
    ]
    summary_path = "C:\\Users\\razan\\Desktop\\ev-simulator-main\\objective_comparison_same_setup_summary.csv"
    df_comp_summary.to_csv(summary_path, index=False)
    print("\nComparison summary (same setup):")
    print("Monte Carlo runs:", int(MONTE_CARLO_RUNS), "| seed range:", int(BASELINE_COMPARISON_SEED), "to", int(BASELINE_COMPARISON_SEED + MONTE_CARLO_RUNS - 1))
    print(df_comp_summary.to_string(index=False))
    print("Saved:", comp_path)
    print("Saved:", summary_path)


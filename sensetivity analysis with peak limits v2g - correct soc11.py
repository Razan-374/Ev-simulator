#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pulp as lp
import re
import os
import tempfile
from matplotlib import pyplot as plt
from queue import PriorityQueue
import pandas as pd
from master_fleet import ensure_master_fleet

SLOT_MINUTES = 15
SLOTS_PER_DAY = int(1440 / SLOT_MINUTES)  # 96 slots
SLOT_HOURS = SLOT_MINUTES / 60.0
CHARGE_EFFICIENCY = 0.90
DISCHARGE_EFFICIENCY = 0.95
ENERGY_TARGET_BASIS = "charger_side"
POWER_CONFIGURATIONS_KW = (11, 22)
POWER_LIMIT_KW = 11
EXTERNAL_OUTLETS = 5
EMERGENCY_USER_IDS = set()  # Populate with protected session_behaviour_clusters.csv user_id values.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1_corrected_outputs")
MASTER_FLEET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_fleets")
active_fleet_hash = ""

# Dwell scenario controls for single-day planning vs realism studies.
# "operational": cap dwell at 24h for strict day-ahead planning.
# "realism": keep long overstays (no upper cap).
DWELL_SCENARIO = "operational"  # options: "operational", "realism"
DWELL_MIN_MINUTES = 10
OPERATIONAL_DWELL_CAP_MINUTES = 24 * 60
ALLOW_FULL_INTERNAL_BLOCK = False  # when False, keep at least one internal charger available if any exist

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
ALLOW_UNMET_ENERGY = True  # allow partial charge completion and record unmet energy instead of infeasibility
UNMET_ENERGY_PENALTY_PER_KWH = 1000.0  # legacy A/B/C penalty (not used by objective D)
LEXI_M = 1e4  # dominant penalty for service-first objective D
SITE_CAP_KW = 2000.0  # hard site cap used to compute EV headroom each slot
EPS_UNMET_KWH = 1e-6
SERVICE_FEASIBILITY_TOLERANCE_KWH = 1e-4
EPS_LIMIT_KW = 1e-6

TARGET_PEAK_REDUCTION_KW = 300.0
cluster_ids = list(range(len(cluster_weights)))
arrival_histograms = {}
dwell_samples_by_cluster = {}
energy_samples_by_cluster = {}
dwell_energy_pairs_by_cluster = {}
internal_triplets_by_cluster = {}
internal_triplets_by_cluster_and_daytype = {}
internal_user_ids_by_cluster = {}
internal_user_ids_by_cluster_and_daytype = {}
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
CBC_MSG = True
CBC_TIME_LIMIT_SEC = 120
CBC_REL_GAP = 0.02

def get_active_dwell_cap_minutes():
    if DWELL_SCENARIO == "operational":
        return OPERATIONAL_DWELL_CAP_MINUTES
    return None

# Empirical internal daily counts (from internal_daily_count_distribution.csv)
internal_daily_counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
internal_daily_probs = np.array([0.1009, 0.1374, 0.1304, 0.1617, 0.1496, 0.1478, 0.0974, 0.0556, 0.0139, 0.0052])
internal_daily_probs = internal_daily_probs / internal_daily_probs.sum()

def sample_internal_daily_count():
    return int(np.random.choice(internal_daily_counts, p=internal_daily_probs))

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

def sample_k_blocked(day_start, rng=None):
    rng = rng or np.random.default_rng()
    dtype = _day_type(day_start)
    dist = k_dist_by_type[dtype]

    k_vals = np.array(list(dist.keys()), dtype=int)
    probs = np.array(list(dist.values()), dtype=float)
    probs = probs / probs.sum()
    return int(rng.choice(k_vals, p=probs))

def build_available_internal_chargers_all_day(day_start, n_internal_chargers, slots_per_day, rng=None):
    """Sample blocked internal outlets once and apply for all slots in the day."""
    k = sample_k_blocked(day_start, rng=rng)
    if ALLOW_FULL_INTERNAL_BLOCK:
        k = min(k, n_internal_chargers)
    else:
        # Keep at least one outlet available whenever internal chargers exist.
        k = min(k, max(n_internal_chargers - 1, 0))
    available = np.full(slots_per_day, n_internal_chargers - k, dtype=int)
    available[available < 0] = 0
    return available, k

# --- Load session data (adjust path if needed)
try:
    df = pd.read_csv("session_behaviour_clusters.csv", parse_dates=["arrival_time"])
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")
    df = df.dropna(subset=["arrival_time"])
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype("string").fillna("")
    else:
        df["user_id"] = ""

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
    internal_user_ids_by_cluster = {}
    internal_user_ids_by_cluster_and_daytype = {}
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
        internal_user_ids_by_cluster[c] = sub["user_id"].to_numpy(dtype=str)

        # Day-type split for internal sessions; keep cluster-based sampling with per-day fallback.
        day_type = sub["arrival_time"].dt.weekday.map(
            lambda wd: "friday" if wd == 4 else ("weekend" if wd >= 5 else "weekday")
        )
        internal_triplets_by_cluster_and_daytype[c] = {"all": all_triplets}
        internal_user_ids_by_cluster_and_daytype[c] = {
            "all": sub["user_id"].to_numpy(dtype=str)
        }
        for dt in ("weekday", "friday", "weekend"):
            dt_mask = day_type == dt
            dt_sub = sub.loc[dt_mask, ["minute_of_day", "dwell_minutes", "energy_kwh"]]
            internal_triplets_by_cluster_and_daytype[c][dt] = (
                dt_sub.to_numpy(dtype=float) if not dt_sub.empty else np.empty((0, 3))
            )
            internal_user_ids_by_cluster_and_daytype[c][dt] = (
                sub.loc[dt_mask, "user_id"].to_numpy(dtype=str)
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

def sample_internal_triplets(
    n,
    cluster_ids_arr=None,
    day_start=None,
    min_pool=30,
    return_user_ids=False,
):
    clusters = cluster_ids_arr if cluster_ids_arr is not None else sample_cluster(n)
    triplets = np.zeros((n, 3), dtype=float)  # [arrival_minute, dwell_minutes, energy_kwh]
    sampled_user_ids = np.full(n, "", dtype=str)
    max_dwell_min = get_active_dwell_cap_minutes()
    if day_start is None:
        day_start = simulation_day_start
    dt = _day_type(day_start)
    max_energy_kwh = float(globals().get("batteryCapacity", 60.0))
    for c in np.unique(clusters):
        mask = clusters == c
        if not np.any(mask):
            continue
        count = mask.sum()
        pool_by_dt = internal_triplets_by_cluster_and_daytype.get(c, {})
        user_ids_by_dt = internal_user_ids_by_cluster_and_daytype.get(c, {})
        pool = pool_by_dt.get(dt, np.empty((0, 3)))
        user_ids = user_ids_by_dt.get(dt, np.array([], dtype=str))
        if pool.shape[0] < min_pool:
            pool = pool_by_dt.get("all", pool)
            user_ids = user_ids_by_dt.get("all", user_ids)
        if pool.shape[0] == 0:
            pool = internal_triplets_by_cluster.get(c, np.empty((0, 3)))
            user_ids = internal_user_ids_by_cluster.get(c, np.array([], dtype=str))
        if pool.shape[0] > 0:
            valid_mask = pool[:, 2] <= max_energy_kwh + 1e-9
            valid_pool = pool[valid_mask]
            valid_user_ids = (
                user_ids[valid_mask]
                if user_ids.size == pool.shape[0]
                else np.full(pool.shape[0], "", dtype=str)[valid_mask]
            )
            if valid_pool.shape[0] == 0:
                raise ValueError(
                    f"No valid internal sessions remain for cluster {c}: "
                    f"all sampled energy requirements exceed {max_energy_kwh:.1f} kWh."
                )
            idx = np.random.choice(valid_pool.shape[0], size=count, replace=True)
            sampled = valid_pool[idx].copy()
            sampled_user_ids[mask] = valid_user_ids[idx]
            sampled[:, 0] = np.clip(sampled[:, 0], 0, 1439)
            sampled[:, 1] = np.maximum(sampled[:, 1], DWELL_MIN_MINUTES)
            if max_dwell_min is not None:
                sampled[:, 1] = np.minimum(sampled[:, 1], max_dwell_min)
            sampled[:, 2] = np.maximum(sampled[:, 2], 0.1)
            triplets[mask] = sampled
        else:
            # Rejection-sample complete fallback sessions until all demands fit the battery.
            accepted = []
            while sum(item.shape[0] for item in accepted) < count:
                batch_size = count - sum(item.shape[0] for item in accepted)
                arr = sample_arrivals(batch_size)
                dwell, energy = sample_dwell_energy(batch_size)
                candidate = np.column_stack([arr, dwell, np.maximum(energy, 0.1)])
                accepted.append(candidate[candidate[:, 2] <= max_energy_kwh + 1e-9])
            sampled = np.vstack(accepted)[:count]
            sampled[:, 1] = np.maximum(sampled[:, 1], DWELL_MIN_MINUTES)
            if max_dwell_min is not None:
                sampled[:, 1] = np.minimum(sampled[:, 1], max_dwell_min)
            triplets[mask] = sampled
    if return_user_ids:
        return triplets, sampled_user_ids
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

v1g_energy_required_kwh = None
v1g_dwell_slots = None
v1g_startingTime = None
v1g_endingTime = None
v1g_startingSOC = None
v1g_user_ids = np.array([], dtype=str)
v1g_emergency_flags = np.array([], dtype=bool)

def build_window_slots(temp_start, temp_end, dwell_slots_i):
    if dwell_slots_i <= 0:
        return []
    if dwell_slots_i >= SLOTS_PER_DAY:
        return list(range(SLOTS_PER_DAY))
    if temp_start < temp_end:
        return list(range(temp_start, temp_end))
    return list(range(temp_start, SLOTS_PER_DAY)) + list(range(0, temp_end))

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

    # Empirical targets are charger-side kWh; convert to stored battery energy.
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
        startingSOC = 1.0 - (energy_required * CHARGE_EFFICIENCY / battery_cap)
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
        energy_minutes = np.maximum(0, (energy_required_kwh * 60) / power)

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
    global v1g_startingTime, v1g_endingTime, v1g_startingSOC
    global v1g_user_ids, v1g_emergency_flags
    master_internal, _, _, _ = ensure_master_fleet(
        seed=active_sweep_seed,
        day_type=_day_type(simulation_day_start),
        n_internal=max(n_V1G_Veh, 100),
        n_external=n_V0G_Veh,
        battery_capacity_kwh=batteryCapacity,
        fleet_dir=MASTER_FLEET_DIR,
        dwell_cap_minutes=get_active_dwell_cap_minutes(),
        protected_user_ids=EMERGENCY_USER_IDS,
    )
    sampled_triplets = master_internal[
        ["start_minute", "dwell_minutes", "energy_required_kwh"]
    ].to_numpy(dtype=float)[:n_V1G_Veh]
    arrival_minutes = sampled_triplets[:, 0].astype(int)
    dwell_minutes = sampled_triplets[:, 1]
    v1g_energy_required_kwh = sampled_triplets[:, 2]
    v1g_user_ids = np.full(n_V1G_Veh, "", dtype=str)
    v1g_emergency_flags = master_internal[
        "emergency_vehicle"
    ].to_numpy(dtype=bool)[:n_V1G_Veh].copy()

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
    derived_soc = np.clip(
        1.0
        - (energy_required_kwh * CHARGE_EFFICIENCY / max(battery_capacity_kwh, 1e-9)),
        0.0,
        1.0,
    )
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
    if objective_mode == "D_service_first_peak" and (not use_energy_targets or v1g_energy_required_kwh is None):
        raise ValueError("D_service_first_peak requires use_energy_targets=True and valid v1g_energy_required_kwh.")

    #define optimization variables
    veh_V1G=range(n_V1G_Veh)
    time_Interval=range(SLOTS_PER_DAY)

    chargeprofiles=lp.LpVariable.dicts('charging_profiles', ((i,j) for i in veh_V1G for j in time_Interval), lowBound=power_V1GLower, upBound=power_V1GUppper)
    charge_power_kw = lp.LpVariable.dicts(
        'charging_power_kw',
        ((i, j) for i in veh_V1G for j in time_Interval),
        lowBound=0.0,
        upBound=power_V1GUppper,
    )
    discharge_power_kw = lp.LpVariable.dicts(
        'discharging_power_kw',
        ((i, j) for i in veh_V1G for j in time_Interval),
        lowBound=0.0,
        upBound=abs(power_V1GLower),
    )
    chargestates=lp.LpVariable.dicts('charging_states', ((i,j) for i in veh_V1G for j in time_Interval), cat='Binary')
    dischargestates=lp.LpVariable.dicts('discharge_states', ((i,j) for i in veh_V1G for j in time_Interval), cat='Binary')
    total_load=lp.LpVariable.dicts('total_load', time_Interval, lowBound=0.0)
    total_ev_load=lp.LpVariable.dicts('total_ev_load', time_Interval, lowBound=None)
    max_load=lp.LpVariable('max_load', lowBound=0)
    min_load=lp.LpVariable('min_load', lowBound=0)
    peak_ev_kw = lp.LpVariable('peak_ev_kw', lowBound=0.0)
    unmet_energy = None
    if use_energy_targets and ALLOW_UNMET_ENERGY and v1g_energy_required_kwh is not None:
        unmet_energy = lp.LpVariable.dicts('unmet_energy', veh_V1G, lowBound=0.0)

    def battery_delta_kwh(i, t):
        return (
            charge_power_kw[i, t] * charge_efficiency
            - discharge_power_kw[i, t] * (1.0 / discharge_efficiency)
        ) * SLOT_HOURS

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
            emax = n_slots * power_V1GUppper * SLOT_HOURS
            if v1g_energy_required_kwh[i] > emax + 1e-9:
                infeasible_evs.append((i, v1g_energy_required_kwh[i], emax, n_slots))
        if infeasible_evs:
            if unmet_energy is not None:
                print("WARNING: Per-EV window shortfall risk detected (unmet energy slack will absorb deficit):")
            else:
                print("WARNING: Per-EV window infeasibility detected:")
            for ev_id, req_kwh, max_kwh, n_slots in infeasible_evs[:10]:
                print(f"  EV {ev_id}: required={req_kwh:.2f} kWh, max={max_kwh:.2f} kWh, window_slots={n_slots}")
            if len(infeasible_evs) > 10:
                print(f"  ... and {len(infeasible_evs) - 10} more EVs")

        e_total_req = float(np.sum(v1g_energy_required_kwh))
        e_total_max = float(np.sum(available_internal_chargers_by_slot)) * power_V1GUppper * SLOT_HOURS
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

    # constraint 3a: managed EV load aggregation + managed EV peak tracker
    for t in time_Interval:
        model += total_ev_load[t] == lp.lpSum(chargeprofiles[i, t] for i in veh_V1G)
        model += total_ev_load[t] <= peak_ev_kw

    # constraint 3aa: hard per-slot headroom under site cap
    for t in time_Interval:
        if opt_includeSolar == 0:
            headroom_t = max(0.0, SITE_CAP_KW - base_Load[t])
        else:
            headroom_t = max(0.0, SITE_CAP_KW - (base_Load[t] - solarCurve[t]))
        model += total_ev_load[t] <= headroom_t

    # constraint 3b: site charger availability by slot
    for t in time_Interval:
        model += lp.lpSum([chargestates[i, t] + dischargestates[i, t] for i in veh_V1G]) <= available_internal_chargers_by_slot[t]

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
                model += chargestates[i, t] + dischargestates[i, t] <= 1
                model += charge_power_kw[i, t] <= chargestates[i, t] * power_V1GUppper
                model += discharge_power_kw[i, t] <= dischargestates[i, t] * abs(power_V1GLower)
                model += chargeprofiles[i, t] == charge_power_kw[i, t] - discharge_power_kw[i, t]
                if i < len(v1g_emergency_flags) and v1g_emergency_flags[i]:
                    model += dischargestates[i, t] == 0
            else:
                model += chargeprofiles[i,t] == 0
                model += charge_power_kw[i, t] == 0
                model += discharge_power_kw[i, t] == 0
                model += chargestates[i,t] ==0
                model += dischargestates[i,t] ==0
        for idx, t in enumerate(window):
            cumulative_energy = lp.lpSum(
                (
                    charge_power_kw[i, window[k]] * charge_efficiency
                    - discharge_power_kw[i, window[k]] * (1.0 / discharge_efficiency)
                ) * SLOT_HOURS
                for k in range(idx + 1)
            )
            soc_after_t = v1g_startingSOC[i] + cumulative_energy / batteryCapacity
            model += soc_after_t <= SOC_max
            model += soc_after_t >= SOC_min
            model += soc_after_t >= SOC_reserve - (1 - dischargestates[i, t])

    for i in veh_V1G: # constraint 5: energy target (or SOC fallback)
        temp_start=v1g_startingTime[i]
        temp_end=v1g_endingTime[i]
        dwell_slots_i = int(v1g_dwell_slots[i]) if v1g_dwell_slots is not None else (
            (SLOTS_PER_DAY - temp_start) + temp_end if temp_start >= temp_end else temp_end - temp_start
        )
        window = build_window_slots(temp_start, temp_end, dwell_slots_i)
        if use_energy_targets and v1g_energy_required_kwh is not None:
            delivered_energy = lp.lpSum(
                (
                    charge_power_kw[i, tn]
                    - discharge_power_kw[i, tn] * (1.0 / (charge_efficiency * discharge_efficiency))
                ) * SLOT_HOURS
                for tn in window
            )
            if unmet_energy is not None:
                if len(window) == 0 and v1g_energy_required_kwh[i] > 1e-9:
                    print(
                        f"WARNING: EV {i} has empty charging window; all required energy "
                        f"({v1g_energy_required_kwh[i]:.2f} kWh) will be unmet."
                    )
                model += delivered_energy + unmet_energy[i] == v1g_energy_required_kwh[i]
            else:
                if len(window) == 0:
                    if v1g_energy_required_kwh[i] > 1e-9:
                        print(
                            f"WARNING: EV {i} has empty charging window but positive required energy "
                            f"({v1g_energy_required_kwh[i]:.2f} kWh)."
                        )
                    model += v1g_energy_required_kwh[i] <= 1e-9
                else:
                    model += delivered_energy == v1g_energy_required_kwh[i]
        else:
            temp_startSOC=v1g_startingSOC[i]
            if temp_start >= temp_end:
                for t in range(temp_start+1, SLOTS_PER_DAY, 1):
                    temp_timer = range (temp_start, t, 1)
                    model += temp_startSOC + lp.lpSum(
                        battery_delta_kwh(i, tn) / batteryCapacity for tn in temp_timer
                    ) <= 1
                for t in range (0, temp_end+1, 1):
                    temp_timer = range (0, t, 1)
                    model += temp_startSOC + lp.lpSum(
                        battery_delta_kwh(i, tn) / batteryCapacity
                        for tn in range(temp_start, SLOTS_PER_DAY, 1)
                    ) + lp.lpSum(
                        battery_delta_kwh(i, tn) / batteryCapacity for tn in temp_timer
                    ) <= 1
                if end_SOC ==2:
                    model += temp_startSOC + lp.lpSum(
                        battery_delta_kwh(i, tn) / batteryCapacity
                        for tn in range(temp_start, SLOTS_PER_DAY, 1)
                    ) + lp.lpSum(
                        battery_delta_kwh(i, tn) / batteryCapacity
                        for tn in range(0, temp_end)
                    ) == 1
            if temp_start < temp_end:
                 for t in range (temp_start+1, temp_end+1, 1):
                    temp_timer = range (temp_start, t, 1)
                    model += temp_startSOC + lp.lpSum(
                        battery_delta_kwh(i, tn) / batteryCapacity for tn in temp_timer
                    ) <= 1
                 if end_SOC ==2:
                    model += temp_startSOC + lp.lpSum(
                        battery_delta_kwh(i, tn) / batteryCapacity
                        for tn in range(temp_start, temp_end, 1)
                    ) == 1

                #for t in time_Interval:  # constraint 6: number of chargers available at time interval t
        #model += lp.lpSum([chargestates[i, t]] for i in veh_V1G) <= 400

    unavoidable_energy_target_kwh = 0.0
    for i in veh_V1G:
        temp_start = int(v1g_startingTime[i])
        temp_end = int(v1g_endingTime[i])
        dwell_slots_i = int(v1g_dwell_slots[i])
        if dwell_slots_i >= SLOTS_PER_DAY:
            window_length = SLOTS_PER_DAY
        elif temp_start < temp_end:
            window_length = temp_end - temp_start
        else:
            window_length = (SLOTS_PER_DAY - temp_start) + temp_end
        max_possible_kwh_i = window_length * power_V1GUppper * SLOT_HOURS
        req_i = float(v1g_energy_required_kwh[i])
        unavoidable_energy_target_kwh += max(0.0, req_i - max_possible_kwh_i)

    if unmet_energy is not None:
        model += (
            lp.lpSum(unmet_energy[i] for i in veh_V1G)
            <= unavoidable_energy_target_kwh + EPS_UNMET_KWH
        ), "service_requirement"

    # define objective function (choose one mode)
    unmet_penalty_term = 0
    if objective_mode == "A_peak":
        model += max_load, "obj_peak"
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
        if unmet_energy is None:
            raise ValueError("D_service_first_peak requires unmet_energy slack; enable use_energy_targets and ALLOW_UNMET_ENERGY.")
        model += LEXI_M * lp.lpSum(unmet_energy[i] for i in veh_V1G) + peak_ev_kw, "obj_service_first_peak"
    else:  # default: minimize spread
        model += max_load - min_load + unmet_penalty_term, "obj_spread"

    #print(model)
    log_fd, log_path = tempfile.mkstemp(prefix="cbc_", suffix=".log")
    os.close(log_fd)
    solver = lp.PULP_CBC_CMD(
        msg=CBC_MSG,
        timeLimit=CBC_TIME_LIMIT_SEC,
        gapRel=CBC_REL_GAP,
        options=["preprocess off"],
        logPath=log_path,
        keepFiles=False,
    )
    if n_V1G_Veh == 10 and objective_mode == "A_peak":
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.writeLP(os.path.join(OUTPUT_DIR, "debug_ev6.lp"))
        print("\n=== EV6 model-variable bounds before solve ===")
        for t in range(SLOTS_PER_DAY):
            if int(v1g_startingTime[6]) <= t < int(v1g_endingTime[6]):
                print(
                    f"t={t} | LB={chargeprofiles[6, t].lowBound} | "
                    f"UB={chargeprofiles[6, t].upBound}"
                )
    try:
        status = model.solve(solver)
        if n_V1G_Veh == 10 and objective_mode == "A_peak":
            print("\n=== Post-solve model validity ===")
            print("model.valid() =", model.valid(1e-7))
            bad_vars = []
            for variable in model.variables():
                value = variable.varValue
                if value is None:
                    continue
                if variable.lowBound is not None and value < variable.lowBound - 1e-6:
                    bad_vars.append((variable.name, value, variable.lowBound, variable.upBound))
                if variable.upBound is not None and value > variable.upBound + 1e-6:
                    bad_vars.append((variable.name, value, variable.lowBound, variable.upBound))
            print("Number of bound-violating variables:", len(bad_vars))
            for item in bad_vars[:20]:
                print(item)
            variable_names = [variable.name for variable in model.variables()]
            duplicate_names = sorted({name for name in variable_names if variable_names.count(name) > 1})
            print("Duplicate variable names:", duplicate_names)
            print("\n=== FULL PuLP validity diagnosis ===")
            validity_eps = 1e-7
            invalid_model_vars = []
            for variable in model.variables():
                if not variable.valid(validity_eps):
                    value = variable.varValue
                    integer_gap = None
                    if value is not None and variable.cat == "Integer":
                        integer_gap = abs(value - round(value))
                    invalid_model_vars.append(
                        (
                            variable.name,
                            value,
                            variable.lowBound,
                            variable.upBound,
                            variable.cat,
                            integer_gap,
                        )
                    )
            print("Invalid variables according to PuLP:", len(invalid_model_vars))
            for item in invalid_model_vars[:30]:
                print(item)

            invalid_constraints = []
            for name, constraint in model.constraints.items():
                if not constraint.valid(validity_eps):
                    invalid_constraints.append(
                        (name, constraint.value(), constraint.sense, constraint.slack)
                    )
            print("\nInvalid constraints according to PuLP:", len(invalid_constraints))
            for item in invalid_constraints[:50]:
                print(item)
            print("\nmodel.valid() =", model.valid(validity_eps))
            with open(
                os.path.join(OUTPUT_DIR, "validity_diagnosis.txt"),
                "w",
                encoding="utf-8",
            ) as diagnosis_file:
                diagnosis_file.write(f"model.valid({validity_eps}) = {model.valid(validity_eps)}\n\n")

                invalid_variables = []
                for variable in model.variables():
                    if not variable.valid(validity_eps):
                        value = variable.varValue
                        integer_gap = None
                        if value is not None and variable.cat == "Integer":
                            integer_gap = abs(value - round(value))
                        invalid_variables.append(
                            (
                                variable.name,
                                value,
                                variable.lowBound,
                                variable.upBound,
                                variable.cat,
                                integer_gap,
                            )
                        )
                diagnosis_file.write(f"INVALID VARIABLES: {len(invalid_variables)}\n")
                for item in invalid_variables:
                    diagnosis_file.write(str(item) + "\n")

                invalid_constraints = []
                for name, constraint in model.constraints.items():
                    if not constraint.valid(validity_eps):
                        value = constraint.value()
                        if constraint.sense == -1:
                            violation = max(0.0, value)
                        elif constraint.sense == 1:
                            violation = max(0.0, -value)
                        else:
                            violation = abs(value)
                        invalid_constraints.append(
                            (
                                violation,
                                name,
                                value,
                                constraint.slack,
                                constraint.sense,
                                str(constraint),
                            )
                        )
                invalid_constraints.sort(key=lambda item: item[0], reverse=True)
                diagnosis_file.write(f"\nINVALID CONSTRAINTS: {len(invalid_constraints)}\n")
                for violation, name, value, slack, sense, expression in invalid_constraints[:100]:
                    diagnosis_file.write(
                        f"\n{name}\n"
                        f"violation = {violation:.12g}\n"
                        f"value     = {value:.12g}\n"
                        f"slack     = {slack}\n"
                        f"sense     = {sense}\n"
                        f"{expression}\n"
                    )
            print("Saved validity diagnosis to validity_diagnosis.txt")
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                solver_log_text = fh.read()
        except OSError:
            solver_log_text = ""
    finally:
        if os.path.exists(log_path):
            try:
                os.remove(log_path)
            except PermissionError:
                # CBC may briefly retain the redirected log handle on Windows.
                pass

    solver_status = lp.LpStatus[status]
    if "Result - Stopped on time limit" in solver_log_text or "Cbc0020I Exiting on maximum time" in solver_log_text:
        solve_class = "time_limit_feasible"
    elif solver_status == "Optimal":
        solve_class = "optimal"
    elif solver_status == "Infeasible":
        solve_class = "infeasible"
    else:
        solve_class = "other"

    validity_eps = 1e-4
    invalid_variable_count = sum(
        1 for variable in model.variables() if not variable.valid(validity_eps)
    )
    invalid_constraint_count = sum(
        1 for constraint in model.constraints.values() if not constraint.valid(validity_eps)
    )
    if invalid_variable_count or invalid_constraint_count:
        print(
            f"VALIDITY WARNING | N={n_V1G_Veh} | "
            f"invalid_variables={invalid_variable_count} | "
            f"invalid_constraints={invalid_constraint_count} | "
            f"tolerance={validity_eps}"
        )

    results = {
        "status_code": int(status),
        "status": solver_status,
        "solver_status": solver_status,
        "solve_class": solve_class,
        "max_load": lp.value(max_load),
        "min_load": lp.value(min_load),
        "objective": lp.value(model.objective),
        "objective_name": getattr(model.objective, "name", "n/a"),
        "cbc_time_limit_sec": float(CBC_TIME_LIMIT_SEC),
        "cbc_gap_rel": float(CBC_REL_GAP),
        "unavoidable_energy_target_kwh": float(unavoidable_energy_target_kwh),
        "_charge_power_kw_vars": charge_power_kw,
        "_discharge_power_kw_vars": discharge_power_kw,
    }
    if objective_mode == "B_peak_above_baseline":
        results["peak_increase"] = lp.value(peak_increase)
    if unmet_energy is not None:
        unmet_vals = np.array([lp.value(unmet_energy[i]) or 0.0 for i in veh_V1G], dtype=float)
        results["total_unmet_energy_kwh"] = float(np.sum(unmet_vals))
        results["max_unmet_energy_kwh"] = float(np.max(unmet_vals)) if unmet_vals.size > 0 else 0.0
        results["unmet_ev_count"] = int(np.sum(unmet_vals > 1e-6))
    else:
        unmet_vals = np.zeros(len(list(veh_V1G)), dtype=float)
        results["total_unmet_energy_kwh"] = 0.0
        results["max_unmet_energy_kwh"] = 0.0
        results["unmet_ev_count"] = 0
    required_total_kwh = float(np.sum(v1g_energy_required_kwh)) if v1g_energy_required_kwh is not None else np.nan
    served_total_kwh = required_total_kwh - results["total_unmet_energy_kwh"] if not np.isnan(required_total_kwh) else np.nan
    served_fraction = (served_total_kwh / required_total_kwh) if (not np.isnan(required_total_kwh) and required_total_kwh > 0) else np.nan
    results["peak_ev_kw"] = float(lp.value(peak_ev_kw) or 0.0)
    results["required_total_kwh"] = required_total_kwh
    results["served_total_kwh"] = served_total_kwh
    results["served_fraction"] = served_fraction

    unavoidable_unmet_kwh = 0.0
    window_infeasible_count = 0
    for i in veh_V1G:
        temp_start = int(v1g_startingTime[i])
        temp_end = int(v1g_endingTime[i])
        dwell_slots_i = int(v1g_dwell_slots[i]) if v1g_dwell_slots is not None else (
            (SLOTS_PER_DAY - temp_start) + temp_end if temp_start >= temp_end else temp_end - temp_start
        )
        allowed_slots_i = build_window_slots(temp_start, temp_end, dwell_slots_i)
        max_possible_kwh_i = len(allowed_slots_i) * power_V1GUppper * SLOT_HOURS
        req_i = float(v1g_energy_required_kwh[i]) if v1g_energy_required_kwh is not None else 0.0
        shortfall_i = max(0.0, req_i - max_possible_kwh_i)
        if shortfall_i > EPS_UNMET_KWH:
            window_infeasible_count += 1
            unavoidable_unmet_kwh += shortfall_i
    total_unmet_kwh = float(results.get("total_unmet_energy_kwh", 0.0))
    if total_unmet_kwh + EPS_UNMET_KWH < unavoidable_unmet_kwh:
        print(
            "ERROR: total unmet is smaller than unavoidable unmet:",
            total_unmet_kwh,
            unavoidable_unmet_kwh,
        )
    avoidable_unmet_kwh = max(0.0, total_unmet_kwh - unavoidable_unmet_kwh)
    max_total_load_kw = float(np.max([
        float(lp.value(total_load[t]) or 0.0) for t in time_Interval
    ])) if len(list(time_Interval)) > 0 else 0.0
    # Under the headroom-only policy, managed EV charging is the controlled quantity.
    # If the model solved, the EV headroom constraint was respected by construction.
    limit_respected = solve_class in {"optimal", "time_limit_feasible"}
    optimality_proven_feasible = (
        solve_class == "optimal"
        and avoidable_unmet_kwh <= SERVICE_FEASIBILITY_TOLERANCE_KWH
    )
    service_feasible = (
        solve_class in {"optimal", "time_limit_feasible"}
        and limit_respected
        and avoidable_unmet_kwh <= SERVICE_FEASIBILITY_TOLERANCE_KWH
    )
    print("solve_class =", solve_class)
    print("limit_respected =", limit_respected)
    print("total_unmet_kwh =", repr(total_unmet_kwh))
    print("unavoidable_unmet_kwh =", repr(unavoidable_unmet_kwh))
    print("avoidable_unmet_kwh =", repr(avoidable_unmet_kwh))
    print("EPS_UNMET_KWH =", EPS_UNMET_KWH)
    print(
        "service_feasible components:",
        solve_class in {"optimal", "time_limit_feasible"},
        limit_respected,
        avoidable_unmet_kwh <= SERVICE_FEASIBILITY_TOLERANCE_KWH
    )
    results["unavoidable_unmet_kwh"] = float(unavoidable_unmet_kwh)
    results["avoidable_unmet_kwh"] = float(avoidable_unmet_kwh)
    results["window_infeasible_count"] = int(window_infeasible_count)
    results["max_total_load_kw"] = max_total_load_kw
    results["limit_respected"] = bool(limit_respected)
    results["service_feasible"] = bool(service_feasible)
    results["optimality_proven_feasible"] = bool(optimality_proven_feasible)
    results["solver_proven_optimal"] = bool(solve_class == "optimal")
    results["practical_feasible"] = bool(service_feasible)

    # Post-solve diagnostics to explain when outlet blocking should or should not change results.
    if results["solve_class"] in {"optimal", "time_limit_feasible"}:
        # Count active charging sessions from delivered power, not the state variable,
        # because some solves may effectively behave like an LP relaxation/presolved form.
        charging_active_matrix = np.zeros((n_V1G_Veh, SLOTS_PER_DAY), dtype=float)
        for (i, t), var in chargeprofiles.items():
            charging_active_matrix[i, t] = 1.0 if float(lp.value(var) or 0.0) > 1e-6 else 0.0
        simultaneous_charging_by_slot = np.sum(charging_active_matrix, axis=0)
        max_simultaneous_charging = float(np.max(simultaneous_charging_by_slot)) if simultaneous_charging_by_slot.size else 0.0
        min_available_internal = int(np.min(available_internal_chargers_by_slot)) if len(available_internal_chargers_by_slot) else 0
        t_max_used = int(np.argmax(simultaneous_charging_by_slot)) if simultaneous_charging_by_slot.size else -1
        used_at_t_max_used = float(simultaneous_charging_by_slot[t_max_used]) if t_max_used >= 0 else 0.0
        available_at_t_max_used = int(available_internal_chargers_by_slot[t_max_used]) if t_max_used >= 0 else 0
        if available_at_t_max_used > 0:
            utilization_max = used_at_t_max_used / float(available_at_t_max_used)
            utilization_max_str = f"{utilization_max:.3f}"
        else:
            utilization_max = np.nan
            utilization_max_str = "undefined (available_at_t_max_used = 0)"

        utilization_by_slot = np.full(SLOTS_PER_DAY, np.nan, dtype=float)
        positive_avail = np.asarray(available_internal_chargers_by_slot, dtype=float) > 0
        utilization_by_slot[positive_avail] = (
            simultaneous_charging_by_slot[positive_avail]
            / np.asarray(available_internal_chargers_by_slot, dtype=float)[positive_avail]
        )
        util_order = np.argsort(np.nan_to_num(utilization_by_slot, nan=-1.0))[::-1]
        top5_util_slots = [int(t) for t in util_order[:5]]

        print("\n=== Capacity binding proof ===")
        print(f"t_max_used = {t_max_used}")
        print(f"used_chargers[t_max_used] = {used_at_t_max_used:.3f}")
        print(f"available_internal[t_max_used] = {available_at_t_max_used}")
        print(f"utilization_max = {utilization_max_str}")
        print(f"min_available_internal = {min_available_internal}")
        print("top 5 time slots by utilization (t, used, available, utilization)")
        for t in top5_util_slots:
            used_t = float(simultaneous_charging_by_slot[t])
            avail_t = int(available_internal_chargers_by_slot[t])
            util_t = utilization_by_slot[t]
            util_t_str = f"{util_t:.3f}" if np.isfinite(util_t) else "undefined"
            print(f"  t={t}: used={used_t:.3f}, available={avail_t}, utilization={util_t_str}")

        window_slots_arr = np.zeros(n_V1G_Veh, dtype=int)
        max_possible_kwh_in_window_arr = np.zeros(n_V1G_Veh, dtype=float)
        for i in veh_V1G:
            temp_start = v1g_startingTime[i]
            temp_end = v1g_endingTime[i]
            dwell_slots_i = int(v1g_dwell_slots[i]) if v1g_dwell_slots is not None else (
                (SLOTS_PER_DAY - temp_start) + temp_end if temp_start >= temp_end else temp_end - temp_start
            )
            window = build_window_slots(temp_start, temp_end, dwell_slots_i)
            n_slots = len(window)
            window_slots_arr[i] = n_slots
            max_possible_kwh_in_window_arr[i] = n_slots * power_V1GUppper * SLOT_HOURS

        if 6 in veh_V1G:
            ev6_start = int(v1g_startingTime[6])
            ev6_end = int(v1g_endingTime[6])
            ev6_dwell = int(v1g_dwell_slots[6])
            ev6_slots = build_window_slots(ev6_start, ev6_end, ev6_dwell)
            ev6_delivered = float(sum(
                (
                    (lp.value(charge_power_kw[6, t]) or 0.0)
                    - (lp.value(discharge_power_kw[6, t]) or 0.0)
                    / (charge_efficiency * discharge_efficiency)
                ) * SLOT_HOURS
                for t in ev6_slots
            ))
            ev6_unmet = float(lp.value(unmet_energy[6]) or 0.0) if unmet_energy is not None else 0.0
            print("\n=== EV6 detailed check ===")
            print(f"start = {ev6_start}")
            print(f"end = {ev6_end}")
            print(f"required = {float(v1g_energy_required_kwh[6]):.6f}")
            print(f"SOC start = {float(v1g_startingSOC[6]):.6f}")
            print(f"allowed slots = {ev6_slots}")
            print(f"number allowed = {len(ev6_slots)}")
            print(f"delivered = {ev6_delivered:.6f}")
            print(f"unmet = {ev6_unmet:.6f}")
            print(f"max_possible = {max_possible_kwh_in_window_arr[6]:.6f}")
            print("\n=== EV6 slot-level power check ===")
            sum_p = 0.0
            for t in ev6_slots:
                power_value = float(lp.value(chargeprofiles[6, t]) or 0.0)
                charge_state = float(lp.value(chargestates[6, t]) or 0.0)
                discharge_state = float(lp.value(dischargestates[6, t]) or 0.0)
                sum_p += power_value
                print(
                    f"t={t:2d} | p={power_value:.6f} kW | "
                    f"LB={chargeprofiles[6, t].lowBound} | UB={chargeprofiles[6, t].upBound} | "
                    f"charge_state={charge_state:.0f} | discharge_state={discharge_state:.0f}"
                )
            print(f"sum power over window = {sum_p:.6f}")
            print(f"maximum possible sum = {len(ev6_slots) * power_V1GUppper:.6f}")
            print(f"net charger-equivalent energy = {ev6_delivered:.6f}")

        print("\n=== Unmet proof ===")
        print(f"total_unmet_kWh = {results['total_unmet_energy_kwh']:.3f}")

        unmet_idx = np.where(unmet_vals > 1e-6)[0]
        if unmet_idx.size > 0 and v1g_energy_required_kwh is not None:
            top_offender = int(unmet_idx[np.argmax(unmet_vals[unmet_idx])])
            req_kwh = float(v1g_energy_required_kwh[top_offender])
            max_kwh = float(max_possible_kwh_in_window_arr[top_offender])
            n_slots = int(window_slots_arr[top_offender])
            unmet_kwh = float(unmet_vals[top_offender])
            if n_slots == 0:
                reason = "empty_window"
            elif req_kwh > max_kwh + 1e-9:
                reason = "window_infeasible"
            else:
                reason = "resource_limited"
            print(
                "top_offender = "
                f"EV {top_offender} | required={req_kwh:.2f} | max_possible={max_kwh:.2f} | "
                f"unmet={unmet_kwh:.2f} | window_slots={n_slots} | reason={reason}"
            )

        print("\n=== Physics check proof ===")
        if v1g_energy_required_kwh is not None:
            window_shortfall = np.asarray(v1g_energy_required_kwh, dtype=float) - max_possible_kwh_in_window_arr
            infeasible_mask = window_shortfall > 1e-9
            count_window_infeasible = int(np.sum(infeasible_mask))
            print(f"count_window_infeasible = {count_window_infeasible}")
            if count_window_infeasible > 0:
                top_shortfall = np.where(infeasible_mask)[0]
                top_shortfall = top_shortfall[np.argsort(window_shortfall[top_shortfall])[::-1][:3]]
                print("first 3 (EV id, required_kWh, max_possible_kWh, shortfall_kWh)")
                for ev_id in top_shortfall:
                    print(
                        f"  EV {ev_id}: required={float(v1g_energy_required_kwh[ev_id]):.2f}, "
                        f"max_possible={float(max_possible_kwh_in_window_arr[ev_id]):.2f}, "
                        f"shortfall={float(window_shortfall[ev_id]):.2f}"
                    )

        # Single-EV consistency debug to validate warning vs solved-model behavior.
        ev_id = 33
        if ev_id < n_V1G_Veh and v1g_energy_required_kwh is not None:
            temp_start = int(v1g_startingTime[ev_id])
            temp_end = int(v1g_endingTime[ev_id])
            dwell_slots_i = int(v1g_dwell_slots[ev_id]) if v1g_dwell_slots is not None else (
                (SLOTS_PER_DAY - temp_start) + temp_end if temp_start >= temp_end else temp_end - temp_start
            )
            allowed_slots = build_window_slots(temp_start, temp_end, dwell_slots_i)
            delivered_kwh = float(sum(
                (
                    (lp.value(charge_power_kw[ev_id, t]) or 0.0)
                    - (lp.value(discharge_power_kw[ev_id, t]) or 0.0)
                    / (charge_efficiency * discharge_efficiency)
                ) * SLOT_HOURS
                for t in allowed_slots
            ))
            required_kwh = float(v1g_energy_required_kwh[ev_id])
            unmet_kwh = float(lp.value(unmet_energy[ev_id]) or 0.0) if unmet_energy is not None else 0.0
            max_possible_kwh = float(len(allowed_slots) * power_V1GUppper * SLOT_HOURS)
            residual = required_kwh - delivered_kwh - unmet_kwh

            print("\n=== EV33 consistency debug ===")
            print(
                f"start_slot={temp_start}, end_slot={temp_end}, "
                f"window_slots_count={len(allowed_slots)}"
            )
            print(f"allowed_slots={allowed_slots}")
            print(f"P_MAX_KW_used_in_model={float(power_V1GUppper):.3f}")
            print(f"delivered_kWh={delivered_kwh:.6f}")
            print(f"required_kWh={required_kwh:.6f}, unmet_kWh={unmet_kwh:.6f}")
            print(f"max_possible_kWh={max_possible_kwh:.6f}")
            print(f"required - delivered - unmet = {residual:.9f}")
        else:
            print(f"\nEV33 debug skipped: ev_id=33 not present for current N={n_V1G_Veh}")

    return chargeprofiles, total_load, results


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
    has_usable_solution = bool(
        results and (
            results.get("usable_solution", False)
            or results.get("objective", None) is not None
        )
    )
    if not has_usable_solution:
        print('No usable incumbent -> skip KPI reporting')
        return EV_load,opt_Totalload

    print('Optimized daily peak load: {:.1f} kW'.format(max_Load))
    print('Maximum aggregated EV charging power: {:.1f} kW'.format(float(np.max(opt_EVload))))
    print('Peak increase vs base load only: {:+.1f} kW'.format(delta_vs_base))
    print('Peak increase vs base + unmanaged EVs: {:+.1f} kW'.format(delta_vs_base_unmanaged))
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
        unmet_total_kwh = results.get("total_unmet_energy_kwh", np.nan) if results else np.nan
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"optimized_load_power{POWER_LIMIT_KW}kW.csv")
    df.to_csv(output_path, index=False)

def extract_total_load_series(total_load_vars):
    out = np.zeros(SLOTS_PER_DAY)
    for t in range(SLOTS_PER_DAY):
        out[t] = float(total_load_vars[t].varValue or 0.0)
    return out

def extract_managed_ev_series(chargeprofiles_vars, n_veh):
    out = np.zeros(SLOTS_PER_DAY)
    for i in range(n_veh):
        for t in range(SLOTS_PER_DAY):
            out[t] += float(chargeprofiles_vars[(i, t)].varValue or 0.0)
    return out

def extract_ev_power_and_soc_profiles(charge_vars, discharge_vars, n_veh):
    charge_kw = np.zeros((n_veh, SLOTS_PER_DAY), dtype=float)
    discharge_kw = np.zeros((n_veh, SLOTS_PER_DAY), dtype=float)
    soc_after_slot = np.full((n_veh, SLOTS_PER_DAY), np.nan, dtype=float)

    for i in range(n_veh):
        for t in range(SLOTS_PER_DAY):
            charge_kw[i, t] = float(charge_vars[i, t].varValue or 0.0)
            discharge_kw[i, t] = float(discharge_vars[i, t].varValue or 0.0)

        dwell_slots_i = int(v1g_dwell_slots[i])
        window = build_window_slots(
            int(v1g_startingTime[i]),
            int(v1g_endingTime[i]),
            dwell_slots_i,
        )
        current_soc = float(v1g_startingSOC[i])
        for t in window:
            battery_delta = (
                charge_kw[i, t] * charge_efficiency
                - discharge_kw[i, t] / discharge_efficiency
            ) * SLOT_HOURS
            current_soc += battery_delta / batteryCapacity
            soc_after_slot[i, t] = current_soc

    return charge_kw, discharge_kw, soc_after_slot

def has_numeric_var_values(var_dict, sample_keys=None):
    keys = sample_keys if sample_keys is not None else list(var_dict.keys())
    for key in keys:
        if key in var_dict and var_dict[key].varValue is not None:
            return True
    return False

def contiguous_hosting_capacity(df_res, flag_col):
    n_star = -1
    for _, row in df_res.sort_values("N_internal").iterrows():
        if bool(row.get(flag_col, False)):
            n_star = int(row["N_internal"])
        else:
            break
    return n_star


def min_n_for_peak_reduction(df_res, target_kw):
    mask = (
        (df_res["service_feasible"] == True)
        & (df_res["site_peak_reduction_kw"] >= target_kw)
    )
    return int(df_res.loc[mask, "N_internal"].min()) if mask.any() else -1


def probabilistic_hosting_capacity(df_combined, pass_col="service_feasible", thresholds=(0.90, 0.95, 1.00)):
    group_cols = [
        "blocked_outlets",
        "available_internal_chargers",
        "base_profile",
        "objective_mode",
        "N_internal",
    ]
    df = df_combined.copy()
    df[pass_col] = df[pass_col].astype(bool)
    probability = (
        df.groupby(group_cols, dropna=False)
        .agg(
            seeds_tested=("seed", "nunique"),
            pass_count=(pass_col, "sum"),
            pass_probability=(pass_col, "mean"),
            served_fraction_mean=("served_fraction", "mean"),
            avoidable_unmet_kwh_mean=("avoidable_unmet_kwh", "mean"),
            site_peak_reduction_kw_mean=("site_peak_reduction_kw", "mean"),
            max_discharge_kw_mean=("max_discharge_kw", "mean"),
            v2g_discharge_energy_kwh_mean=("v2g_discharge_energy_kwh", "mean"),
        )
        .reset_index()
    )

    summary_rows = []
    base_group_cols = ["blocked_outlets", "available_internal_chargers", "base_profile", "objective_mode"]
    for keys, group in probability.groupby(base_group_cols, dropna=False):
        row = dict(zip(base_group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for threshold in thresholds:
            passing = group[group["pass_probability"] >= threshold]
            pct = int(round(threshold * 100))
            row[f"HC_{pct}"] = int(passing["N_internal"].max()) if not passing.empty else -1
        row["max_n_tested"] = int(group["N_internal"].max()) if not group.empty else -1
        row["min_pass_probability"] = float(group["pass_probability"].min()) if not group.empty else np.nan
        row["mean_pass_probability"] = float(group["pass_probability"].mean()) if not group.empty else np.nan
        summary_rows.append(row)
    return probability, pd.DataFrame(summary_rows)


def run_hosting_capacity_sweep(
    n_values,
    epsilon_kw=5.0,
    day_type="weekday",
    seed=42,
    output_csv="hosting_capacity_results.csv",
    external_count_fixed=15,
    forced_k_blocked=None,
    stop_on_first_peak_violation=False,
    stop_on_first_unmet=False,
    objective_mode="A_peak",
    base_profile_name="average",
):
    global active_fleet_hash, active_sweep_seed
    global simulation_day_start
    global available_internal_chargers_by_slot
    global n_V0G_Veh, n_V1G_Veh, n_fastVeh, n_slowVeh
    global nfast_Chargers, nslow_Chargers, ngen_Chargers
    global v1g_startingTime, v1g_endingTime, v1g_startingSOC
    global v1g_energy_required_kwh, v1g_dwell_slots
    global v1g_user_ids, v1g_emergency_flags
    global slow_Load_1h_avg, fast_Load_1h_avg, gen_Load_1h_avg, unmanaged_Load

    np.random.seed(seed)
    active_sweep_seed = int(seed)

    day_map = {
        "weekday": pd.Timestamp("2026-02-16"),
        "friday": pd.Timestamp("2026-02-20"),
        "weekend": pd.Timestamp("2026-02-21"),
    }
    simulation_day_start = day_map.get(day_type, pd.Timestamp.today().normalize())
    master_internal, master_external, _, _ = ensure_master_fleet(
        seed=seed,
        day_type=day_type,
        n_internal=max(int(max(n_values)), 100) if len(n_values) > 0 else 100,
        n_external=external_count_fixed,
        battery_capacity_kwh=float(globals().get("batteryCapacity", 60.0)),
        fleet_dir=MASTER_FLEET_DIR,
        dwell_cap_minutes=get_active_dwell_cap_minutes(),
        protected_user_ids=EMERGENCY_USER_IDS,
    )
    active_fleet_hash = (
        f"{master_internal['fleet_hash'].iloc[0]}:"
        f"{master_external['fleet_hash'].iloc[0]}"
    )

    available_internal_chargers_by_slot, k_blocked = build_available_internal_chargers_all_day(
        day_start=simulation_day_start,
        n_internal_chargers=internal_chargers_available,
        slots_per_day=SLOTS_PER_DAY,
    )
    if forced_k_blocked is not None:
        # Sensitivity mode: keep everything fixed except booked internal outlets.
        k_blocked = int(max(0, min(forced_k_blocked, internal_chargers_available)))
        available_internal_chargers_by_slot = np.full(
            SLOTS_PER_DAY,
            internal_chargers_available - k_blocked,
            dtype=int,
        )

    # Keep external unmanaged demand fixed across the internal EV sweep.
    n_V0G_Veh = int(external_count_fixed)
    n_fastVeh = np.int_(np.round(n_V0G_Veh * percentFast))
    n_slowVeh = np.int_(np.round(n_V0G_Veh * (1 - percentFast)))
    n_external_total_outlets = EXTERNAL_OUTLETS

    # keep compatibility with existing unmanaged use-case internals
    nfast_backup = nfast_Chargers
    nslow_backup = nslow_Chargers
    ngen_backup = ngen_Chargers if "ngen_Chargers" in globals() else None
    if useCase == 4:
        ngen_Chargers = n_external_total_outlets
    else:
        nfast_Chargers = int(round(n_external_total_outlets * percentFast))
        nslow_Chargers = int(n_external_total_outlets - nfast_Chargers)

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

    # restore global charger counts after external load generation
    nfast_Chargers = nfast_backup
    nslow_Chargers = nslow_backup
    if ngen_backup is not None:
        ngen_Chargers = ngen_backup

    # Hospital baseline for KPI reporting (independent of unmanaged EVs).
    baseline_series = base_Load - (solarCurve if opt_includeSolar else 0)
    baseline_peak = float(np.max(baseline_series))
    # No-managed total (base + fixed external unmanaged EV load) used for N=0 reference only.
    total_no_managed = base_Load + unmanaged_Load - (solarCurve if opt_includeSolar else 0)
    external_waited = avgWaitedGen if avgWaitedGen >= 0 else (avgWaitedFast if avgWaitedFast >= 0 else avgWaitedSlow)
    external_uncharged = avgUnchargedGen if avgUnchargedGen >= 0 else (avgUnchargedFast if avgUnchargedFast >= 0 else avgUnchargedSlow)

    n_max_internal = max(int(max(n_values)), 100) if len(n_values) > 0 else 100
    profile_shape = (len(n_values), n_max_internal, SLOTS_PER_DAY)
    profile_charge_kw = np.full(profile_shape, np.nan, dtype=float)
    profile_discharge_kw = np.full(profile_shape, np.nan, dtype=float)
    profile_soc_after_slot = np.full(profile_shape, np.nan, dtype=float)
    profile_initial_soc = np.full((len(n_values), n_max_internal), np.nan, dtype=float)
    profile_total_load_kw = np.full((len(n_values), SLOTS_PER_DAY), np.nan, dtype=float)
    profile_status = np.full(len(n_values), "not_solved", dtype="U32")
    rows = []
    fixed_start = np.array([], dtype=int)
    fixed_end = np.array([], dtype=int)
    fixed_soc = np.array([], dtype=float)
    fixed_energy = np.array([], dtype=float)
    fixed_dwell = np.array([], dtype=int)
    fixed_user_ids = np.array([], dtype=str)
    fixed_emergency_flags = np.array([], dtype=bool)
    if n_max_internal > 0:
        initiate_V1G(n_max_internal)
        assert len(v1g_startingTime) == n_max_internal, "v1g_startingTime length mismatch"
        assert len(v1g_endingTime) == n_max_internal, "v1g_endingTime length mismatch"
        assert len(v1g_startingSOC) == n_max_internal, "v1g_startingSOC length mismatch"
        assert len(v1g_energy_required_kwh) == n_max_internal, "v1g_energy_required_kwh length mismatch"
        assert len(v1g_dwell_slots) == n_max_internal, "v1g_dwell_slots length mismatch"
        fixed_start = np.array(v1g_startingTime, dtype=int).copy()
        fixed_end = np.array(v1g_endingTime, dtype=int).copy()
        fixed_soc = np.array(v1g_startingSOC, dtype=float).copy()
        fixed_energy = np.array(v1g_energy_required_kwh, dtype=float).copy()
        fixed_dwell = np.array(v1g_dwell_slots, dtype=int).copy()
        fixed_user_ids = np.array(v1g_user_ids, dtype=str).copy()
        fixed_emergency_flags = np.array(v1g_emergency_flags, dtype=bool).copy()

        print("\n=== Check physical validity of sampled fleet ===")
        bad = np.where(
            (fixed_soc < -1e-9)
            | (fixed_soc > 1.0 + 1e-9)
            | (fixed_energy > batteryCapacity + 1e-9)
        )[0]
        print("Invalid EV indices:", bad.tolist())
        for i in bad:
            print(
                f"EV {i}: "
                f"E_req={fixed_energy[i]:.3f} kWh | "
                f"SOC_start={fixed_soc[i]:.4f} | "
                f"start={fixed_start[i]} | "
                f"end={fixed_end[i]} | "
                f"dwell_slots={fixed_dwell[i]}"
            )

        print("\n=== EVs added from N=65 to N=70 ===")
        for i in range(65, min(70, n_max_internal)):
            print(
                f"EV {i}: "
                f"E_req={fixed_energy[i]:.3f} kWh | "
                f"SOC_start={fixed_soc[i]:.4f} | "
                f"start={fixed_start[i]} | "
                f"end={fixed_end[i]} | "
                f"dwell_slots={fixed_dwell[i]}"
            )

    sweep_objective_mode = objective_mode
    print(f"Sweep start | day_type={day_type} | k_blocked={k_blocked} | hospital_baseline_peak={baseline_peak:.2f} kW | objective={sweep_objective_mode}")
    for n_index, n_internal in enumerate(n_values):
        n_V1G_Veh = int(n_internal)
        if n_V1G_Veh <= 0:
            v1g_user_ids = np.array([], dtype=str)
            v1g_emergency_flags = np.array([], dtype=bool)
            profile_total_load_kw[n_index] = total_no_managed
            profile_status[n_index] = "baseline"
            rows.append({
                "base_profile": base_profile_name,
                "power_limit_kw": POWER_LIMIT_KW,
                "N_internal": int(n_internal),
                "status": "Optimal",
                "solver_status": "Optimal",
                "solve_class": "optimal",
                "usable_solution": True,
                "feasible": True,
                "service_feasible": True,
                "optimality_proven_feasible": True,
                "solver_proven_optimal": True,
                "practical_feasible": True,
                "objective_mode": sweep_objective_mode,
                "objective_value": 0.0,
                "objective_name": f"obj_{sweep_objective_mode}",
                "peak_increase_obj_kw": np.nan,
                "peak_increase_measured_kw": float(np.max(total_no_managed - baseline_series)),
                "optimized_peak_kw": float(np.max(total_no_managed)),
                "site_peak_reduction_kw": float(baseline_peak - np.max(total_no_managed)),
                "max_total_load_kw": float(np.max(total_no_managed)),
                "managed_peak_kw": 0.0,
                "max_discharge_kw": 0.0,
                "v2g_discharge_energy_kwh": 0.0,
                "managed_peak_obj_kw": 0.0,
                "baseline_peak_kw": baseline_peak,
                "k_blocked": int(k_blocked),
                "avail_internal_min": int(np.min(available_internal_chargers_by_slot)),
                "avail_internal_max": int(np.max(available_internal_chargers_by_slot)),
                "external_waited_avg": float(external_waited),
                "external_uncharged_avg": float(external_uncharged),
                "total_unmet_energy_kwh": 0.0,
                "unavoidable_unmet_kwh": 0.0,
                "avoidable_unmet_kwh": 0.0,
                "max_unmet_energy_kwh": 0.0,
                "unmet_ev_count": 0,
                "window_infeasible_count": 0,
                "required_total_kwh": 0.0,
                "served_total_kwh": 0.0,
                "served_fraction": 1.0,
                "limit_respected": True,
            })
            print(f"N={n_internal:>3} | feasible=True | peak_inc=0.00 kW")
            continue

        # Deterministic cumulative subsets: N uses first N vehicles from one fixed sample.
        v1g_startingTime = fixed_start[:n_V1G_Veh].copy()
        v1g_endingTime = fixed_end[:n_V1G_Veh].copy()
        v1g_startingSOC = fixed_soc[:n_V1G_Veh].copy()
        v1g_energy_required_kwh = fixed_energy[:n_V1G_Veh].copy()
        v1g_dwell_slots = fixed_dwell[:n_V1G_Veh].copy()
        v1g_user_ids = fixed_user_ids[:n_V1G_Veh].copy()
        v1g_emergency_flags = fixed_emergency_flags[:n_V1G_Veh].copy()
        chargeprofiles, total_load, results = V1G_optimization(objective_mode=sweep_objective_mode)
        status = results.get("status", "Unknown")
        solver_status = results.get("solver_status", status)
        solve_class = results.get("solve_class", "other")
        optimal = (solve_class == "optimal")
        has_incumbent = bool(
            results.get("objective", None) is not None
            and has_numeric_var_values(
                total_load,
                sample_keys=[0, SLOTS_PER_DAY // 2, SLOTS_PER_DAY - 1],
            )
        )
        usable_solution = has_incumbent and solve_class in ["optimal", "time_limit_feasible"]
        service_feasible = bool(results.get("service_feasible", False)) if usable_solution else False
        optimality_proven_feasible = bool(results.get("optimality_proven_feasible", False)) if usable_solution else False
        solver_proven_optimal = bool(results.get("solver_proven_optimal", False)) if usable_solution else False
        practical_feasible = service_feasible

        if usable_solution:
            total_series = extract_total_load_series(total_load)
            managed_series = extract_managed_ev_series(chargeprofiles, n_V1G_Veh)
            charge_kw_i, discharge_kw_i, soc_after_i = extract_ev_power_and_soc_profiles(
                results["_charge_power_kw_vars"],
                results["_discharge_power_kw_vars"],
                n_V1G_Veh,
            )
            profile_charge_kw[n_index, :n_V1G_Veh] = charge_kw_i
            profile_discharge_kw[n_index, :n_V1G_Veh] = discharge_kw_i
            profile_soc_after_slot[n_index, :n_V1G_Veh] = soc_after_i
            profile_initial_soc[n_index, :n_V1G_Veh] = v1g_startingSOC
            profile_total_load_kw[n_index] = total_series
            profile_status[n_index] = solve_class
            optimized_peak = float(np.max(total_series))
            site_peak_reduction_kw = float(baseline_peak - optimized_peak)
            peak_increase_measured = float(np.max(total_series - baseline_series))
            objective_value = float(results.get("objective", np.nan))
            peak_increase_obj = np.nan
            managed_peak = float(np.max(managed_series))
            max_discharge_kw = float(max(0.0, -np.min(managed_series)))
            v2g_discharge_energy_kwh = float(np.sum(np.maximum(-managed_series, 0.0)) * SLOT_HOURS)
            managed_peak_obj = float(results.get("peak_ev_kw", np.nan))
            total_unmet_energy_kwh = float(results.get("total_unmet_energy_kwh", 0.0))
            unavoidable_unmet_kwh = float(results.get("unavoidable_unmet_kwh", 0.0))
            avoidable_unmet_kwh = float(results.get("avoidable_unmet_kwh", 0.0))
            max_unmet_energy_kwh = float(results.get("max_unmet_energy_kwh", 0.0))
            unmet_ev_count = int(results.get("unmet_ev_count", 0))
            window_infeasible_count = int(results.get("window_infeasible_count", 0))
            required_total_kwh = float(results.get("required_total_kwh", np.nan))
            served_total_kwh = float(results.get("served_total_kwh", np.nan))
            served_fraction = float(results.get("served_fraction", np.nan))
            max_total_load_kw = float(results.get("max_total_load_kw", optimized_peak))
            limit_respected = bool(results.get("limit_respected", False))
        else:
            profile_status[n_index] = solve_class
            optimized_peak = np.nan
            site_peak_reduction_kw = np.nan
            peak_increase_measured = np.nan
            objective_value = np.nan
            peak_increase_obj = np.nan
            managed_peak = np.nan
            max_discharge_kw = np.nan
            v2g_discharge_energy_kwh = np.nan
            managed_peak_obj = np.nan
            total_unmet_energy_kwh = np.nan
            unavoidable_unmet_kwh = np.nan
            avoidable_unmet_kwh = np.nan
            max_unmet_energy_kwh = np.nan
            unmet_ev_count = np.nan
            window_infeasible_count = np.nan
            required_total_kwh = np.nan
            served_total_kwh = np.nan
            served_fraction = np.nan
            max_total_load_kw = np.nan
            limit_respected = False

        rows.append({
            "base_profile": base_profile_name,
            "power_limit_kw": POWER_LIMIT_KW,
            "N_internal": int(n_internal),
            "status": status,
            "solver_status": solver_status,
            "solve_class": solve_class,
            "optimal": bool(optimal),
            "usable_solution": bool(usable_solution),
            "service_feasible": bool(service_feasible),
            "optimality_proven_feasible": bool(optimality_proven_feasible),
            "solver_proven_optimal": bool(solver_proven_optimal),
            "practical_feasible": bool(practical_feasible),
            "objective_mode": sweep_objective_mode,
            "objective_value": objective_value,
            "objective_name": results.get("objective_name", "n/a"),
            "peak_increase_obj_kw": peak_increase_obj,
            "peak_increase_measured_kw": peak_increase_measured,
            "optimized_peak_kw": optimized_peak,
            "site_peak_reduction_kw": site_peak_reduction_kw,
            "max_total_load_kw": max_total_load_kw,
            "managed_peak_kw": managed_peak,
            "max_discharge_kw": max_discharge_kw,
            "v2g_discharge_energy_kwh": v2g_discharge_energy_kwh,
            "managed_peak_obj_kw": managed_peak_obj,
            "baseline_peak_kw": baseline_peak,
            "k_blocked": int(k_blocked),
            "avail_internal_min": int(np.min(available_internal_chargers_by_slot)),
            "avail_internal_max": int(np.max(available_internal_chargers_by_slot)),
            "external_waited_avg": float(external_waited),
            "external_uncharged_avg": float(external_uncharged),
            "total_unmet_energy_kwh": total_unmet_energy_kwh,
            "unavoidable_unmet_kwh": unavoidable_unmet_kwh,
            "avoidable_unmet_kwh": avoidable_unmet_kwh,
            "max_unmet_energy_kwh": max_unmet_energy_kwh,
            "unmet_ev_count": unmet_ev_count,
            "window_infeasible_count": window_infeasible_count,
            "required_total_kwh": required_total_kwh,
            "served_total_kwh": served_total_kwh,
            "served_fraction": served_fraction,
            "limit_respected": bool(limit_respected),
        })
        if np.isnan(total_unmet_energy_kwh):
            unmet_str = "NA"
        else:
            unmet_str = f"{total_unmet_energy_kwh:.2f}"
        if np.isnan(avoidable_unmet_kwh):
            avoidable_unmet_str = "NA"
        else:
            avoidable_unmet_str = f"{avoidable_unmet_kwh:.2f}"
        print(
            f"N={n_internal:>3} | solve_class={solve_class} | peak_inc={peak_increase_measured:.2f} kW "
            f"| unmet={unmet_str} kWh | avoidable_unmet={avoidable_unmet_str} kWh "
            f"| service_feasible={service_feasible} | proven_optimal={solver_proven_optimal} "
            f"| usable_solution={usable_solution} | status={status}"
        )
        if usable_solution and stop_on_first_unmet and total_unmet_energy_kwh > 1e-6:
            print(f"Early stop: unmet energy became positive at N={n_internal}.")
            break
        if usable_solution and stop_on_first_peak_violation and peak_increase_measured > epsilon_kw + 1e-9:
            print(f"Early stop: peak increase exceeded epsilon={epsilon_kw:.2f} kW at N={n_internal}.")
            break

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    profile_window_mask = np.zeros((n_max_internal, SLOTS_PER_DAY), dtype=bool)
    for i in range(len(fixed_start)):
        for t in build_window_slots(
            int(fixed_start[i]),
            int(fixed_end[i]),
            int(fixed_dwell[i]),
        ):
            profile_window_mask[i, t] = True
    profile_path = os.path.join(
        OUTPUT_DIR,
        f"profiles_power{POWER_LIMIT_KW}kW_seed{seed}_blocked{k_blocked}_"
        f"{base_profile_name}_{sweep_objective_mode}.npz",
    )
    np.savez_compressed(
        profile_path,
        n_internal=np.asarray(n_values, dtype=int),
        slot_start_minute=np.arange(SLOTS_PER_DAY, dtype=int) * SLOT_MINUTES,
        charge_power_kw=profile_charge_kw,
        discharge_power_kw=profile_discharge_kw,
        net_power_kw=profile_charge_kw - profile_discharge_kw,
        soc_after_slot=profile_soc_after_slot,
        initial_soc=profile_initial_soc,
        total_load_kw=profile_total_load_kw,
        base_load_kw=np.asarray(base_Load, dtype=float),
        unmanaged_load_kw=np.asarray(unmanaged_Load, dtype=float),
        available_internal_chargers=np.asarray(available_internal_chargers_by_slot, dtype=int),
        window_mask=profile_window_mask,
        emergency_vehicle=np.asarray(fixed_emergency_flags, dtype=bool),
        arrival_slot=fixed_start,
        dwell_slots=fixed_dwell,
        energy_required_kwh=fixed_energy,
        solve_status=profile_status,
        seed=np.asarray(seed),
        day_type=np.asarray(day_type),
        base_profile=np.asarray(base_profile_name),
        objective_mode=np.asarray(sweep_objective_mode),
        power_limit_kw=np.asarray(POWER_LIMIT_KW),
        internal_outlets=np.asarray(internal_chargers_available),
        external_outlets=np.asarray(EXTERNAL_OUTLETS),
        charge_efficiency=np.asarray(charge_efficiency),
        discharge_efficiency=np.asarray(discharge_efficiency),
        energy_target_basis=np.asarray(ENERGY_TARGET_BASIS),
        soc_min=np.asarray(SOC_min),
        soc_max=np.asarray(SOC_max),
        discharge_reserve=np.asarray(SOC_reserve),
        fleet_hash=np.asarray(active_fleet_hash),
    )
    print("15-minute power/SoC profiles:", profile_path)

    df_res = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if not os.path.isabs(output_csv) and output_csv == "hosting_capacity_results.csv":
            output_csv = f"hosting_capacity_results_power{POWER_LIMIT_KW}kW.csv"
        out_path = output_csv if os.path.isabs(output_csv) else os.path.join(OUTPUT_DIR, output_csv)
        df_res.to_csv(out_path, index=False)
    else:
        out_path = None

    optimal_mask = df_res["optimal"] == True
    usable_mask = df_res["usable_solution"] == True
    peak_mask = usable_mask & (df_res["peak_increase_measured_kw"] <= epsilon_kw)
    service_mask = df_res["service_feasible"] == True
    practical_mask = df_res["practical_feasible"] == True
    n_star_optimal = int(df_res.loc[optimal_mask, "N_internal"].max()) if optimal_mask.any() else -1
    n_star_usable = int(df_res.loc[usable_mask, "N_internal"].max()) if usable_mask.any() else -1
    n_star_peak = int(df_res.loc[peak_mask, "N_internal"].max()) if peak_mask.any() else -1
    n_star_service_any = int(df_res.loc[service_mask, "N_internal"].max()) if service_mask.any() else -1
    n_star_service_contiguous = contiguous_hosting_capacity(df_res, "service_feasible")
    n_star_proven_contiguous = contiguous_hosting_capacity(df_res, "optimality_proven_feasible")
    n_star_practical = int(df_res.loc[practical_mask, "N_internal"].max()) if practical_mask.any() else -1
    n_star_final = n_star_service_contiguous

    print("Sweep finished")
    print(f"N*_optimal: {n_star_optimal}")
    print(f"N*_usable: {n_star_usable}")
    print(f"N*_peak(epsilon={epsilon_kw} kW): {n_star_peak}")
    print(f"N*_service_any: {n_star_service_any}")
    print(f"N*_service_contiguous: {n_star_service_contiguous}")
    print(f"N*_proven_contiguous: {n_star_proven_contiguous}")
    print(f"N*_practical: {n_star_practical}")
    print(f"N*_final: {n_star_final}")
    if out_path:
        print("Results CSV:", out_path)

    return df_res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run corrected V2G hospital hosting-capacity sweep.")
    parser.add_argument(
        "--power-kw",
        type=int,
        choices=POWER_CONFIGURATIONS_KW,
        default=11,
        help="Symmetric charger-side charge/discharge rating (11 or 22 kW).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--day-type",
        choices=("weekday", "friday", "weekend"),
        default="weekday",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("A_peak", "B_peak_above_baseline"),
        default="A_peak",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--master-fleet-dir", default=MASTER_FLEET_DIR)
    args = parser.parse_args()
    POWER_LIMIT_KW = args.power_kw
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    MASTER_FLEET_DIR = os.path.abspath(args.master_fleet_dir)
    if not EMERGENCY_USER_IDS:
        print(
            "WARNING: EMERGENCY_USER_IDS is empty; emergency-vehicle V2G protection "
            "will activate only after authoritative user_id values are configured."
        )

    useCase= 4 # 1:residential, 2:office, 3: commercial, 4: user defined
    active_cap = get_active_dwell_cap_minutes()
    if active_cap is None:
        print("Dwell scenario:", DWELL_SCENARIO, "| upper dwell cap: none")
    else:
        print("Dwell scenario:", DWELL_SCENARIO, "| upper dwell cap (min):", active_cap)
    batteryCapacity = 60
    energyEfficiency = 90  # energy consumption per 100km travelled
    SOC_min = 0.0
    SOC_reserve = 0.5
    SOC_max = 1.0

    isManaged=1 #unmanaged charging=0 or managed charging=1
    power_V1GLower = -POWER_LIMIT_KW
    power_V1GUppper = POWER_LIMIT_KW
    # site hardware counts
    internal_chargers_available = 17  # controllable internal outlets; 17 + 5 external = 22 total
    public_chargers_available = EXTERNAL_OUTLETS

    # Daily internal/external counts (managed = internal)
    n_internal_today = sample_internal_daily_count()
    n_external_today = 15  # unmanaged/external vehicles (adjust if needed)
    n_vehicles = n_internal_today + n_external_today
    percentV1G = n_internal_today / n_vehicles if n_vehicles > 0 else 0

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

    # Optional: override base load from hospital CSV with reusable profiles.
    base_Load_from_file = None
    base_load_profiles = {}
    hospital_load_file = "hospital_data_with_weekday_weekend.csv"
    base_load_mode = "average"  # options: "average", "worst_peak", "specific_date"
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

        if base_load_mode == "average":
            base_Load_from_file = base_load_profiles["average"]
        elif base_load_mode == "worst_peak":
            base_Load_from_file = base_load_profiles["worst_peak"]
        elif base_load_mode == "specific_date" and base_load_specific_date:
            target_date = pd.to_datetime(base_load_specific_date).date()
            df_day = df_load[df_load["Timestamp"].dt.date == target_date]
            if not df_day.empty:
                slot_avg = df_day.groupby("slot")["hospitalpower_kw"].mean()
                base_Load_from_file = slot_avg.reindex(range(SLOTS_PER_DAY)).fillna(0).to_numpy()
                base_load_profiles["specific_date"] = base_Load_from_file.copy()
            else:
                base_Load_from_file = None
    except FileNotFoundError:
        base_Load_from_file = None

    if useCase==1:
        base_Load = np.array([1578.7, 1414.7, 1290.1, 1258.6, 1199.2, 1279.8, 1327.9, 1378, 1492.4, 1666.8, 1738.2, 1497.9, 1433.9,
                     1446.8, 1463.8, 1434.5, 1523.9, 1651.1, 1727.2, 1922, 2162.6, 2192.6, 1944.4, 1762.9])
    if useCase==2:
        base_Load =np.array([1043.071385, 1009.268146, 980.293941, 985.1229752, 973.0503897, 1011.682663, 1294.181163, 1675.674865, 2087.350029,2290.169466, 2391.579184, 2125.982303,2237.050089, 1812.095081, 1746.903119,1733.623275, 1717.928914, 1800.022495,1558.570785, 1302.631973, 1154.139172,981.5011995, 874.0551888,	802.8269344])
    if useCase==3:
        base_Load=np.array([726.1660166,503.4268145,449.1001799,443.6675164,449.1001799,488.9397119,467.2090581,599.4038691,918.1201258,2147.712957,3290.383173,3451.552189,3536.663916,3507.689711,3324.790041,3047.724204,3054.967756,3281.328733,3348.331583,3422.577984,3409.901769,3248.732753,2692.790191,1774.670066])
    if useCase==4:
        base_Load=np.zeros(SLOTS_PER_DAY) #user defined

    if base_Load_from_file is not None:
        base_Load = base_Load_from_file
    else:
        # expand 24-element base load to 96 slots by repeating each hour into four 15-min slots
        if base_Load.size == 24:
            base_Load = np.repeat(base_Load, 60 // SLOT_MINUTES)
    if not base_load_profiles:
        base_load_profiles["average"] = np.array(base_Load, dtype=float).copy()
        base_load_profiles["worst_peak"] = np.array(base_Load, dtype=float).copy()

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
    charge_efficiency = CHARGE_EFFICIENCY
    discharge_efficiency = DISCHARGE_EFFICIENCY
    maxEfficiency=0.8 #maximual capacity of transformer
    n_iter=25   #number of iterations for the unmanaged model
    Maxtime_Interval=1440 #time interval for unmanaged model
    isSlow=1

    batteryRange = batteryCapacity / energyEfficiency * 100
    n_V1G_Veh=np.int_(np.round(n_vehicles*percentV1G))
    n_V0G_Veh=np.int_(np.round(n_vehicles*(1-percentV1G)))
    # Use physical internal charger count for site availability (not EV count).
    nV1G_Chargers = internal_chargers_available
    day_start = pd.Timestamp.today().normalize()
    simulation_day_start = day_start
    available_internal_chargers_by_slot, k_blocked = build_available_internal_chargers_all_day(
        day_start=day_start,
        n_internal_chargers=nV1G_Chargers,
        slots_per_day=SLOTS_PER_DAY,
    )
    print("Midnight blocked internal outlets (k:):", k_blocked, "| day type:", _day_type(day_start))
    n_fastVeh=np.int_(np.round(n_V0G_Veh*percentFast))
    n_slowVeh=np.int_(np.round(n_V0G_Veh*(1-percentFast)))
    if nfast_Chargers> n_fastVeh:
        nfast_Chargers=n_fastVeh
    if nslow_Chargers > n_slowVeh:
        nslow_Chargers=n_slowVeh
    # if ngen_Chargers > n_fastVeh: #add this line
    #     ngen_Chargers=n_fastVeh #add this line

    if percentFast>0:
        isSlow=2
    slow_Load_1h_avg = np.zeros(SLOTS_PER_DAY)
    fast_Load_1h_avg = np.zeros(SLOTS_PER_DAY)
    gen_Load_1h_avg = np.zeros(SLOTS_PER_DAY)
    unmanaged_Load=np.zeros(SLOTS_PER_DAY)

    run_hosting_sweep = True
    if run_hosting_sweep:
        sweep_values = list(range(0, 101, 5))
        sweep_epsilon_kw = 5.0
        sweep_day_type = args.day_type
        sweep_seeds = list(args.seeds)
        sweep_objectives = [args.objective_mode]
        sweep_base_profiles = ["average", "worst_peak"]
        sweep_blocked_outlets = [0, 2, 4, 6, 8, 10, 12, 14]
        combined_results = []
        summary_rows = []
        base_load_original = np.array(base_Load, dtype=float).copy()
        for sweep_seed in sweep_seeds:
            print("\n" + "=" * 72)
            print(f"Hosting-capacity sweep seed: {sweep_seed}")
            for blocked_outlets in sweep_blocked_outlets:
                available_internal_chargers = max(0, internal_chargers_available - blocked_outlets)
                print("\n" + "=" * 72)
                print(f"Blocked outlets: {blocked_outlets} | Available internal chargers: {available_internal_chargers}")
                for base_profile_name in sweep_base_profiles:
                    if base_profile_name not in base_load_profiles:
                        continue
                    base_Load = np.array(base_load_profiles[base_profile_name], dtype=float).copy()
                    print("\n" + "#" * 72)
                    print(f"Sensitivity base profile: {base_profile_name}")
                    for sweep_objective_mode in sweep_objectives:
                        df_case = run_hosting_capacity_sweep(
                            n_values=sweep_values,
                            epsilon_kw=sweep_epsilon_kw,
                            day_type=sweep_day_type,
                            seed=sweep_seed,
                            output_csv=None,
                            external_count_fixed=n_external_today,
                            forced_k_blocked=blocked_outlets,
                            objective_mode=sweep_objective_mode,
                            base_profile_name=base_profile_name,
                        )
                        df_case["seed"] = sweep_seed
                        df_case["blocked_outlets"] = blocked_outlets
                        df_case["available_internal_chargers"] = available_internal_chargers
                        combined_results.append(df_case)

                        service_mask = df_case["service_feasible"] == True
                        proven_mask = df_case["optimality_proven_feasible"] == True
                        hosting_capacity_any_pass = int(df_case.loc[service_mask, "N_internal"].max()) if service_mask.any() else -1
                        hosting_capacity = contiguous_hosting_capacity(df_case, "service_feasible")
                        hosting_capacity_proven = contiguous_hosting_capacity(df_case, "optimality_proven_feasible")
                        min_n_peak_target = min_n_for_peak_reduction(df_case, TARGET_PEAK_REDUCTION_KW)
                        target_mask = (
                            (df_case["service_feasible"] == True)
                            & (df_case["site_peak_reduction_kw"] >= TARGET_PEAK_REDUCTION_KW)
                        )
                        df_target = df_case.loc[df_case["N_internal"] == min_n_peak_target].iloc[-1] if min_n_peak_target >= 0 else None
                        highest_n = int(df_case["N_internal"].max()) if not df_case.empty else -1
                        df_highest = df_case.loc[df_case["N_internal"] == highest_n].iloc[-1] if highest_n >= 0 else None
                        summary_rows.append({
                            "seed": int(sweep_seed),
                            "power_limit_kw": int(POWER_LIMIT_KW),
                            "blocked_outlets": int(blocked_outlets),
                            "available_internal_chargers": int(available_internal_chargers),
                            "base_profile": base_profile_name,
                            "objective_mode": sweep_objective_mode,
                            "hosting_capacity_ev": hosting_capacity,
                            "hosting_capacity_ev_any_pass": hosting_capacity_any_pass,
                            "hosting_capacity_ev_proven_contiguous": hosting_capacity_proven,
                            "service_feasible_cases": int(service_mask.sum()),
                            "optimality_proven_feasible_cases": int(proven_mask.sum()),
                            "highest_n_evaluated": highest_n,
                            "peak_reduction_target_kw": float(TARGET_PEAK_REDUCTION_KW),
                            "min_n_for_target_peak_reduction": min_n_peak_target,
                            "target_peak_reduction_achieved": bool(target_mask.any()),
                            "max_site_peak_reduction_kw": float(df_case["site_peak_reduction_kw"].max()) if not df_case.empty else np.nan,
                            "baseline_peak_kw": float(df_highest["baseline_peak_kw"]) if df_highest is not None else np.nan,
                            "optimized_peak_kw": float(df_highest["optimized_peak_kw"]) if df_highest is not None else np.nan,
                            "site_peak_reduction_kw": float(df_highest["site_peak_reduction_kw"]) if df_highest is not None else np.nan,
                            "peak_increase_kw": float(df_highest["peak_increase_measured_kw"]) if df_highest is not None else np.nan,
                            "max_discharge_kw": float(df_highest["max_discharge_kw"]) if df_highest is not None else np.nan,
                            "v2g_discharge_energy_kwh": float(df_highest["v2g_discharge_energy_kwh"]) if df_highest is not None else np.nan,
                            "target_n_max_discharge_kw": float(df_target["max_discharge_kw"]) if df_target is not None else np.nan,
                            "target_n_v2g_discharge_energy_kwh": float(df_target["v2g_discharge_energy_kwh"]) if df_target is not None else np.nan,
                            "total_unmet_at_highest_n_kwh": float(df_highest["total_unmet_energy_kwh"]) if df_highest is not None else np.nan,
                            "unavoidable_unmet_at_highest_n_kwh": float(df_highest["unavoidable_unmet_kwh"]) if df_highest is not None else np.nan,
                            "avoidable_unmet_at_highest_n_kwh": float(df_highest["avoidable_unmet_kwh"]) if df_highest is not None else np.nan,
                        })
        base_Load = base_load_original
        if combined_results:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            df_combined = pd.concat(combined_results, ignore_index=True)
            power_prefix = f"power{POWER_LIMIT_KW}kW_stage1_corrected"
            combined_output = os.path.join(OUTPUT_DIR, f"hosting_capacity_results_{power_prefix}.csv")
            df_combined.to_csv(combined_output, index=False)
            df_probability, df_probability_summary = probabilistic_hosting_capacity(df_combined)
            probability_output = os.path.join(OUTPUT_DIR, f"hosting_capacity_probability_{power_prefix}.csv")
            probability_summary_output = os.path.join(OUTPUT_DIR, f"hosting_capacity_probability_summary_{power_prefix}.csv")
            df_probability.to_csv(probability_output, index=False)
            df_probability_summary.to_csv(probability_summary_output, index=False)
            df_summary_seed = pd.DataFrame(summary_rows)
            summary_output = os.path.join(OUTPUT_DIR, f"hosting_capacity_summary_{power_prefix}.csv")
            df_summary_seed.to_csv(summary_output, index=False)
            agg_metrics = [
                "hosting_capacity_ev",
                "hosting_capacity_ev_any_pass",
                "hosting_capacity_ev_proven_contiguous",
                "service_feasible_cases",
                "optimality_proven_feasible_cases",
                "min_n_for_target_peak_reduction",
                "max_site_peak_reduction_kw",
                "baseline_peak_kw",
                "optimized_peak_kw",
                "site_peak_reduction_kw",
                "peak_increase_kw",
                "max_discharge_kw",
                "v2g_discharge_energy_kwh",
                "target_n_max_discharge_kw",
                "target_n_v2g_discharge_energy_kwh",
                "total_unmet_at_highest_n_kwh",
                "unavoidable_unmet_at_highest_n_kwh",
                "avoidable_unmet_at_highest_n_kwh",
            ]
            df_summary = (
                df_summary_seed.groupby(
                    [
                        "power_limit_kw",
                        "blocked_outlets",
                        "available_internal_chargers",
                        "base_profile",
                        "objective_mode",
                    ],
                    dropna=False,
                )[agg_metrics]
                .agg(["mean", "std", "min", "max"])
                .reset_index()
            )
            df_summary.columns = [
                "_".join(col).strip("_") if isinstance(col, tuple) else col
                for col in df_summary.columns.to_flat_index()
            ]
            agg_summary_output = os.path.join(OUTPUT_DIR, f"hosting_capacity_summary_aggregated_{power_prefix}.csv")
            df_summary.to_csv(agg_summary_output, index=False)
            print("\nCombined sensitivity summary by seed:")
            print(
                df_summary_seed[
                    [
                        "seed",
                        "blocked_outlets",
                        "available_internal_chargers",
                        "base_profile",
                        "objective_mode",
                        "hosting_capacity_ev",
                        "hosting_capacity_ev_any_pass",
                        "hosting_capacity_ev_proven_contiguous",
                        "min_n_for_target_peak_reduction",
                        "max_site_peak_reduction_kw",
                        "baseline_peak_kw",
                        "optimized_peak_kw",
                        "site_peak_reduction_kw",
                        "peak_increase_kw",
                        "max_discharge_kw",
                        "v2g_discharge_energy_kwh",
                        "total_unmet_at_highest_n_kwh",
                        "unavoidable_unmet_at_highest_n_kwh",
                        "avoidable_unmet_at_highest_n_kwh",
                    ]
                ].to_string(index=False)
            )
            print("\nAggregated hosting-capacity summary:")
            print(
                df_summary[
                    [
                        "blocked_outlets",
                        "available_internal_chargers",
                        "base_profile",
                        "objective_mode",
                        "hosting_capacity_ev_mean",
                        "hosting_capacity_ev_std",
                        "hosting_capacity_ev_min",
                        "hosting_capacity_ev_max",
                        "hosting_capacity_ev_any_pass_mean",
                        "hosting_capacity_ev_proven_contiguous_mean",
                        "min_n_for_target_peak_reduction_mean",
                        "max_site_peak_reduction_kw_mean",
                    ]
                ].to_string(index=False)
            )
            print("\nProbabilistic hosting-capacity summary:")
            print(df_probability_summary.to_string(index=False))
            print("Combined Results CSV:", combined_output)
            print("Probability by N CSV:", probability_output)
            print("Probability Summary CSV:", probability_summary_output)
            print("Summary Results CSV:", summary_output)
            print("Aggregated Summary CSV:", agg_summary_output)
    else:
        #unmanaged charging function
        (
            avgwaitSlow, avgwaitFast, avgwaitGen,
            avgWaitedSlow, avgWaitedFast, avgWaitedGen,
            avgUnchargedSlow, avgUnchargedFast, avgUnchargedGen
        ) = Unmanaged_usecase(useCase)
        unmanaged_Load=slow_Load_1h_avg+fast_Load_1h_avg + gen_Load_1h_avg
        if avgWaitedSlow >= 0:
            print('Avg slow unmanaged EVs that waited:', round(avgWaitedSlow, 2))
            print('Avg slow unmanaged EVs uncharged:', round(avgUnchargedSlow, 2))
        if avgWaitedFast >= 0:
            print('Avg fast unmanaged EVs that waited:', round(avgWaitedFast, 2))
            print('Avg fast unmanaged EVs uncharged:', round(avgUnchargedFast, 2))
        if avgWaitedGen >= 0:
            print('Avg external unmanaged EVs that waited:', round(avgWaitedGen, 2))
            print('Avg external unmanaged EVs uncharged:', round(avgUnchargedGen, 2))
        print('start managed charging')

        if isManaged ==1:
            v1g_startingTime, v1g_endingTime,v1g_startingSOC=initiate_V1G(n_V1G_Veh) #initiate V1G vehicles' states
            diagnose_v1g_samples(
                v1g_startingTime,
                v1g_endingTime,
                v1g_startingSOC,
                v1g_energy_required_kwh,
                batteryCapacity,
                dwell_slots=v1g_dwell_slots,
            )
            print("avail chargers min/max:", int(np.min(available_internal_chargers_by_slot)), int(np.max(available_internal_chargers_by_slot)))
            print("avail chargers first 20:", available_internal_chargers_by_slot[:20].tolist())
            print("sum avail chargers:", float(np.sum(available_internal_chargers_by_slot)))
            print("power_V1GUppper:", power_V1GUppper, "eff:", charge_efficiency, "SLOT_HOURS:", SLOT_HOURS)
            objective_mode = "A_peak"
            chargeprofiles, total_load, results = V1G_optimization(objective_mode=objective_mode)  #optimize EV charging profiles
            EV_load, opTotal_load=loadAnalysis(chargeprofiles, total_load, objective_mode=objective_mode, results=results)
        else:
            # Unmanaged-only run: use unmanaged EV load directly without managed optimization.
            EV_load = unmanaged_Load
            if opt_includeSolar == 0:
                opTotal_load = base_Load + unmanaged_Load
            else:
                opTotal_load = base_Load + unmanaged_Load - solarCurve

        writeCSV(EV_load,opTotal_load)


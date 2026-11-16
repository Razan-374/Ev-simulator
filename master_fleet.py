"""Shared deterministic EV realizations for V1G/V2G comparisons."""

import hashlib
import os

import numpy as np
import pandas as pd


MASTER_FLEET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_fleets")


def _day_type(timestamp):
    weekday = pd.Timestamp(timestamp).weekday()
    if weekday == 4:
        return "friday"
    if weekday >= 5:
        return "weekend"
    return "weekday"


def _load_internal_pool(day_type):
    frame = pd.read_csv("session_behaviour_clusters.csv", parse_dates=["arrival_time"])
    frame["arrival_time"] = pd.to_datetime(frame["arrival_time"], errors="coerce")
    frame = frame.dropna(subset=["arrival_time"])
    frame["day_type"] = frame["arrival_time"].map(_day_type)
    selected = frame[frame["day_type"] == day_type]
    return selected if len(selected) >= 30 else frame


def _load_external_pool(day_type):
    frame = pd.read_csv(
        "external.csv",
        parse_dates=["chargingStart ISO-8601", "chargingStop ISO-8601"],
    )
    frame["chargingStart ISO-8601"] = pd.to_datetime(frame["chargingStart ISO-8601"], errors="coerce")
    frame["chargingStop ISO-8601"] = pd.to_datetime(frame["chargingStop ISO-8601"], errors="coerce")
    frame = frame.dropna(subset=["chargingStart ISO-8601", "chargingStop ISO-8601"])
    frame["arrival_minute"] = (
        frame["chargingStart ISO-8601"].dt.hour * 60
        + frame["chargingStart ISO-8601"].dt.minute
    )
    if "chargingDuration[hh:mm:ss]" in frame.columns:
        frame["dwell_minutes"] = pd.to_timedelta(frame["chargingDuration[hh:mm:ss]"]).dt.total_seconds() / 60
    else:
        frame["dwell_minutes"] = (
            frame["chargingStop ISO-8601"] - frame["chargingStart ISO-8601"]
        ).dt.total_seconds() / 60
    frame["energy_kwh"] = pd.to_numeric(frame["energyConsumptionInKWh"], errors="coerce")
    frame = frame.dropna(subset=["dwell_minutes", "energy_kwh"])
    frame["day_type"] = frame["chargingStart ISO-8601"].map(_day_type)
    selected = frame[frame["day_type"] == day_type]
    return selected if len(selected) >= 50 else frame


def _fleet_hash(frame):
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_master_fleet(
    seed,
    day_type="weekday",
    n_internal=100,
    n_external=15,
    battery_capacity_kwh=60.0,
    fleet_dir=None,
    dwell_cap_minutes=None,
    protected_user_ids=(),
):
    """Create or load one deterministic internal/external realization for a seed."""
    fleet_dir = fleet_dir or MASTER_FLEET_DIR
    os.makedirs(fleet_dir, exist_ok=True)
    suffix = f"_dwellcap{int(dwell_cap_minutes)}" if dwell_cap_minutes is not None else ""
    internal_path = os.path.join(fleet_dir, f"master_internal_fleet_seed{seed}{suffix}.csv")
    external_path = os.path.join(fleet_dir, f"master_external_fleet_seed{seed}.csv")

    if not os.path.exists(internal_path) or not os.path.exists(external_path):
        rng = np.random.default_rng(10000 + int(seed))
        internal_pool = _load_internal_pool(day_type)
        internal_rows = internal_pool.iloc[rng.integers(0, len(internal_pool), size=n_internal)].copy()
        dwell_minutes = np.maximum(
            internal_rows["parking_duration"].astype(float).to_numpy() * 60.0,
            10.0,
        )
        if dwell_cap_minutes is not None:
            dwell_minutes = np.minimum(dwell_minutes, float(dwell_cap_minutes))
        protected_ids = {str(user_id) for user_id in protected_user_ids}
        internal = pd.DataFrame({
            "seed": int(seed),
            "EV_id": np.arange(n_internal),
            "start_minute": np.clip(
                internal_rows["arrival_time"].dt.hour.to_numpy() * 60
                + internal_rows["arrival_time"].dt.minute.to_numpy(), 0, 1439
            ),
            "dwell_minutes": dwell_minutes,
            "energy_required_kwh": np.clip(internal_rows["kWhDelivered"].astype(float).to_numpy(), 0.1, battery_capacity_kwh),
            "emergency_vehicle": (
                internal_rows["user_id"].astype(str).isin(protected_ids).to_numpy()
                if "user_id" in internal_rows.columns
                else np.zeros(n_internal, dtype=bool)
            ),
        })
        internal = internal.sort_values("start_minute", kind="stable").reset_index(drop=True)
        internal["EV_id"] = np.arange(n_internal)
        internal["start_slot"] = np.floor(internal["start_minute"] / 15).astype(int)
        internal["dwell_slots"] = np.round(internal["dwell_minutes"] / 15).astype(int)
        internal["end_slot"] = (internal["start_slot"] + internal["dwell_slots"]) % 96
        internal["starting_SOC"] = 1.0 - internal["energy_required_kwh"] / battery_capacity_kwh
        internal["fleet_hash"] = _fleet_hash(internal)
        internal.to_csv(internal_path, index=False)

        external_pool = _load_external_pool(day_type)
        external_rows = external_pool.iloc[rng.integers(0, len(external_pool), size=n_external)].copy()
        external = pd.DataFrame({
            "seed": int(seed),
            "EV_id": np.arange(n_external),
            "arrival_minute": external_rows["arrival_minute"].to_numpy(),
            "dwell_minutes": np.maximum(external_rows["dwell_minutes"].astype(float).to_numpy(), 10.0),
            "energy_required_kwh": np.maximum(external_rows["energy_kwh"].astype(float).to_numpy(), 0.1),
        })
        external["fleet_hash"] = _fleet_hash(external)
        external.to_csv(external_path, index=False)

    return pd.read_csv(internal_path), pd.read_csv(external_path), internal_path, external_path

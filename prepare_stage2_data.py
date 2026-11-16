"""Reconstruct observed EV charging and provisionally separate hospital load.

Run from any directory: python prepare_stage2_data.py
Inputs remain unchanged; outputs are written next to this script in stage2_data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "stage2_data"
SLOT = pd.Timedelta(minutes=15)
SLOT_HOURS = 0.25
POWER_KW = 11.0
TOLERANCE_KWH = 1e-8
SESSION_SOURCES = {
    "session_behaviour_clusters.csv": ("Arrival Time", "Departure Time", "kWhDelivered"),
    "external.csv": (
        "chargingStart ISO-8601",
        "chargingStop ISO-8601",
        "energyConsumptionInKWh",
    ),
}


def load_sessions(root):
    sources = []
    for filename, (start_col, end_col, energy_col) in SESSION_SOURCES.items():
        data = pd.read_csv(root / filename)
        required = ["evseId", start_col, end_col, energy_col]
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"{filename}: missing columns {sorted(missing)}")
        frame = pd.DataFrame({
            "source": filename,
            "source_row": np.arange(2, len(data) + 2),
            "evse_id": data["evseId"],
            "start": pd.to_datetime(data[start_col], errors="coerce", utc=True),
            "stop": pd.to_datetime(data[end_col], errors="coerce", utc=True),
            "recorded_energy_kwh": pd.to_numeric(data[energy_col], errors="coerce"),
        })
        sources.append(frame)
    sessions = pd.concat(sources, ignore_index=True)
    sessions["status"] = "pending"
    invalid = (
        sessions["start"].isna() | sessions["stop"].isna()
        | sessions["evse_id"].isna()
        | ~np.isfinite(sessions["recorded_energy_kwh"])
        | (sessions["recorded_energy_kwh"] < 0)
        | (sessions["stop"] <= sessions["start"])
    )
    sessions.loc[invalid, "status"] = "invalid_input"
    # Exact repeated records are not additional charging sessions. Conflicting
    # records with the same EVSE and interval are excluded, not guessed away.
    keys = ["evse_id", "start", "stop"]
    valid = sessions["status"].eq("pending")
    conflict = sessions.loc[valid].groupby(keys)["recorded_energy_kwh"].transform("nunique").gt(1)
    sessions.loc[conflict.index[conflict], "status"] = "conflicting_duplicate"
    valid = sessions["status"].eq("pending")
    duplicate = sessions.loc[valid].duplicated(keys + ["recorded_energy_kwh"], keep="first")
    sessions.loc[duplicate.index[duplicate], "status"] = "duplicate_excluded"
    return sessions


def reconstruct_session(start, stop, energy_kwh, power_kw=POWER_KW):
    """Return slot energy, delivered energy and shortfall within the recorded window."""
    duration_hours = (stop - start).total_seconds() / 3600.0
    delivered = min(energy_kwh, power_kw * duration_hours)
    charge_end = min(stop, start + pd.to_timedelta(delivered / power_kw, unit="h"))
    slot_energy = {}
    cursor = start
    while cursor < charge_end:
        slot = cursor.floor("15min")
        boundary = min(slot + SLOT, charge_end)
        slot_energy[slot] = power_kw * (boundary - cursor).total_seconds() / 3600.0
        cursor = boundary
    actual = sum(slot_energy.values())
    return slot_energy, actual, max(0.0, energy_kwh - actual)


def load_hospital(path):
    data = pd.read_csv(path)
    required = ["Timestamp", "hospitalpower_kw", "powerev_kw", "powerhospitalalone", "EnergyevKWh"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    data["timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    if data["timestamp"].isna().any():
        raise ValueError("Hospital data contains invalid timestamps; cannot establish a time grid")
    if (data["timestamp"] != data["timestamp"].dt.floor("15min")).any():
        raise ValueError("Hospital data contains timestamps off the 15-minute grid")
    for col in required[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data.loc[~np.isfinite(data[col]), col] = np.nan
    identity_error = (data["hospitalpower_kw"] - data["powerhospitalalone"] - data["powerev_kw"]).abs()
    ev_conversion_error = (data["powerev_kw"] * SLOT_HOURS - data["EnergyevKWh"]).abs()
    duplicates = data["timestamp"].duplicated(keep=False)
    counts = data.groupby("timestamp").size()
    grid = pd.date_range(data["timestamp"].min(), data["timestamp"].max(), freq="15min")
    # Repeated naive times may be distinct DST hours. Do not select or average them.
    unique = data.loc[~duplicates].set_index("timestamp").reindex(grid)
    unique.index.name = "Timestamp"
    unique["interval_status"] = np.where(
        unique.index.isin(counts[counts > 1].index), "ambiguous_duplicate",
        np.where(unique.index.isin(counts.index), "present", "missing"),
    )
    return unique, data, identity_error, ev_conversion_error, int(duplicates.sum())


def prepare(root=ROOT, output=OUTPUT):
    hospital, raw_hospital, identity_error, ev_conversion_error, duplicate_rows = load_hospital(
        root / "hospital_data_with_weekday_weekend.csv"
    )
    sessions = load_sessions(root)
    grid = hospital.index
    slot_energy = np.zeros(len(grid), dtype=float)
    session_results = []
    for row in sessions.itertuples(index=False):
        result = {
            "source": row.source, "source_row": row.source_row, "evse_id": row.evse_id,
            "start_utc": row.start, "stop_utc": row.stop,
            "recorded_energy_kwh": row.recorded_energy_kwh,
            "reconstructed_energy_kwh": np.nan, "shortfall_kwh": np.nan,
            "energy_on_hospital_grid_kwh": np.nan, "status": row.status,
        }
        if row.status == "pending":
            energies, delivered, shortfall = reconstruct_session(row.start, row.stop, row.recorded_energy_kwh)
            on_grid = 0.0
            for timestamp, energy in energies.items():
                # The hospital clock is naive; treating it as UTC is provisional.
                nominal_time = timestamp.tz_convert(None)
                index = grid.get_indexer([nominal_time])[0]
                if index >= 0:
                    slot_energy[index] += energy
                    on_grid += energy
            result.update(
                reconstructed_energy_kwh=delivered, shortfall_kwh=shortfall,
                energy_on_hospital_grid_kwh=on_grid,
                status="infeasible_11kw" if shortfall > TOLERANCE_KWH else "reconstructed",
            )
        session_results.append(result)
    results = pd.DataFrame(session_results)
    hospital["measured_hospital_load_kw"] = hospital["hospitalpower_kw"]
    hospital["measured_ev_power_kw_unverified"] = hospital["powerev_kw"]
    hospital["reconstructed_ev_power_kw"] = slot_energy / SLOT_HOURS
    # This is the dataset's existing numerical separation, not a second subtraction.
    hospital["estimated_building_only_load_kw"] = hospital["powerhospitalalone"]
    hospital["measurement_verification_status"] = np.where(
        hospital["interval_status"].eq("present"),
        "unverified_definitions_and_time_alignment", hospital["interval_status"],
    )
    output_columns = [
        "measured_hospital_load_kw", "measured_ev_power_kw_unverified",
        "reconstructed_ev_power_kw", "estimated_building_only_load_kw",
        "measurement_verification_status", "interval_status",
    ]
    valid_measurements = (
        hospital["interval_status"].eq("present")
        & hospital["hospitalpower_kw"].notna()
        & hospital["powerev_kw"].notna()
        & hospital["powerhospitalalone"].notna()
    )
    hospital.loc[
        hospital["interval_status"].eq("present") & ~valid_measurements,
        "measurement_verification_status",
    ] = "missing_measurement_values"
    comparison = valid_measurements
    measured_kwh = hospital.loc[comparison, "powerev_kw"].sum() * SLOT_HOURS
    reconstructed_kwh = hospital.loc[comparison, "reconstructed_ev_power_kw"].sum() * SLOT_HOURS
    difference = hospital.loc[comparison, "reconstructed_ev_power_kw"] - hospital.loc[comparison, "powerev_kw"]
    included = results["status"].isin(["reconstructed", "infeasible_11kw"])
    feasible = results["status"].eq("reconstructed")
    energy_error = (results.loc[feasible, "recorded_energy_kwh"] - results.loc[feasible, "reconstructed_energy_kwh"]).abs()
    metrics = [
        ("hospital_source_rows", len(raw_hospital), "Source rows, including repeated timestamps"),
        ("grid_intervals", len(grid), "Nominal continuous 15-minute grid"),
        ("hospital_duplicate_rows", duplicate_rows, "Rows excluded because their naive timestamp is ambiguous"),
        ("hospital_ambiguous_intervals", int(hospital["interval_status"].eq("ambiguous_duplicate").sum()), "Not resolved without a verified time zone"),
        ("hospital_missing_intervals", int(hospital["interval_status"].eq("missing").sum()), "Kept on grid with missing hospital values"),
        ("hospital_nonfinite_or_missing_values", int((~valid_measurements & hospital["interval_status"].eq("present")).sum()), "Rows with missing values in separation columns"),
        ("hospital_identity_max_abs_error_kw", identity_error.max(), "hospitalpower_kw - powerhospitalalone - powerev_kw; numerical identity only"),
        ("ev_energy_conversion_max_abs_error_kwh", ev_conversion_error.max(), "powerev_kw / 4 - EnergyevKWh; numerical identity only"),
        ("session_source_rows", len(results), "All observed sessions from both sources"),
        ("sessions_reconstructed_including_infeasible", int(included.sum()), "Includes capped infeasible sessions"),
        ("sessions_feasible", int(feasible.sum()), "11 kW can deliver recorded energy within recorded interval"),
        ("sessions_infeasible_11kw", int(results["status"].eq("infeasible_11kw").sum()), "Capped at 11 kW and recorded stop; shortfall is not silently delivered"),
        ("sessions_invalid", int(results["status"].eq("invalid_input").sum()), "Excluded from reconstruction"),
        ("sessions_duplicate_excluded", int(results["status"].eq("duplicate_excluded").sum()), "Exact duplicates excluded"),
        ("sessions_conflicting_duplicate", int(results["status"].eq("conflicting_duplicate").sum()), "All conflicting records excluded"),
        ("recorded_energy_included_kwh", results.loc[included, "recorded_energy_kwh"].sum(), "Recorded energy for included sessions"),
        ("reconstructed_energy_all_sessions_kwh", results.loc[included, "reconstructed_energy_kwh"].sum(), "Full session windows, including outside hospital coverage"),
        ("infeasible_energy_shortfall_kwh", results.loc[included, "shortfall_kwh"].sum(), "Recorded energy not delivered by capped reconstruction"),
        ("session_energy_balance_error_kwh", abs(results.loc[included, "recorded_energy_kwh"].sum() - results.loc[included, "reconstructed_energy_kwh"].sum() - results.loc[included, "shortfall_kwh"].sum()), "Recorded = reconstructed + shortfall"),
        ("feasible_session_max_energy_error_kwh", energy_error.max() if not energy_error.empty else np.nan, "Rounding error for feasible sessions"),
        ("cross_midnight_sessions", int((included & (sessions["start"].dt.date != sessions["stop"].dt.date)).sum()), "UTC dates; session windows, not inferred charging duration"),
        ("reconstructed_energy_hospital_grid_kwh", slot_energy.sum(), "Nominal hospital grid, includes missing and ambiguous hospital intervals"),
        ("grid_session_energy_balance_error_kwh", abs(slot_energy.sum() - results.loc[included, "energy_on_hospital_grid_kwh"].sum()), "Grid energy versus session energy allocated to grid"),
        ("comparable_interval_count_provisional", int(comparison.sum()), "Numerical overlap only; measurement scope and timezone unverified"),
        ("measured_ev_energy_comparison_kwh", measured_kwh, "Provisional: valid hospital intervals only"),
        ("reconstructed_ev_energy_comparison_kwh", reconstructed_kwh, "Provisional: same valid hospital intervals"),
        ("ev_power_comparison_mae_kw", difference.abs().mean(), "Provisional: not proof of comparable coverage"),
        ("building_negative_intervals", int((hospital.loc[valid_measurements, "powerhospitalalone"] < 0).sum()), "Negative values are reported, not clipped"),
        ("separation_identity_max_abs_error_kw", (hospital.loc[valid_measurements, "hospitalpower_kw"] - hospital.loc[valid_measurements, "powerhospitalalone"] - hospital.loc[valid_measurements, "powerev_kw"]).abs().max(), "Dataset numerical separation; not independent verification"),
    ]
    metadata = {
        "inputs": ["hospital_data_with_weekday_weekend.csv", *SESSION_SOURCES],
        "hospital_nominal_coverage": [str(grid.min()), str(grid.max())],
        "session_utc_coverage": [str(sessions["start"].min()), str(sessions["stop"].max())],
        "interval_minutes": 15,
        "assumptions_and_limitations": [
            "All sessions from both session files are included together, except flagged invalid or duplicate records.",
            "11 kW constant charging starts at the recorded start; session stop is treated as the latest permitted charging time, but may be departure rather than charging stop.",
            "Infeasible sessions are capped at 11 kW through stop, with undelivered energy reported; exact energy and the power/time limits cannot all be satisfied.",
            "The naive hospital clock is provisionally interpreted as UTC and as interval starts to align with offset-aware session timestamps. Its actual timezone and interval start/end convention are unknown.",
            "Duplicate hospital clock times are left blank and missing intervals are inserted; no DST resolution or interpolation is assumed.",
            "powerev_kw equals four times EnergyevKWh numerically, but its source, measurement status and charger coverage are unverified.",
            "powerhospitalalone numerically equals hospitalpower_kw minus powerev_kw; this is a conditional building estimate, not proof that the hospital meter includes all EV load.",
            "The comparison is exploratory, not a verified like-for-like comparison; session completeness, timezone, meter coverage and temporal semantics require confirmation.",
            "Session energy is assumed to be charger-side; no charging efficiency is applied.",
        ],
        "measurement_verification_status": "unverified",
        "building_estimate_method": "existing powerhospitalalone; equivalent numerically to hospitalpower_kw minus powerev_kw",
    }
    output.mkdir(parents=True, exist_ok=True)
    hospital[output_columns].to_csv(output / "stage2_load_separation.csv", index_label="Timestamp")
    results.to_csv(output / "stage2_session_reconstruction.csv", index=False)
    pd.DataFrame(metrics, columns=["metric", "value", "notes"]).to_csv(output / "stage2_validation_report.csv", index=False)
    with (output / "stage2_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
    return dict((key, value) for key, value, _ in metrics)


if __name__ == "__main__":
    for name, value in prepare().items():
        print(f"{name}: {value}")

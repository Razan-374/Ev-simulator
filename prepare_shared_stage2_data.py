"""Prepare provisional shared Stage 2 inputs without modifying the optimizers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from investigate_stage2_data import overlapping_pairs
from prepare_stage2_data import (
    ROOT, SLOT, SLOT_HOURS, TOLERANCE_KWH, load_hospital, load_sessions,
    reconstruct_session,
)


OUTPUT = ROOT / "stage2_data" / "shared_inputs"


def build_shared_inputs(root=ROOT, output=OUTPUT):
    hospital, _, _, _, _ = load_hospital(root / "hospital_data_with_weekday_weekend.csv")
    grid = hospital.index
    window_start = grid[0]
    window_end = grid[-1] + SLOT
    sessions = load_sessions(root)
    raw_sources = {}
    for source in sessions["source"].unique():
        raw = pd.read_csv(root / source, dtype=str, keep_default_na=False)
        if "ev_type" not in raw:
            raise ValueError(f"{source}: ev_type is missing")
        categories = raw["ev_type"].str.strip().str.lower()
        if (~categories.isin(["internal", "external"])).any():
            raise ValueError(f"{source}: unknown or missing ev_type")
        sessions.loc[sessions["source"].eq(source), "category"] = categories.to_numpy()
        raw_sources[source] = raw

    external_energy = np.zeros(len(grid))
    window_delta = np.zeros(len(grid) + 1, dtype=int)
    infeasible_delta = np.zeros(len(grid) + 1, dtype=int)
    internal_rows, external_rows, excluded_rows = [], [], []
    for row in sessions.itertuples(index=False):
        raw = raw_sources[row.source].iloc[row.source_row - 2]
        if row.source == "session_behaviour_clusters.csv":
            start_col, stop_col, energy_col = "Arrival Time", "Departure Time", "kWhDelivered"
        else:
            start_col, stop_col, energy_col = "chargingStart ISO-8601", "chargingStop ISO-8601", "energyConsumptionInKWh"
        record = {
            "session_id": f"{row.source}:{row.source_row}",
            "source": row.source, "source_row": row.source_row, "evse_id": raw["evseId"],
            "user_category": row.category,
            "recorded_start": raw[start_col], "recorded_stop": raw[stop_col],
            "recorded_energy_kwh": raw[energy_col],
            "input_status": row.status,
        }
        if row.category == "internal":
            record["recorded_user_id"] = raw["user_id"] if "user_id" in raw.index else raw["card number"]
        if row.status != "pending":
            record["exclusion_reason"] = row.status
            excluded_rows.append(record)
            continue
        shortfall = max(0.0, row.recorded_energy_kwh - 11 * (row.stop - row.start).total_seconds() / 3600)
        record["infeasible_at_11kw_diagnostic"] = shortfall > TOLERANCE_KWH
        record["diagnostic_11kw_shortfall_kwh"] = shortfall
        # A UTC interpretation of the hospital clock is provisional, not verified.
        start = row.start.tz_convert(None)
        stop = row.stop.tz_convert(None)
        if start >= window_end or stop <= window_start:
            record["exclusion_reason"] = "outside_hospital_grid"
            excluded_rows.append(record)
            continue
        record.update({
            "recorded_start_utc": row.start.isoformat(), "recorded_stop_utc": row.stop.isoformat(),
            "crosses_grid_start": start < window_start, "crosses_grid_end": stop > window_end,
            "end_time_semantics": "unverified_stop_may_not_be_departure",
        })
        if row.category == "internal":
            # Diagnostic only; the optimizer, not this preprocessing, decides internal charging.
            internal_rows.append(record)
        else:
            slot_energies, delivered, shortfall = reconstruct_session(row.start, row.stop, row.recorded_energy_kwh)
            on_grid = 0.0
            for timestamp, energy in slot_energies.items():
                nominal_time = timestamp.tz_convert(None)
                index = grid.get_indexer([nominal_time])[0]
                if index >= 0:
                    external_energy[index] += energy
                    on_grid += energy
            first = max(0, int((start.floor("15min") - window_start) / SLOT))
            last = min(len(grid), int((stop.ceil("15min") - window_start) / SLOT))
            window_delta[first] += 1
            window_delta[last] -= 1
            if shortfall > TOLERANCE_KWH:
                infeasible_delta[first] += 1
                infeasible_delta[last] -= 1
            record.update({
                "reconstructed_energy_full_session_kwh": delivered,
                "reconstructed_energy_on_grid_kwh": on_grid,
                "shortfall_full_session_kwh": shortfall,
                "reconstruction_status": "infeasible_11kw" if shortfall > TOLERANCE_KWH else "reconstructed_11kw",
            })
            external_rows.append(record)

    internal = pd.DataFrame(internal_rows)
    external = pd.DataFrame(external_rows)
    excluded = pd.DataFrame(excluded_rows)
    if excluded.empty:
        excluded = pd.DataFrame(columns=[
            "session_id", "source", "source_row", "evse_id", "user_category", "recorded_start",
            "recorded_stop", "recorded_energy_kwh", "input_status", "exclusion_reason",
            "infeasible_at_11kw_diagnostic", "diagnostic_11kw_shortfall_kwh",
        ])
    for column in ["infeasible_at_11kw_diagnostic", "diagnostic_11kw_shortfall_kwh"]:
        if column not in excluded:
            excluded[column] = np.nan
    if internal.empty or external.empty:
        raise ValueError("No internal or external sessions overlap the hospital grid")
    unresolved = excluded.loc[excluded["exclusion_reason"].isin(["invalid_input", "conflicting_duplicate"])]
    external_unresolved = unresolved.loc[unresolved["user_category"].eq("external")]
    overlaps = overlapping_pairs(sessions.loc[
        sessions["status"].eq("pending") & sessions["category"].eq("external")
    ])
    window_counts = np.cumsum(window_delta[:-1])
    infeasible_counts = np.cumsum(infeasible_delta[:-1])
    building = hospital["powerhospitalalone"]
    external_power = external_energy / SLOT_HOURS
    fixed_load = building + external_power
    fixed_status = np.full(len(grid), "provisional_unverified_definitions_and_timezone", dtype=object)
    fixed_status[hospital["interval_status"].ne("present").to_numpy()] = hospital.loc[
        hospital["interval_status"].ne("present"), "interval_status"
    ].to_numpy()
    fixed_status[(hospital["interval_status"].eq("present") & building.isna()).to_numpy()] = "missing_building_measurement"
    if not external_unresolved.empty:
        fixed_status[:] = "unresolved_external_records_and_provisional_alignment"
    fixed = pd.DataFrame({
        "timestamp_nominal": grid,
        "building_load_kw_provisional": building.to_numpy(),
        "external_charging_kw_assumed_11kw": external_power,
        "fixed_load_kw_provisional": fixed_load.to_numpy(),
        "external_infeasible_session_window_count": infeasible_counts,
        "hospital_interval_status": hospital["interval_status"].to_numpy(),
        "fixed_load_status": fixed_status,
    })
    # These are observed session windows, not verified connector occupancy.
    outlets = pd.DataFrame({
        "timestamp_nominal": grid,
        "observed_external_session_window_count": window_counts,
        "verified_shared_outlets_occupied": np.nan,
        "verified_controllable_outlets_available": np.nan,
        "availability_status": "unknown_mapping_departure_and_timezone",
    })
    valid = building.notna()
    identity_error = (fixed_load[valid] - building[valid] - external_power[valid]).abs().max()
    external_recorded = pd.to_numeric(external["recorded_energy_kwh"])
    energy_balance = (
        external_recorded - external["reconstructed_energy_full_session_kwh"]
        - external["shortfall_full_session_kwh"]
    ).abs().max()
    grid_error = abs(external["reconstructed_energy_on_grid_kwh"].sum() - external_energy.sum())
    metrics = [
        ("hospital_grid_intervals", len(grid), "15-minute nominal continuous grid"),
        ("missing_hospital_intervals", int(hospital["interval_status"].eq("missing").sum()), "Building and fixed load missing"),
        ("ambiguous_hospital_intervals", int(hospital["interval_status"].eq("ambiguous_duplicate").sum()), "Building and fixed load missing"),
        ("internal_sessions_overlapping_grid", len(internal), "Original session times and energy retained"),
        ("internal_infeasible_at_11kw_diagnostic", int(internal["infeasible_at_11kw_diagnostic"].sum()), "Does not prescribe internal optimizer power"),
        ("internal_sessions_crossing_grid_boundaries", int((internal["crosses_grid_start"] | internal["crosses_grid_end"]).sum()), "Requires boundary state handling before optimization"),
        ("external_sessions_overlapping_grid", len(external), "Includes infeasible sessions and boundary crossings"),
        ("external_sessions_crossing_grid_boundaries", int((external["crosses_grid_start"] | external["crosses_grid_end"]).sum()), "Energy and session window are retained across boundaries"),
        ("external_infeasible_11kw_sessions", int(external["reconstruction_status"].eq("infeasible_11kw").sum()), "Shortfall retained per session"),
        ("external_recorded_energy_full_sessions_kwh", external_recorded.sum(), "Includes portions outside the hospital grid"),
        ("external_reconstructed_energy_full_sessions_kwh", external["reconstructed_energy_full_session_kwh"].sum(), "Limited to recorded session windows"),
        ("external_shortfall_full_sessions_kwh", external["shortfall_full_session_kwh"].sum(), "Not delivered in the 11 kW reconstruction"),
        ("external_reconstructed_energy_on_grid_kwh", external_energy.sum(), "Includes intervals with missing hospital measurements"),
        ("external_session_energy_balance_max_error_kwh", energy_balance, "Recorded = full-session reconstructed + shortfall"),
        ("external_grid_energy_balance_error_kwh", grid_error, "Sum of session grid energy versus aggregate grid"),
        ("fixed_load_identity_max_error_kw", identity_error, "Fixed = building + external; powerev_kw is not added"),
        ("negative_building_intervals", int((building < 0).sum()), "Negative values are not clipped"),
        ("external_same_evse_overlap_pairs", len(overlaps), "Across source history; overlap may indicate data issues"),
        ("excluded_outside_grid", int(excluded["exclusion_reason"].eq("outside_hospital_grid").sum()) if not excluded.empty else 0, "Outside nominal grid, preserved in source and exclusions file"),
        ("excluded_outside_grid_infeasible_at_11kw", int(excluded.loc[excluded["exclusion_reason"].eq("outside_hospital_grid"), "infeasible_at_11kw_diagnostic"].eq(True).sum()), "Diagnostic shortfalls retained in exclusions file"),
        ("excluded_invalid_or_conflicting", len(unresolved), "Not used in the fixed load or internal input"),
        ("unknown_shared_occupancy_intervals", len(outlets), "No verified EVSE mapping, departure semantics or time alignment"),
    ]
    metadata = {
        "status": "provisional_not_ready_for_optimizer_integration",
        "shared_inputs_for": ["V1G", "V2G"],
        "source_files": ["hospital_data_with_weekday_weekend.csv", "session_behaviour_clusters.csv", "external.csv"],
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ["hospital_data_with_weekday_weekend.csv", "session_behaviour_clusters.csv", "external.csv"]
        },
        "coverage_nominal": {"start": str(window_start), "end_exclusive": str(window_end), "interval_minutes": 15},
        "fixed_load_formula": "powerhospitalalone + reconstructed external charging; powerev_kw is not added",
        "external_charging": "11 kW from recorded start, capped at recorded stop; infeasible-session shortfalls are reported, not invented",
        "internal_charging": "No internal charging profile is precomputed for the fixed load; 11 kW feasibility fields are diagnostic only.",
        "session_scope": "Sessions whose recorded windows overlap the nominal hospital grid, with original fields preserved. Other records are listed separately.",
        "shared_outlets": "No verified mapping or departure semantics available. Shared occupancy and controllable capacity remain unknown for every interval; window counts include each external session overlapping any part of a slot and are not occupancy.",
        "limitations": [
            "Stage 1 completion for both optimizers is not confirmed; do not integrate.",
            "The hospital clock is naive and provisionally treated as UTC interval starts; UTC offsets, DST, and interval convention are not verified.",
            "powerhospitalalone and powerev_kw meter definitions and charging coverage are unverified; adding reconstructed external load can still double-count if building-only definition is wrong.",
            "External session stop may be charging end rather than physical departure; EVSE-to-shared-outlet mapping, session completeness and outlet ratings are unverified.",
            "Session category and charger-side energy meanings are not independently verified; same-identifier overlaps reported in the earlier investigation remain unresolved.",
            "Internal arrival SoC, battery capacity, energy conventions and boundary states are not supplied by these records and must be resolved before optimization.",
            "Shortfall and infeasibility use the recorded window, not a confirmed physical charging window; 11 kW may exceed an unknown outlet rating.",
            "The recorded and reconstructed EV profiles differ substantially and cannot yet be compared as verified like-for-like measurements.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(output / "stage2_fixed_load.csv", index=False)
    internal.to_csv(output / "stage2_internal_sessions.csv", index=False)
    external.to_csv(output / "stage2_external_sessions.csv", index=False)
    excluded.to_csv(output / "stage2_excluded_sessions.csv", index=False)
    outlets.to_csv(output / "stage2_outlet_availability.csv", index=False)
    overlaps.to_csv(output / "stage2_external_evse_overlaps.csv", index=False)
    pd.DataFrame(metrics, columns=["metric", "value", "notes"]).to_csv(output / "stage2_shared_validation_report.csv", index=False)
    with (output / "stage2_shared_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
    for name, value, _ in metrics:
        print(f"{name}: {value}")
    return fixed, internal, external, excluded, outlets, dict((name, value) for name, value, _ in metrics)


if __name__ == "__main__":
    build_shared_inputs()

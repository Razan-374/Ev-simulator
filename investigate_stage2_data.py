"""Audit Stage 2 session overlap, capacity assumptions and monthly EV energy."""

import numpy as np
import pandas as pd

from prepare_stage2_data import (
    OUTPUT, ROOT, SLOT_HOURS, TOLERANCE_KWH, load_hospital, load_sessions,
    reconstruct_session,
)


def overlapping_pairs(sessions):
    pairs = []
    for evse, group in sessions.groupby("evse_id"):
        active = []
        for row in group.sort_values("start").itertuples(index=False):
            active = [previous for previous in active if previous.stop > row.start]
            for previous in active:
                pairs.append({
                    "evse_id": evse,
                    "source_1": previous.source,
                    "source_row_1": previous.source_row,
                    "source_2": row.source,
                    "source_row_2": row.source_row,
                    "overlap_seconds": (min(previous.stop, row.stop) - row.start).total_seconds(),
                })
            active.append(row)
    return pd.DataFrame(pairs, columns=[
        "evse_id", "source_1", "source_row_1", "source_2", "source_row_2", "overlap_seconds",
    ])


def identifier_overlaps(sessions):
    pairs = []
    for _, group in sessions.groupby("identifier"):
        active = []
        for row in group.sort_values("start").itertuples(index=False):
            active = [previous for previous in active if previous.stop > row.start]
            for previous in active:
                pairs.append({
                    "source_1": previous.source, "source_row_1": previous.source_row,
                    "evse_1": previous.evse_id, "source_2": row.source,
                    "source_row_2": row.source_row, "evse_2": row.evse_id,
                    "overlap_seconds": (min(previous.stop, row.stop) - row.start).total_seconds(),
                })
            active.append(row)
    return pd.DataFrame(pairs, columns=[
        "source_1", "source_row_1", "evse_1", "source_2", "source_row_2", "evse_2", "overlap_seconds",
    ])


def investigate(root=ROOT, output=OUTPUT):
    sessions = load_sessions(root)
    cluster = pd.read_csv(root / "session_behaviour_clusters.csv")
    external = pd.read_csv(root / "external.csv")
    for filename, source in [("session_behaviour_clusters.csv", cluster), ("external.csv", external)]:
        if "ev_type" not in source:
            raise ValueError(f"{filename}: ev_type is missing")
        categories = source["ev_type"].astype("string").str.strip().str.lower()
        unknown = ~categories.isin(["internal", "external"])
        if unknown.any():
            raise ValueError(f"{filename}: {unknown.sum()} unknown or missing session categories")
        mask = sessions["source"].eq(filename)
        sessions.loc[mask, "category"] = categories.to_numpy()
        identifier_col = "user_id" if filename == "session_behaviour_clusters.csv" else "card number"
        sessions.loc[mask, "identifier"] = source[identifier_col].astype("string").to_numpy()

    included = sessions.loc[sessions["status"].eq("pending")].copy()
    included["duration_hours"] = (included["stop"] - included["start"]).dt.total_seconds() / 3600
    included["average_required_kw"] = included["recorded_energy_kwh"] / included["duration_hours"]
    included["shortfall_kwh"] = (included["recorded_energy_kwh"] - 11 * included["duration_hours"]).clip(lower=0)
    included["infeasible_11kw"] = included["shortfall_kwh"] > TOLERANCE_KWH

    hospital, _, _, _, _ = load_hospital(root / "hospital_data_with_weekday_weekend.csv")
    valid = (
        hospital["interval_status"].eq("present") & hospital["hospitalpower_kw"].notna()
        & hospital["powerev_kw"].notna() & hospital["powerhospitalalone"].notna()
    )
    categories = ["internal", "external"]
    profiles = pd.DataFrame(0.0, index=hospital.index, columns=categories + [
        "cluster_source", "external_file", "evse_6053_6054",
    ])
    for row in included.itertuples(index=False):
        energies, _, _ = reconstruct_session(row.start, row.stop, row.recorded_energy_kwh)
        for timestamp, energy in energies.items():
            nominal = timestamp.tz_convert(None)
            if nominal in profiles.index:
                profiles.at[nominal, row.category] += energy
                source_column = "cluster_source" if row.source == "session_behaviour_clusters.csv" else "external_file"
                profiles.at[nominal, source_column] += energy
                if row.evse_id in ("DE*SAE*E6053*03", "DE*SAE*E6054*03"):
                    profiles.at[nominal, "evse_6053_6054"] += energy

    prepared = pd.read_csv(output / "stage2_load_separation.csv", parse_dates=["Timestamp"]).set_index("Timestamp")
    if not prepared.index.equals(hospital.index):
        raise ValueError("Prepared load grid differs from source data; rerun preparation before investigation")
    total = profiles[categories].sum(axis=1)
    if not np.allclose(total.to_numpy() / SLOT_HOURS, prepared["reconstructed_ev_power_kw"].to_numpy(), atol=1e-8, rtol=0):
        raise ValueError("Category reconstructions differ from the prepared load; do not compare stale outputs")

    monthly_rows = []
    for month, indices in hospital.groupby(hospital.index.to_period("M")).groups.items():
        month_valid = valid.loc[indices]
        valid_indices = month_valid.index[month_valid]
        measured = hospital.loc[valid_indices, "powerev_kw"].sum() * SLOT_HOURS
        internal = profiles.loc[valid_indices, "internal"].sum()
        external_kwh = profiles.loc[valid_indices, "external"].sum()
        cluster_kwh = profiles.loc[valid_indices, "cluster_source"].sum()
        evse_6053_6054_kwh = profiles.loc[valid_indices, "evse_6053_6054"].sum()
        monthly_rows.append({
            "month_nominal": str(month),
            "first_interval": str(indices.min()), "last_interval": str(indices.max()),
            "valid_intervals": len(valid_indices), "excluded_intervals": len(indices) - len(valid_indices),
            "recorded_ev_energy_unverified_kwh": measured,
            "reconstructed_internal_kwh": internal,
            "reconstructed_external_kwh": external_kwh,
            "reconstructed_cluster_source_kwh": cluster_kwh,
            "reconstructed_external_file_kwh": profiles.loc[valid_indices, "external_file"].sum(),
            "reconstructed_evse_6053_6054_kwh": evse_6053_6054_kwh,
            "diagnostic_cluster_excluding_6053_6054_kwh": cluster_kwh - evse_6053_6054_kwh,
            "reconstructed_all_kwh": internal + external_kwh,
            "difference_kwh": internal + external_kwh - measured,
        })
    monthly = pd.DataFrame(monthly_rows)

    evse = included.groupby("evse_id").agg(
        session_count=("evse_id", "size"),
        internal_sessions=("category", lambda x: int(x.eq("internal").sum())),
        external_sessions=("category", lambda x: int(x.eq("external").sum())),
        recorded_energy_kwh=("recorded_energy_kwh", "sum"),
        infeasible_sessions=("infeasible_11kw", "sum"),
        shortfall_kwh=("shortfall_kwh", "sum"),
        first_session_utc=("start", "min"),
        last_session_end_utc=("stop", "max"),
    ).reset_index()
    infeasible = included.loc[included["infeasible_11kw"], [
        "source", "source_row", "evse_id", "category", "start", "stop",
        "recorded_energy_kwh", "duration_hours", "average_required_kw", "shortfall_kwh",
    ]]
    overlaps = overlapping_pairs(included)
    same_identifier = identifier_overlaps(included)

    # A clock shift is diagnostic only; hold the comparison mask fixed across shifts.
    shifts = [-8, -4, -1, 0, 1, 4, 8]
    reconstructed_kw = total / SLOT_HOURS
    common = valid.copy()
    for shift in shifts:
        common &= reconstructed_kw.shift(shift).notna()
    measured = hospital.loc[common, "powerev_kw"]
    timing_rows = []
    monthly_timing_rows = []
    for shift in shifts:
        candidate = reconstructed_kw.shift(shift).loc[common]
        timing_rows.append({
            "reconstructed_shift_minutes": shift * 15,
            "intervals": int(common.sum()),
            "mean_absolute_error_kw": (candidate - measured).abs().mean(),
            "correlation": candidate.corr(measured),
        })
        for month, values in candidate.groupby(candidate.index.to_period("M")):
            reference = measured.loc[values.index]
            monthly_timing_rows.append({
                "month_nominal": str(month), "reconstructed_shift_minutes": shift * 15,
                "intervals": len(values),
                "mean_absolute_error_kw": (values - reference).abs().mean(),
                "correlation": values.corr(reference),
            })
    timing = pd.DataFrame(timing_rows)

    # Check same timestamp/energy independently of EVSE and same identifier near the start.
    left = cluster.copy()
    right = external.copy()
    left["start"] = pd.to_datetime(left["Arrival Time"], utc=True)
    left["stop"] = pd.to_datetime(left["Departure Time"], utc=True)
    right["start"] = pd.to_datetime(right["chargingStart ISO-8601"], utc=True)
    right["stop"] = pd.to_datetime(right["chargingStop ISO-8601"], utc=True)
    left["energy"] = pd.to_numeric(left["kWhDelivered"])
    right["energy"] = pd.to_numeric(right["energyConsumptionInKWh"])
    exact_without_evse = len(left.merge(right, on=["start", "stop", "energy"]))
    left["identifier"] = left["user_id"].astype(str)
    right["identifier"] = right["card number"].astype(str)
    possible = left.merge(right, on="identifier", suffixes=("_left", "_right"))
    nearby = ((possible["start_left"] - possible["start_right"]).abs() <= pd.Timedelta(minutes=15)).sum()

    output.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output / "stage2_monthly_ev_comparison.csv", index=False)
    evse.to_csv(output / "stage2_evse_investigation.csv", index=False)
    infeasible.to_csv(output / "stage2_infeasible_sessions.csv", index=False)
    overlaps.to_csv(output / "stage2_evse_overlaps.csv", index=False)
    same_identifier.to_csv(output / "stage2_identifier_overlaps.csv", index=False)
    timing.to_csv(output / "stage2_timing_sensitivity.csv", index=False)
    pd.DataFrame(monthly_timing_rows).to_csv(output / "stage2_monthly_timing_sensitivity.csv", index=False)
    print(f"Sessions: {len(included)}; same-EVSE overlap pairs: {len(overlaps)}")
    print(f"Same numeric identifier overlap pairs: {len(same_identifier)} (identifier namespaces unverified)")
    print(f"Cross-file exact start/stop/energy matches without EVSE: {exact_without_evse}")
    print(f"Cross-file same numeric identifier and start within 15 minutes: {nearby}")
    print(f"EVSE identifiers: {len(evse)}; observed with both user categories: {((evse.internal_sessions > 0) & (evse.external_sessions > 0)).sum()}")
    print(f"Monthly measured total: {monthly.recorded_ev_energy_unverified_kwh.sum():.3f} kWh")
    print(f"Monthly reconstructed internal: {monthly.reconstructed_internal_kwh.sum():.3f} kWh")
    print(f"Monthly reconstructed external: {monthly.reconstructed_external_kwh.sum():.3f} kWh")
    print(f"Monthly reconstructed total: {monthly.reconstructed_all_kwh.sum():.3f} kWh")
    return monthly, evse, infeasible, overlaps, timing


if __name__ == "__main__":
    investigate()

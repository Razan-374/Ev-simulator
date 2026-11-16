"""Compare external charging assumptions without changing prepared Stage 2 inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_stage2_data import (
    OUTPUT, POWER_KW, ROOT, SLOT_HOURS, TOLERANCE_KWH,
    load_hospital, load_sessions, reconstruct_session,
)


ALTERNATIVE_OUTPUT = OUTPUT / "alternative_reconstruction"


def load_ratings(path, known_external_evse):
    if path is None:
        return {}
    data = pd.read_csv(path, dtype={"evse_id": "string", "evidence": "string"})
    required = {"evse_id", "rated_power_kw", "evidence"}
    if not required.issubset(data.columns):
        raise ValueError(f"Ratings file must contain {sorted(required)}")
    data["evse_id"] = data["evse_id"].str.strip()
    data["evidence"] = data["evidence"].str.strip()
    data["rated_power_kw"] = pd.to_numeric(data["rated_power_kw"], errors="coerce")
    invalid = (
        data["evse_id"].isna() | data["evse_id"].eq("")
        | data["evidence"].isna() | data["evidence"].eq("")
        | ~np.isfinite(data["rated_power_kw"]) | (data["rated_power_kw"] <= 0)
    )
    if invalid.any() or data["evse_id"].duplicated().any():
        raise ValueError("Ratings must have unique EVSE IDs, positive finite kW and nonempty evidence")
    unknown = set(data["evse_id"]) - set(known_external_evse)
    if unknown:
        raise ValueError(f"Ratings contain EVSE IDs not in external sessions: {sorted(unknown)}")
    return {
        row.evse_id: (float(row.rated_power_kw), row.evidence)
        for row in data.itertuples(index=False)
    }


def alternative_power(category, required_kw, rating):
    if category != "external":
        return POWER_KW, "internal_unchanged"
    if rating is None:
        return POWER_KW, "outlet_rating_unknown"
    limit = min(float(rating), 22.0)
    if limit < POWER_KW:
        return limit, "documented_outlet_below_11kw"
    if required_kw <= POWER_KW:
        return POWER_KW, "11kw_sufficient"
    if limit <= POWER_KW:
        return POWER_KW, "documented_outlet_limited_to_11kw"
    return min(required_kw, limit), "higher_power_supported_by_supplied_rating"


def analyze(root=ROOT, output=ALTERNATIVE_OUTPUT, ratings_path=None):
    sessions = load_sessions(root)
    for source in sessions["source"].unique():
        data = pd.read_csv(root / source)
        if "ev_type" not in data:
            raise ValueError(f"{source}: ev_type is missing")
        categories = data["ev_type"].astype("string").str.strip().str.lower()
        if (~categories.isin(["internal", "external"])).any():
            raise ValueError(f"{source}: unknown or missing ev_type")
        sessions.loc[sessions["source"].eq(source), "category"] = categories.to_numpy()
    included = sessions.loc[sessions["status"].eq("pending")].copy()
    ratings = load_ratings(ratings_path, included.loc[included["category"].eq("external"), "evse_id"])
    hospital, _, _, _, _ = load_hospital(root / "hospital_data_with_weekday_weekend.csv")
    grid = hospital.index
    valid = (
        hospital["interval_status"].eq("present") & hospital["hospitalpower_kw"].notna()
        & hospital["powerev_kw"].notna() & hospital["powerhospitalalone"].notna()
    )
    profiles = pd.DataFrame(0.0, index=grid, columns=["baseline_internal", "baseline_external", "alternative_external"])
    results = []
    contributions = {}
    for row in included.itertuples(index=False):
        hours = (row.stop - row.start).total_seconds() / 3600.0
        required_kw = row.recorded_energy_kwh / hours
        rating = ratings.get(row.evse_id)
        power, decision = alternative_power(row.category, required_kw, rating[0] if rating else None)
        base_slots, base_delivered, base_shortfall = reconstruct_session(row.start, row.stop, row.recorded_energy_kwh)
        alt_slots, alt_delivered, alt_shortfall = reconstruct_session(row.start, row.stop, row.recorded_energy_kwh, power)
        baseline_on_valid = 0.0
        alternative_on_valid = 0.0
        for timestamp, energy in base_slots.items():
            nominal = timestamp.tz_convert(None)
            if nominal in grid:
                profiles.at[nominal, f"baseline_{row.category}"] += energy
                if valid.at[nominal]:
                    baseline_on_valid += energy
        for timestamp, energy in alt_slots.items():
            nominal = timestamp.tz_convert(None)
            if nominal in grid and valid.at[nominal]:
                alternative_on_valid += energy
            if row.category == "external" and nominal in grid:
                profiles.at[nominal, "alternative_external"] += energy
        key = (row.evse_id, row.category)
        if key not in contributions:
            contributions[key] = [0, 0.0, 0.0, 0.0, 0.0]
        c = contributions[key]
        c[0] += 1
        c[1] += baseline_on_valid
        c[2] += alternative_on_valid
        c[3] += base_shortfall
        c[4] += alt_shortfall
        results.append({
            "source": row.source, "source_row": row.source_row, "evse_id": row.evse_id,
            "category": row.category, "start_utc": row.start, "stop_utc": row.stop,
            "recorded_energy_kwh": row.recorded_energy_kwh, "duration_seconds": hours * 3600,
            "required_average_kw": required_kw, "outlet_rating_kw": rating[0] if rating else np.nan,
            "rating_evidence": rating[1] if rating else "",
            "baseline_power_kw": POWER_KW, "alternative_power_kw": power, "decision": decision,
            "baseline_delivered_kwh": base_delivered, "baseline_shortfall_kwh": base_shortfall,
            "alternative_delivered_kwh": alt_delivered, "alternative_shortfall_kwh": alt_shortfall,
        })
    results = pd.DataFrame(results)
    baseline_total = profiles["baseline_internal"] + profiles["baseline_external"]
    alt_total = profiles["baseline_internal"] + profiles["alternative_external"]
    prepared = pd.read_csv(OUTPUT / "stage2_load_separation.csv", parse_dates=["Timestamp"]).set_index("Timestamp") if root == ROOT else None
    if prepared is not None and (
        not prepared.index.equals(grid) or
        not np.allclose(baseline_total.to_numpy() / SLOT_HOURS, prepared["reconstructed_ev_power_kw"].to_numpy(), atol=1e-8, rtol=0)
    ):
        raise ValueError("Baseline differs from prepared profile; check input files before comparing")
    session_mask = results["baseline_shortfall_kwh"] > TOLERANCE_KWH
    flagged = results.loc[session_mask].copy()
    flagged["power_group"] = np.select(
        [flagged["required_average_kw"] <= 11.5, flagged["required_average_kw"] <= 22],
        ["above_11_through_11_5_kw", "above_11_5_through_22_kw"],
        default="above_22_kw",
    )
    flagged["additional_seconds_at_11kw"] = flagged["baseline_shortfall_kwh"] / POWER_KW * 3600
    starts = pd.to_datetime(flagged["start_utc"], utc=True)
    stops = pd.to_datetime(flagged["stop_utc"], utc=True)
    flagged["grid_envelope_extra_seconds"] = (
        (stops.dt.ceil("15min") - starts.dt.floor("15min")).dt.total_seconds()
        - flagged["duration_seconds"]
    )
    flagged["fits_expanded_15min_envelope"] = (
        flagged["additional_seconds_at_11kw"] <= flagged["grid_envelope_extra_seconds"] + 1e-7
    )
    flagged["fits_hypothetical_nearest_second_rounding"] = flagged["additional_seconds_at_11kw"] <= 1 + 1e-7
    flagged["fits_hypothetical_nearest_minute_rounding"] = flagged["additional_seconds_at_11kw"] <= 60 + 1e-7
    flagged["fits_hypothetical_001kwh_rounding"] = flagged["baseline_shortfall_kwh"] <= 0.005 + TOLERANCE_KWH
    flagged["fits_hypothetical_001kwh_and_second_rounding"] = flagged["baseline_shortfall_kwh"] <= 0.005 + POWER_KW / 3600 + TOLERANCE_KWH
    groups = flagged.groupby("power_group", sort=False).agg(
        sessions=("power_group", "size"), recorded_energy_kwh=("recorded_energy_kwh", "sum"),
        baseline_shortfall_kwh=("baseline_shortfall_kwh", "sum"),
        median_additional_seconds=("additional_seconds_at_11kw", "median"),
    ).reset_index()
    contribution = pd.DataFrame([
        {
            "evse_id": key[0], "category": key[1], "sessions": vals[0],
            "baseline_reconstructed_kwh_valid_hospital_intervals": vals[1],
            "alternative_reconstructed_kwh_valid_hospital_intervals": vals[2],
            "baseline_shortfall_all_session_windows_kwh": vals[3],
            "alternative_shortfall_all_session_windows_kwh": vals[4],
        }
        for key, vals in contributions.items()
    ]).sort_values("baseline_reconstructed_kwh_valid_hospital_intervals", ascending=False)
    monthly = []
    for month, indices in hospital.groupby(grid.to_period("M")).groups.items():
        all_interval_count = len(indices)
        indices = indices[valid.loc[indices]]
        measured = hospital.loc[indices, "powerev_kw"]
        base = baseline_total.loc[indices] / SLOT_HOURS
        alt = alt_total.loc[indices] / SLOT_HOURS
        monthly.append({
            "month_nominal": str(month), "valid_intervals": len(indices),
            "excluded_intervals": all_interval_count - len(indices),
            "recorded_ev_all_categories_unverified_kwh": measured.sum() * SLOT_HOURS,
            "baseline_external_kwh": profiles.loc[indices, "baseline_external"].sum(),
            "alternative_external_kwh": profiles.loc[indices, "alternative_external"].sum(),
            "baseline_all_categories_kwh": base.sum() * SLOT_HOURS,
            "alternative_all_categories_kwh": alt.sum() * SLOT_HOURS,
            "baseline_minus_recorded_kwh": (base - measured).sum() * SLOT_HOURS,
            "alternative_minus_recorded_kwh": (alt - measured).sum() * SLOT_HOURS,
            "baseline_mae_kw": (base - measured).abs().mean(),
            "alternative_mae_kw": (alt - measured).abs().mean(),
        })
    monthly = pd.DataFrame(monthly)
    summary = []
    for category in ["external", "internal"]:
        part = results.loc[results["category"].eq(category)]
        for scenario in ["baseline", "alternative"]:
            summary.append({
                "category": category, "scenario": scenario, "sessions": len(part),
                "recorded_energy_kwh": part["recorded_energy_kwh"].sum(),
                "reconstructed_delivered_kwh": part[f"{scenario}_delivered_kwh"].sum(),
                "shortfall_kwh": part[f"{scenario}_shortfall_kwh"].sum(),
                "infeasible_sessions": int((part[f"{scenario}_shortfall_kwh"] > TOLERANCE_KWH).sum()),
            })
    output.mkdir(parents=True, exist_ok=True)
    profiles_out = pd.DataFrame({
        "timestamp_nominal": grid,
        "recorded_ev_power_kw_unverified": hospital["powerev_kw"].to_numpy(),
        "baseline_external_power_kw_assumed": profiles["baseline_external"].to_numpy() / SLOT_HOURS,
        "alternative_external_power_kw_assumed": profiles["alternative_external"].to_numpy() / SLOT_HOURS,
        "hospital_interval_status": hospital["interval_status"].to_numpy(),
    })
    profiles_out.to_csv(output / "external_profiles_assumed.csv", index=False)
    results.to_csv(output / "session_comparison.csv", index=False)
    flagged.to_csv(output / "exceeding_11kw_analysis.csv", index=False)
    groups.to_csv(output / "exceeding_11kw_groups.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "scenario_summary.csv", index=False)
    contribution.to_csv(output / "evse_category_contributions.csv", index=False)
    monthly.to_csv(output / "monthly_comparison.csv", index=False)
    metadata = {
        "ratings_file": str(Path(ratings_path).resolve()) if ratings_path else None,
        "rated_evse_count": len(ratings),
        "ratings_status": "user-supplied evidence is not independently verified" if ratings else "no documented outlet ratings supplied; 11 kW remains an assumption",
        "policy": "Internal sessions unchanged. External sessions use 11 kW by default; when needed and rating supplied, use the minimum required average up to the supplied rating and 22 kW. A supplied rating below 11 kW caps the alternative at that rating.",
        "fallback": "Unknown outlet ratings never trigger a 22 kW boost; the 11 kW assumption may also exceed an unknown physical rating.",
        "time": "Hospital naive timestamps provisionally treated as UTC interval starts; missing and ambiguous hospital measurements remain flagged.",
        "comparison": "Recorded EV series coverage is unknown; external energy is not comparable on its own. Total comparison includes unchanged internal 11 kW profiles.",
        "alignment_test": "The expanded 15-minute envelope is a hypothetical upper bound, not a correction or evidence of actual timestamp rounding. The reconstruction integrates actual overlap and does not expand windows.",
        "rounding_tests": "Nearest-second/minute tests assume each endpoint can move by half a unit in the favorable direction. Energy test assumes nearest 0.01 kWh. None of these rounding conventions is verified.",
        "energy": "Original recorded energy and timestamps are unchanged. Shortfalls remain reported whenever chosen power cannot deliver within the recorded session window.",
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(groups.to_string(index=False))
    for category in ["external", "internal"]:
        part = results.loc[results["category"].eq(category)]
        print(category, "sessions", len(part),
              "baseline delivered", part["baseline_delivered_kwh"].sum(),
              "baseline shortfall", part["baseline_shortfall_kwh"].sum(),
              "baseline infeasible", int((part["baseline_shortfall_kwh"] > TOLERANCE_KWH).sum()),
              "alternative delivered", part["alternative_delivered_kwh"].sum(),
              "alternative shortfall", part["alternative_shortfall_kwh"].sum(),
              "alternative infeasible", int((part["alternative_shortfall_kwh"] > TOLERANCE_KWH).sum()))
    print("known rated EVSEs:", len(ratings))
    return results, flagged, groups, contribution, monthly, profiles_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, help="CSV: evse_id,rated_power_kw,evidence; provide authoritative EVSE-specific evidence")
    parser.add_argument("--output", type=Path, default=ALTERNATIVE_OUTPUT, help="Separate output directory for this analysis")
    args = parser.parse_args()
    analyze(output=args.output, ratings_path=args.ratings)

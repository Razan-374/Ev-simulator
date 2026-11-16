"""Independent check of facility-load accounting in corrected Stage 1 outputs.

Recomputes the building profile directly from the raw hospital CSV (powerhospitalalone),
the external profile from the Stage 2 reconstruction, and checks every exported NPZ:

  facility[t] == building[t] + external[t] + sum_i (charge[i,t] - discharge[i,t])

and that neither hospitalpower_kw nor powerev_kw enters the facility load.

Usage: py -3.13 verify_stage1_load_accounting.py <run_dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import stage1_common as c


def raw_profiles(dates):
    raw = pd.read_csv(c.ROOT / "hospital_data_with_weekday_weekend.csv")
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], format="%m/%d/%Y %H:%M")
    raw = raw.drop_duplicates("Timestamp", keep=False)          # ambiguous DST repeats excluded, as in Stage 2
    raw["date"] = raw["Timestamp"].dt.strftime("%Y-%m-%d")
    raw["slot"] = (raw["Timestamp"].dt.hour * 60 + raw["Timestamp"].dt.minute) // 15
    raw = raw[raw["date"].isin(dates)]
    piv = lambda col: raw.pivot_table(index="date", columns="slot", values=col, aggfunc="first").reindex(columns=range(96))
    return piv("powerhospitalalone"), piv("hospitalpower_kw"), piv("powerev_kw")


def main(run_dir):
    profiles = c.load_fixed_load_profiles("weekday")
    fixed = pd.read_csv(c.FIXED_LOAD_CSV, parse_dates=["timestamp_nominal"])
    fixed["date"] = fixed["timestamp_nominal"].dt.strftime("%Y-%m-%d")
    fixed["slot"] = (fixed["timestamp_nominal"].dt.hour * 60 + fixed["timestamp_nominal"].dt.minute) // 15
    report = []
    for name, prof in profiles.items():
        building_raw, total_raw, ev_raw = raw_profiles(prof["source_dates"])
        ext = fixed[fixed["date"].isin(prof["source_dates"])].pivot_table(
            index="date", columns="slot", values="external_charging_kw_assumed_11kw").reindex(columns=range(96))
        b_ind, e_ind = building_raw.mean(axis=0).to_numpy(), ext.mean(axis=0).to_numpy()
        report.append({
            "profile": name, "days": len(prof["source_dates"]),
            "building_vs_raw_powerhospitalalone_max_abs_kw": float(np.max(np.abs(prof["building_kw"] - b_ind))),
            "external_vs_stage2_reconstruction_max_abs_kw": float(np.max(np.abs(prof["external_kw"] - e_ind))),
            "raw_measured_ev_powerev_kw_mean_profile_max_kw": float(ev_raw.mean(axis=0).max()),
            "building_vs_hospitalpower_kw_max_abs_kw": float(np.max(np.abs(prof["building_kw"]
                                                                          - total_raw.mean(axis=0).to_numpy()))),
            "external_daily_energy_kwh": float(prof["external_kw"].sum() * c.SLOT_HOURS),
        })
    print(pd.DataFrame(report).to_string(index=False))

    worst = 0.0
    checked = 0
    for path in sorted(Path(run_dir).glob("*_power*kW/*.npz")):
        z = np.load(path, allow_pickle=False)
        prof = profiles[str(z["base_profile"])]
        assert np.allclose(z["building_load_kw"], prof["building_kw"]) and np.allclose(z["external_load_kw"], prof["external_kw"])
        assert np.allclose(z["fixed_load_kw"], z["building_load_kw"] + z["external_load_kw"])
        for k in range(len(z["n_internal"])):
            if np.isnan(z["facility_load_kw"][k]).any():
                continue
            net = np.nansum(z["charge_power_kw"][k] - z["discharge_power_kw"][k], axis=0)
            err = np.max(np.abs(z["facility_load_kw"][k] - (z["building_load_kw"] + z["external_load_kw"] + net)))
            worst = max(worst, err)
            checked += 1
    print(f"NPZ facility-balance checks: {checked} solved N-profiles, max |error| = {worst:.3e} kW")


if __name__ == "__main__":
    main(sys.argv[1])

"""Create V1G master internal fleets by replaying the unchanged V2G sampler."""

import importlib.util
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
V2G_PATH = os.path.join(ROOT, "sensetivity analysis with peak limits v2g - correct soc11.py")
MASTER_DIR = os.path.join(ROOT, "master_fleets")
REFERENCE_RESULTS = os.path.join(
    ROOT,
    "hosting_capacity_results_A_B_v2g_average_worst_blocked_sweep_10seed_90s_11kw.csv",
)


def load_v2g_module():
    spec = importlib.util.spec_from_file_location("v2g_reference", V2G_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(module, seed):
    module.useCase = 4
    module.batteryCapacity = 60.0
    module.energyEfficiency = 90.0
    module.isSlow = 1
    module.charge_efficiency = 0.9
    module.power_V1GLower = -10.0
    module.power_V1GUppper = 11.0
    module.internal_chargers_available = 17
    module.public_chargers_available = 4
    module.power_General = 7.0
    module.ngen_Chargers = 4
    module.n_V0G_Veh = 15
    module.n_fastVeh = 0
    module.n_slowVeh = 15
    module.nfast_Chargers = 0
    module.nslow_Chargers = 0
    module.n_iter = 25
    module.Maxtime_Interval = 1440
    module.opt_includeSolar = 0
    module.solarCurve = np.zeros(module.SLOTS_PER_DAY)
    module.base_Load = np.zeros(module.SLOTS_PER_DAY)
    module.unmanaged_Load = np.zeros(module.SLOTS_PER_DAY)
    module.slow_Load_1h_avg = np.zeros(module.SLOTS_PER_DAY)
    module.fast_Load_1h_avg = np.zeros(module.SLOTS_PER_DAY)
    module.gen_Load_1h_avg = np.zeros(module.SLOTS_PER_DAY)
    module.available_internal_chargers_by_slot = np.full(module.SLOTS_PER_DAY, 17, dtype=int)
    module.simulation_day_start = pd.Timestamp("2026-02-16")
    np.random.seed(seed)


def create_fleet(module, seed):
    configure(module, seed)
    module.Unmanaged_usecase(module.useCase)
    module.initiate_V1G(100)

    fleet = pd.DataFrame({
        "seed": seed,
        "EV_id": np.arange(100),
        "start_minute": np.asarray(module.v1g_startingTime, dtype=int) * 15,
        "start_slot": np.asarray(module.v1g_startingTime, dtype=int),
        "end_slot": np.asarray(module.v1g_endingTime, dtype=int),
        "dwell_slots": np.asarray(module.v1g_dwell_slots, dtype=int),
        "energy_required_kwh": np.asarray(module.v1g_energy_required_kwh, dtype=float),
        "starting_SOC": np.asarray(module.v1g_startingSOC, dtype=float),
    })
    fleet["fleet_hash"] = "pending"
    digest = __import__("hashlib").sha256(fleet.to_csv(index=False).encode("utf-8")).hexdigest()
    fleet["fleet_hash"] = digest
    return fleet


def verify_fleet(fleet, reference):
    """Verify nested fleet aggregates against frozen V2G rows for one seed."""
    seed = int(fleet["seed"].iloc[0])
    reference = reference[reference["seed"] == seed].copy()
    reference = reference[reference["N_internal"] > 0]
    reference = reference.sort_values("N_internal").drop_duplicates("N_internal")
    checks = []
    for _, row in reference.iterrows():
        n = int(row["N_internal"])
        subset = fleet.iloc[:n]
        required = float(subset["energy_required_kwh"].sum())
        unavoidable = 0.0
        window_infeasible = 0
        for _, ev in subset.iterrows():
            max_energy = float(ev["dwell_slots"]) * 11.0 * 0.25 * 0.9
            shortfall = max(0.0, float(ev["energy_required_kwh"]) - max_energy)
            if shortfall > 1e-6:
                window_infeasible += 1
                unavoidable += shortfall
        checks.append({
            "seed": seed,
            "N_internal": n,
            "required_match": abs(required - float(row["required_total_kwh"])) <= 1e-6,
            "unavoidable_match": abs(unavoidable - float(row["unavoidable_unmet_kwh"])) <= 1e-5,
            "window_match": window_infeasible == int(row["window_infeasible_count"]),
        })
    checks_frame = pd.DataFrame(checks)
    verified = bool(not checks_frame.empty and checks_frame.all(axis=None))
    return verified, checks_frame


def main():
    if not os.path.exists(REFERENCE_RESULTS):
        raise FileNotFoundError(f"Frozen V2G reference results not found: {REFERENCE_RESULTS}")
    os.makedirs(MASTER_DIR, exist_ok=True)
    module = load_v2g_module()
    reference = pd.read_csv(REFERENCE_RESULTS)
    provenance = []
    for seed in range(10):
        fleet = create_fleet(module, seed)
        verified, checks = verify_fleet(fleet, reference)
        fleet["v2g_reference_seed"] = seed
        fleet["fleet_replay_verified"] = verified
        fleet["v2g_code_version_hash"] = hashlib.sha256(open(V2G_PATH, "rb").read()).hexdigest()
        fleet["input_data_hash"] = hashlib.sha256(
            (open(os.path.join(ROOT, "session_behaviour_clusters.csv"), "rb").read()
             + open(os.path.join(ROOT, "external.csv"), "rb").read())
        ).hexdigest()
        path = os.path.join(MASTER_DIR, f"master_internal_fleet_seed{seed}.csv")
        fleet.to_csv(path, index=False)
        provenance.append({
            "seed": seed,
            "fleet_hash": fleet["fleet_hash"].iloc[0],
            "fleet_replay_verified": verified,
            "checks_tested": len(checks),
            "path": path,
        })
        print(f"seed={seed} | verified={verified} | hash={fleet['fleet_hash'].iloc[0]} | path={path}")
    pd.DataFrame(provenance).to_csv(
        os.path.join(MASTER_DIR, "fleet_replay_provenance.csv"), index=False
    )


if __name__ == "__main__":
    main()

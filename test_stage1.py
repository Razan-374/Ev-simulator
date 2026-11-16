"""Focused validation tests for the corrected Stage 1 V1G/V2G implementation.

Run: py -3.13 -m unittest -v test_stage1
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import stage1_common as c
from stage1_v1g_model import solve_v1g
from stage1_v2g_model import solve_v2g

FAST_SOLVER = {"time_limit_sec": 60, "gap_rel": 1e-6, "gap_abs": 1e-4}


def flat_profile(building=1000.0, external=0.0, peak_slots=(), peak_kw=0.0):
    building_kw = np.full(c.SLOTS_PER_DAY, building, dtype=float)
    for t in peak_slots:
        building_kw[t] = peak_kw
    external_kw = np.full(c.SLOTS_PER_DAY, external, dtype=float)
    fixed = building_kw + external_kw
    headroom, excess = c.site_headroom(fixed)
    return {"building_kw": building_kw, "external_kw": external_kw, "fixed_kw": fixed,
            "headroom_kw": headroom, "fixed_excess_kw": excess}


def fleet(arrivals, dwells, energies, emergency=None):
    energies = np.asarray(energies, float)
    return {"arrival_slot": np.asarray(arrivals, int), "dwell_slots": np.asarray(dwells, int),
            "energy_kwh": energies, "soc0": 1.0 - c.CHARGE_EFFICIENCY * energies / c.BATTERY_CAPACITY_KWH,
            "emergency": np.zeros(len(energies), bool) if emergency is None else np.asarray(emergency, bool),
            "session_id": np.array([f"s{i}" for i in range(len(energies))])}


def evaluate(model, arr, profile, avail, power, arrays):
    return c.evaluate_solution(arrays["charge_kw"], arrays["discharge_kw"], arr, profile, avail, power,
                               unmet_kwh_model=arrays["unmet_kwh"], total_load_model=arrays["total_load_kw"],
                               allow_discharge=(model == "v2g"))[0]


class LoadAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = c.load_fixed_load_profiles("weekday")
        cls.fixed = pd.read_csv(c.FIXED_LOAD_CSV, parse_dates=["timestamp_nominal"])
        cls.hospital = pd.read_csv(c.ROOT / "hospital_data_with_weekday_weekend.csv")

    def test_fixed_is_building_plus_external_every_slot(self):
        for profile in self.profiles.values():
            np.testing.assert_allclose(profile["fixed_kw"], profile["building_kw"] + profile["external_kw"], atol=1e-9)
            self.assertEqual(len(profile["fixed_kw"]), c.SLOTS_PER_DAY)
            self.assertFalse(np.isnan(profile["fixed_kw"]).any())

    def test_building_excludes_measured_ev_power(self):
        identity = self.hospital["hospitalpower_kw"] - self.hospital["powerev_kw"] - self.hospital["powerhospitalalone"]
        self.assertLess(identity.abs().max(), 1e-6)
        # The Stage 1 building profile is powerhospitalalone, not hospitalpower_kw.
        day = self.profiles["worst_peak"]["source_dates"][0]
        rows = self.fixed[self.fixed["timestamp_nominal"].dt.strftime("%Y-%m-%d") == day]
        np.testing.assert_allclose(self.profiles["worst_peak"]["building_kw"],
                                   rows["building_load_kw_provisional"].to_numpy(), atol=1e-9)

    def test_worst_day_is_maximum_fixed_peak_weekday(self):
        worst = self.profiles["worst_peak"]
        self.assertEqual(c.day_type_of(worst["source_dates"][0]), "weekday")
        self.assertGreaterEqual(worst["fixed_kw"].max(), self.profiles["average"]["fixed_kw"].max())

    def test_headroom_accounts_for_external_and_reports_excess(self):
        headroom, excess = c.site_headroom(np.array([1500.0, 2000.0, 2100.0]))
        np.testing.assert_allclose(headroom, [500.0, 0.0, 0.0])
        np.testing.assert_allclose(excess, [0.0, 0.0, 100.0])


class FleetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = c.fleet_config(3, "weekday")
        self.fleet, self.path, self.meta = c.ensure_fleet(self.config, self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_internal_only_and_no_duplicates(self):
        pool, info = c.load_internal_pool("weekday")
        internal = pd.read_csv(c.INTERNAL_SESSIONS_CSV)
        self.assertTrue(set(self.fleet["session_id"]) <= set(internal["session_id"]))
        external = pd.read_csv(c.ROOT / "stage2_data" / "shared_inputs" / "stage2_external_sessions.csv")
        self.assertFalse(set(self.fleet["session_id"]) & set(external["session_id"]))
        self.assertEqual(self.fleet["session_id"].nunique(), len(self.fleet))
        self.assertFalse(self.meta["info"]["sampled_with_replacement"])
        self.assertTrue((self.fleet["session_day_type"] == "weekday").all())

    def test_random_order_not_sorted_and_nested(self):
        self.assertFalse(self.fleet["arrival_minute"].is_monotonic_increasing)
        first10 = c.fleet_arrays(self.fleet, 10)["session_id"]
        first20 = c.fleet_arrays(self.fleet, 20)["session_id"]
        np.testing.assert_array_equal(first10, first20[:10])

    def test_triplets_match_pool_rows(self):
        pool, _ = c.load_internal_pool("weekday")
        merged = self.fleet.merge(pool[["session_id", "arrival_minute", "energy_recorded_kwh", "dwell_minutes_recorded"]],
                                  on="session_id", suffixes=("", "_pool"))
        self.assertEqual(len(merged), len(self.fleet))
        np.testing.assert_array_equal(merged["arrival_minute"], merged["arrival_minute_pool"])
        np.testing.assert_allclose(merged["energy_recorded_kwh"], merged["energy_recorded_kwh_pool"])
        np.testing.assert_allclose(merged["dwell_minutes_recorded"], merged["dwell_minutes_recorded_pool"])

    def test_reproducible_cache_and_config_sensitive_key(self):
        again, path2, _ = c.ensure_fleet(self.config, self.tmp.name)
        self.assertEqual(path2, self.path)
        pd.testing.assert_frame_equal(again, self.fleet)
        rebuilt, _ = c.build_fleet(self.config)
        np.testing.assert_array_equal(rebuilt["session_id"], self.fleet["session_id"])
        _, friday_path, _ = c.ensure_fleet(c.fleet_config(3, "friday"), self.tmp.name)
        _, battery_path, _ = c.ensure_fleet(c.fleet_config(3, "weekday", battery_kwh=75), self.tmp.name)
        self.assertEqual(len({self.path, friday_path, battery_path}), 3)

    def test_concurrent_cache_creation_is_safe(self):
        from concurrent.futures import ThreadPoolExecutor
        config = c.fleet_config(5, "weekday")
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda _: c.ensure_fleet(config, self.tmp.name)[2]["fleet_hash"], range(4)))
        self.assertEqual(len(set(results)), 1)

    def test_tampered_cache_is_rejected(self):
        frame = pd.read_csv(self.path)
        frame.loc[0, "energy_required_kwh"] += 1.0
        frame.to_csv(self.path, index=False)
        with self.assertRaises(RuntimeError):
            c.ensure_fleet(self.config, self.tmp.name)

    def test_soc_and_energy_convention(self):
        expected = 1.0 - c.CHARGE_EFFICIENCY * self.fleet["energy_required_kwh"] / c.BATTERY_CAPACITY_KWH
        np.testing.assert_allclose(self.fleet["starting_soc"], expected)
        self.assertTrue((self.fleet["starting_soc"] >= 0).all() and (self.fleet["starting_soc"] <= 1).all())

    def test_emergency_flag_from_user_ids(self):
        user = str(self.fleet["source_user_id"].iloc[0])
        flagged, _, _ = c.ensure_fleet(c.fleet_config(3, "weekday", emergency_user_ids=[user]), self.tmp.name)
        np.testing.assert_array_equal(flagged["emergency_vehicle"], flagged["source_user_id"].astype(str) == user)


class ModelPhysicsTests(unittest.TestCase):
    power = 11

    def solve(self, model, arr, profile, avail, **kw):
        if model == "v1g":
            return solve_v1g(arr, profile, avail, self.power, FAST_SOLVER, **kw)
        return solve_v2g(arr, profile, avail, self.power, FAST_SOLVER, **kw)

    def test_energy_soc_power_balance_both_models(self):
        arr = fleet([10, 20, 80], [16, 30, 40], [20.0, 35.0, 50.0])
        profile = flat_profile(1200, 30, peak_slots=range(40, 60), peak_kw=1500)
        avail = np.full(c.SLOTS_PER_DAY, 3)
        for model in ("v1g", "v2g"):
            outcome, arrays = self.solve(model, arr, profile, avail)
            self.assertEqual(outcome["solve_class"], "optimal", model)
            m = evaluate(model, arr, profile, avail, self.power, arrays)
            self.assertTrue(m["validation_passed"], (model, m["validation_failures"]))
            self.assertLess(m["total_unmet_kwh"], c.SERVICE_TOLERANCE_KWH, model)
            np.testing.assert_allclose(arrays["total_load_kw"],
                                       profile["building_kw"] + profile["external_kw"]
                                       + (arrays["charge_kw"] - arrays["discharge_kw"]).sum(axis=0), atol=1e-3)

    def test_v2g_discharge_respects_reserve_and_reduces_peak(self):
        arr = fleet([0, 0], [96, 96], [6.0, 6.0])      # high arrival SoC, long stays
        profile = flat_profile(1000, 0, peak_slots=range(40, 44), peak_kw=1100)
        avail = np.full(c.SLOTS_PER_DAY, 2)
        outcome, arrays = self.solve("v2g", arr, profile, avail)
        m = evaluate("v2g", arr, profile, avail, self.power, arrays)
        self.assertTrue(m["validation_passed"], m["validation_failures"])
        self.assertGreater(m["ev_discharge_energy_kwh"], 0.0)
        self.assertGreater(m["peak_reduction_vs_fixed_kw"], 0.0)
        self.assertGreaterEqual(m["soc_min_observed"], c.SOC_MIN - c.SOC_TOLERANCE)
        soc = c.soc_trajectory(arrays["charge_kw"], arrays["discharge_kw"], arr)
        discharging = arrays["discharge_kw"] > 1e-6
        self.assertTrue((soc[discharging] >= c.V2G_DISCHARGE_RESERVE_SOC - 1e-6).all())
        self.assertLess(np.minimum(arrays["charge_kw"], arrays["discharge_kw"]).max(), 1e-6)

    def test_v2g_low_soc_vehicle_does_not_discharge_below_reserve(self):
        arr = fleet([0], [96], [50.0])                  # arrival SoC 0.25 < reserve
        profile = flat_profile(1000, 0, peak_slots=[10], peak_kw=1100)
        avail = np.full(c.SLOTS_PER_DAY, 1)
        _, arrays = self.solve("v2g", arr, profile, avail)
        soc = c.soc_trajectory(arrays["charge_kw"], arrays["discharge_kw"], arr)
        discharging = arrays["discharge_kw"][0] > 1e-6
        self.assertTrue((soc[0][discharging] >= c.V2G_DISCHARGE_RESERVE_SOC - 1e-6).all())

    def test_emergency_vehicle_never_discharges(self):
        arr = fleet([0, 0], [96, 96], [6.0, 6.0], emergency=[True, False])
        profile = flat_profile(1000, 0, peak_slots=range(40, 44), peak_kw=1100)
        avail = np.full(c.SLOTS_PER_DAY, 2)
        _, arrays = self.solve("v2g", arr, profile, avail)
        self.assertLess(arrays["discharge_kw"][0].max(), 1e-6)
        self.assertGreater(arrays["discharge_kw"][1].max(), 1e-6)

    def test_outlet_availability_binds(self):
        arr = fleet([0] * 4, [8] * 4, [20.0] * 4)        # 4 EVs, 2 h window, 1 outlet
        profile = flat_profile()
        avail = np.full(c.SLOTS_PER_DAY, 1)
        for model in ("v1g", "v2g"):
            outcome, arrays = self.solve(model, arr, profile, avail)
            if model == "v1g":
                m = evaluate(model, arr, profile, avail, self.power, arrays)
                self.assertTrue(m["validation_passed"], m["validation_failures"])
                self.assertLessEqual(m["max_active_evs"], 1)
                self.assertGreater(m["avoidable_unmet_kwh"], 1.0)
                self.assertFalse(m["service_requirement_met"])
            else:  # hard service requirement cannot be met -> infeasible, not silently passed
                self.assertEqual(outcome["solve_class"], "infeasible")
                self.assertIsNone(arrays)

    def test_window_infeasible_energy_is_unavoidable_not_failure(self):
        arr = fleet([30], [2], [20.0])                   # 30 min at 11 kW -> 5.5 kWh max
        profile = flat_profile()
        avail = np.full(c.SLOTS_PER_DAY, 1)
        for model in ("v1g", "v2g"):
            _, arrays = self.solve(model, arr, profile, avail)
            m = evaluate(model, arr, profile, avail, self.power, arrays)
            self.assertAlmostEqual(m["unavoidable_unmet_kwh"], 14.5, places=3)
            self.assertTrue(m["service_requirement_met"], model)

    def test_headroom_and_fixed_excess(self):
        # Fixed load exceeds the cap in slots 40-47; an EV can only charge outside them.
        arr = fleet([36], [16], [10.0])
        profile = flat_profile(1980, 0, peak_slots=range(40, 48), peak_kw=2100)
        avail = np.full(c.SLOTS_PER_DAY, 1)
        for model in ("v1g", "v2g"):
            outcome, arrays = self.solve(model, arr, profile, avail)
            self.assertIn(outcome["solve_class"], ("optimal",), model)
            m = evaluate(model, arr, profile, avail, self.power, arrays)
            self.assertTrue(m["validation_passed"], (model, m["validation_failures"]))
            net = (arrays["charge_kw"] - arrays["discharge_kw"]).sum(axis=0)
            self.assertTrue((net[40:48] <= 1e-6).all(), model)
            self.assertEqual(m["fixed_excess_slots"], 8)
            self.assertAlmostEqual(m["fixed_excess_max_kw"], 100.0)

    def test_fixed_excess_alone_is_not_infeasible(self):
        # Fixed demand above the cap all day: N=0 and EVs needing no import are feasible;
        # an EV that needs import is cap-blocked and this is reported, not hidden.
        profile = flat_profile(2100, 50)
        avail = np.full(c.SLOTS_PER_DAY, 17)
        empty = fleet([], [], [])
        m0, _ = c.evaluate_solution(np.zeros((0, 96)), np.zeros((0, 96)), empty, profile, avail, 11,
                                    total_load_model=profile["fixed_kw"])
        self.assertTrue(m0["validation_passed"])
        self.assertEqual(m0["fixed_excess_slots"], 96)
        self.assertAlmostEqual(m0["fixed_excess_max_kw"], 150.0)

        arr = fleet([0], [96], [5.0])
        outcome1, arrays1 = self.solve("v1g", arr, profile, avail)
        self.assertEqual(outcome1["solve_class"], "optimal")
        m1 = evaluate("v1g", arr, profile, avail, self.power, arrays1)
        self.assertTrue(m1["validation_passed"], m1["validation_failures"])
        self.assertAlmostEqual(m1["cap_blocked_shortfall_kwh"], 5.0, places=6)
        self.assertFalse(m1["service_requirement_met"])

        outcome2, arrays2 = solve_v2g(arr, profile, avail, self.power, FAST_SOLVER, diagnose_infeasible=True)
        self.assertEqual(outcome2["solve_class"], "infeasible")
        self.assertFalse(outcome2["usable_solution"])
        self.assertEqual(outcome2["solution_source"], "soft_service_diagnostic")
        m2 = evaluate("v2g", arr, profile, avail, self.power, arrays2)
        self.assertTrue(m2["validation_passed"], m2["validation_failures"])
        self.assertAlmostEqual(m2["avoidable_unmet_kwh"], 5.0, places=3)

    def test_v2v_cannot_bypass_charger_or_service_limits(self):
        # Zero headroom everywhere: V1G cannot charge; V2G may only charge by discharging
        # another EV, with each EV still bounded by its own charger power and SoC reserve.
        profile = flat_profile(2000, 0)
        avail = np.full(c.SLOTS_PER_DAY, 2)
        arr = fleet([0, 0], [96, 96], [3.0, 20.0])
        outcome, arrays = self.solve("v2g", arr, profile, avail)
        if arrays is not None:
            m = evaluate("v2g", arr, profile, avail, self.power, arrays)
            self.assertTrue(m["validation_passed"], m["validation_failures"])
            self.assertLessEqual(m["ev_net_import_peak_kw"], 1e-3)
            self.assertLessEqual(arrays["charge_kw"].max(), self.power + 1e-6)
        _, arrays1 = self.solve("v1g", arr, profile, avail)
        m1 = evaluate("v1g", arr, profile, avail, self.power, arrays1)
        self.assertAlmostEqual(m1["delivered_total_kwh"], 0.0, places=4)

    def test_n0_has_no_artificial_peak_reduction(self):
        profile = flat_profile(1500, 20, peak_slots=[50], peak_kw=1800)
        empty = fleet([], [], [])
        m, _ = c.evaluate_solution(np.zeros((0, 96)), np.zeros((0, 96)), empty, profile,
                                   np.full(96, 17), 11, total_load_model=profile["fixed_kw"])
        self.assertEqual(m["peak_reduction_vs_fixed_kw"], 0.0)
        self.assertTrue(m["validation_passed"])


class FormulationEquivalenceTests(unittest.TestCase):
    """The SoC recursion must reproduce the original cumulative-sum formulation."""

    def compare(self, arr, profile, avail, power):
        for model in ("v1g", "v2g"):
            results = {}
            for form in ("cumulative", "recursive"):
                if model == "v1g":
                    outcome, arrays = solve_v1g(arr, profile, avail, power, FAST_SOLVER, soc_form=form)
                else:
                    outcome, arrays = solve_v2g(arr, profile, avail, power, FAST_SOLVER, soc_form=form)
                self.assertEqual(outcome["solve_class"], "optimal", (model, form))
                m = evaluate(model, arr, profile, avail, power, arrays)
                self.assertTrue(m["validation_passed"], (model, form, m["validation_failures"]))
                results[form] = (outcome["objective_value"], m["facility_peak_kw"], m["total_unmet_kwh"])
            np.testing.assert_allclose(results["cumulative"], results["recursive"], atol=1e-3, err_msg=model)

    def test_synthetic_instances(self):
        arr = fleet([0, 5, 30, 70, 90], [96, 40, 30, 20, 30], [6.0, 30.0, 45.0, 12.0, 25.0])
        profile = flat_profile(1500, 20, peak_slots=range(36, 60), peak_kw=1650)
        self.compare(arr, profile, np.full(c.SLOTS_PER_DAY, 3), 11)
        self.compare(arr, profile, np.full(c.SLOTS_PER_DAY, 2), 22)

    def test_real_fleet_and_load(self):
        profiles = c.load_fixed_load_profiles("weekday")
        with tempfile.TemporaryDirectory() as tmp:
            fleet_df, _, _ = c.ensure_fleet(c.fleet_config(0, "weekday"), tmp)
        arr = c.fleet_arrays(fleet_df, 8)
        for name in ("average", "worst_peak"):
            self.compare(arr, profiles[name], np.full(c.SLOTS_PER_DAY, 9), 11)


class OutcomeCategoryTests(unittest.TestCase):
    def test_categories_are_distinct(self):
        from run_stage1 import outcome_category
        cases = [
            ({"service_feasible": True, "validation_passed": True}, {"solve_class": "optimal"}, "optimal_service_feasible"),
            ({"service_feasible": True, "validation_passed": True}, {"solve_class": "time_limit_feasible"},
             "time_limited_service_feasible"),
            ({"service_feasible": False, "validation_passed": True}, {"solve_class": "optimal"},
             "optimal_service_infeasible"),
            ({"service_feasible": False, "validation_passed": True},
             {"solve_class": "time_limit_feasible", "service_failure_proven": True},
             "time_limited_service_infeasible_proven_by_bound"),
            ({"service_feasible": False, "validation_passed": True}, {"solve_class": "time_limit_feasible"},
             "time_limited_service_unresolved"),
            ({"service_feasible": False}, {"solve_class": "infeasible"}, "service_infeasible_proven"),
            ({"service_feasible": False}, {"solve_class": "time_limit_no_solution"}, "unresolved_no_solution"),
            ({"service_feasible": True, "solution_promoted_from_soft_resolve": True},
             {"solve_class": "time_limit_no_solution"}, "time_limited_service_feasible_via_soft_resolve"),
            ({"service_feasible": False, "validation_passed": False}, {"solve_class": "optimal"}, "constraint_violation"),
        ]
        for row, outcome, expected in cases:
            self.assertEqual(outcome_category(row, outcome), expected)

    def test_infeasible_diagnostic_never_promoted(self):
        arr = fleet([0] * 4, [8] * 4, [20.0] * 4)
        outcome, arrays = solve_v2g(arr, flat_profile(), np.full(96, 1), 11, FAST_SOLVER, diagnose_infeasible=True)
        self.assertEqual(outcome["solve_class"], "infeasible")
        self.assertFalse(outcome["usable_solution"])


class EvaluationTests(unittest.TestCase):
    def test_validation_detects_injected_violations(self):
        arr = fleet([0], [4], [5.0], emergency=[True])
        profile = flat_profile()
        charge = np.zeros((1, 96)); discharge = np.zeros((1, 96))
        charge[0, 10] = 5.0                                # outside window
        discharge[0, 1] = 5.0                              # emergency discharge
        charge[0, 1] = 5.0                                 # simultaneous
        m, _ = c.evaluate_solution(charge, discharge, arr, profile, np.full(96, 0), 11, allow_discharge=True)
        for key in ("outside_window_power", "emergency_discharge", "simultaneous_charge_discharge",
                    "outlet_availability"):
            self.assertIn(key, m["validation_failures"])

    def test_hosting_capacity_contiguous_and_non_monotonic(self):
        ns = np.array([0, 5, 10, 15, 20])
        passed = np.array([True, True, False, True, False])
        self.assertEqual(c.contiguous_capacity(ns, passed), 5)
        self.assertEqual(c.non_monotonic_ns(ns, passed), [15])
        df = pd.DataFrame({"seed": [0, 0, 0, 1, 1, 1], "N_internal": [0, 5, 10] * 2, "g": "x",
                           "service_feasible": [True, True, True, True, False, True]})
        _, summary = c.probabilistic_capacity(df, ["g"])
        self.assertEqual(summary.loc[0, "HC_100"], 0)
        self.assertEqual(summary.loc[0, "HC_100_any_pass"], 10)
        self.assertEqual(summary.loc[0, "HC_100_non_monotonic_N"], "10")

    def test_cbc_log_parsing(self):
        info = c.parse_cbc_log("Result - Stopped on time limit\nObjective value:  1600.5\nLower bound:  1590.25\n")
        self.assertEqual(info["log_result"], "time_limit")
        self.assertAlmostEqual(info["best_bound"], 1590.25)


if __name__ == "__main__":
    unittest.main()

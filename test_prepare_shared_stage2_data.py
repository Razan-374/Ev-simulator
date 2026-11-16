import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prepare_shared_stage2_data import build_shared_inputs


class SharedInputTests(unittest.TestCase):
    def test_external_only_fixed_load_and_unknown_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({
                "Timestamp": [
                    "2024-01-01 00:00", "2024-01-01 00:15",
                    "2024-01-01 00:15", "2024-01-01 00:45",
                ],
                "hospitalpower_kw": [100] * 4, "powerhospitalalone": [96] * 4,
                "powerev_kw": [4] * 4, "EnergyevKWh": [1] * 4,
            }).to_csv(root / "hospital_data_with_weekday_weekend.csv", index=False)
            pd.DataFrame([
                {"evseId": "A", "Arrival Time": "2023-12-31 23:55:00+00:00",
                 "Departure Time": "2024-01-01 00:20:00+00:00", "kWhDelivered": "4.00",
                 "ev_type": "external", "user_id": "1"},
                {"evseId": "B", "Arrival Time": "2024-01-01 00:10:00+00:00",
                 "Departure Time": "2024-01-01 01:10:00+00:00", "kWhDelivered": "1.00",
                 "ev_type": "internal", "user_id": "2"},
                {"evseId": "B", "Arrival Time": "2024-01-02 00:00:00+00:00",
                 "Departure Time": "2024-01-02 01:00:00+00:00", "kWhDelivered": "2.00",
                 "ev_type": "internal", "user_id": "2"},
            ]).to_csv(root / "session_behaviour_clusters.csv", index=False)
            pd.DataFrame([
                {"evseId": "C", "chargingStart ISO-8601": "2024-01-01 00:45:00+00:00",
                 "chargingStop ISO-8601": "2024-01-01 01:00:00+00:00",
                 "energyConsumptionInKWh": "4.00", "ev_type": "external"},
                {"evseId": "C", "chargingStart ISO-8601": "2024-01-02 00:00:00+00:00",
                 "chargingStop ISO-8601": "2024-01-02 00:15:00+00:00",
                 "energyConsumptionInKWh": "4.00", "ev_type": "external"},
                {"evseId": "C", "chargingStart ISO-8601": "2024-01-01 00:15:00+00:00",
                 "chargingStop ISO-8601": "2024-01-01 00:30:00+00:00",
                 "energyConsumptionInKWh": "-1.00", "ev_type": "external"},
            ]).to_csv(root / "external.csv", index=False)
            output = root / "shared"
            fixed, internal, external, excluded, outlets, metrics = build_shared_inputs(root, output)
            self.assertEqual(len(fixed), 4)
            self.assertEqual(fixed["hospital_interval_status"].tolist(), ["present", "ambiguous_duplicate", "missing", "present"])
            self.assertAlmostEqual(fixed.loc[0, "fixed_load_kw_provisional"], 107)
            self.assertAlmostEqual(fixed.loc[3, "fixed_load_kw_provisional"], 107)
            self.assertTrue(pd.isna(fixed.loc[1, "fixed_load_kw_provisional"]))
            self.assertTrue(pd.isna(fixed.loc[2, "fixed_load_kw_provisional"]))
            self.assertTrue(fixed["fixed_load_status"].eq("unresolved_external_records_and_provisional_alignment").all())
            self.assertEqual(fixed["external_infeasible_session_window_count"].tolist(), [0, 0, 0, 1])
            self.assertEqual(outlets["observed_external_session_window_count"].tolist(), [1, 1, 0, 1])
            self.assertTrue(outlets["verified_shared_outlets_occupied"].isna().all())
            self.assertTrue(outlets["verified_controllable_outlets_available"].isna().all())
            self.assertEqual(len(internal), 1)
            self.assertTrue(internal.iloc[0]["crosses_grid_end"])
            self.assertEqual(internal.iloc[0]["recorded_energy_kwh"], "1.00")
            self.assertEqual(external.iloc[0]["recorded_start"], "2023-12-31 23:55:00+00:00")
            self.assertEqual(external.iloc[0]["recorded_energy_kwh"], "4.00")
            self.assertTrue(external.iloc[0]["crosses_grid_start"])
            self.assertEqual(external.iloc[1]["reconstruction_status"], "infeasible_11kw")
            self.assertAlmostEqual(external.iloc[1]["shortfall_full_session_kwh"], 1.25)
            self.assertEqual(excluded.iloc[0]["exclusion_reason"], "outside_hospital_grid")
            self.assertTrue(excluded.iloc[1]["infeasible_at_11kw_diagnostic"])
            self.assertAlmostEqual(excluded.iloc[1]["diagnostic_11kw_shortfall_kwh"], 1.25)
            self.assertEqual(metrics["excluded_outside_grid_infeasible_at_11kw"], 1)
            self.assertEqual(metrics["excluded_invalid_or_conflicting"], 1)
            self.assertAlmostEqual(metrics["external_grid_energy_balance_error_kwh"], 0)
            with (output / "stage2_shared_metadata.json").open(encoding="utf-8") as file:
                self.assertEqual(json.load(file)["status"], "provisional_not_ready_for_optimizer_integration")


if __name__ == "__main__":
    unittest.main()

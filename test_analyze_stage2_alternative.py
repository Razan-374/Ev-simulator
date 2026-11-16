import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze_stage2_alternative import alternative_power, analyze, load_ratings


class AlternativeTests(unittest.TestCase):
    def test_power_selection_requires_known_outlet_rating(self):
        self.assertEqual(alternative_power("external", 16, None), (11, "outlet_rating_unknown"))
        self.assertEqual(alternative_power("external", 10, 22), (11, "11kw_sufficient"))
        self.assertEqual(alternative_power("external", 16, 22), (16, "higher_power_supported_by_supplied_rating"))
        self.assertEqual(alternative_power("external", 24, 50), (22, "higher_power_supported_by_supplied_rating"))
        self.assertEqual(alternative_power("external", 16, 7), (7, "documented_outlet_below_11kw"))
        self.assertEqual(alternative_power("internal", 16, 22), (11, "internal_unchanged"))

    def test_rating_evidence_and_ids_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.csv"
            pd.DataFrame([{"evse_id": "A", "rated_power_kw": 22, "evidence": " "}]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_ratings(path, {"A"})
            pd.DataFrame([{"evse_id": "B", "rated_power_kw": 22, "evidence": "spec"}]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_ratings(path, {"A"})

    def test_profiles_only_boost_documented_external_evse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({
                "Timestamp": pd.date_range("2024-01-01", periods=4, freq="15min"),
                "hospitalpower_kw": [100] * 4, "powerhospitalalone": [96] * 4,
                "powerev_kw": [4] * 4, "EnergyevKWh": [1] * 4,
            }).to_csv(root / "hospital_data_with_weekday_weekend.csv", index=False)
            pd.DataFrame([
                {"evseId": "A", "Arrival Time": "2024-01-01T00:00:00Z",
                 "Departure Time": "2024-01-01T00:15:00Z", "kWhDelivered": 4, "ev_type": "external"},
                {"evseId": "A", "Arrival Time": "2024-01-01T00:15:00Z",
                 "Departure Time": "2024-01-01T00:30:00Z", "kWhDelivered": 0.5, "ev_type": "internal"},
            ]).to_csv(root / "session_behaviour_clusters.csv", index=False)
            pd.DataFrame([{
                "evseId": "B", "chargingStart ISO-8601": "2024-01-01T00:00:00Z",
                "chargingStop ISO-8601": "2024-01-01T00:15:00Z",
                "energyConsumptionInKWh": 4, "ev_type": "external",
            }]).to_csv(root / "external.csv", index=False)
            rating_file = root / "ratings.csv"
            pd.DataFrame([{"evse_id": "A", "rated_power_kw": 20, "evidence": "charger specification"}]).to_csv(rating_file, index=False)
            results, _, groups, _, monthly, profiles = analyze(root, root / "rated", rating_file)
            self.assertEqual(results["alternative_power_kw"].tolist(), [16, 11, 11])
            self.assertAlmostEqual(results["alternative_shortfall_kwh"].sum(), 1.25)
            self.assertAlmostEqual(monthly.loc[0, "baseline_all_categories_kwh"], 6)
            self.assertAlmostEqual(monthly.loc[0, "alternative_all_categories_kwh"], 7.25)
            self.assertEqual(profiles.loc[0, "baseline_external_power_kw_assumed"], 22)
            self.assertEqual(profiles.loc[0, "alternative_external_power_kw_assumed"], 27)
            self.assertEqual(groups["sessions"].sum(), 2)
            unknown_results, _, _, _, _, unknown_profiles = analyze(root, root / "unknown")
            self.assertEqual(unknown_results["alternative_power_kw"].tolist(), [11, 11, 11])
            pd.testing.assert_series_equal(
                unknown_profiles["baseline_external_power_kw_assumed"],
                unknown_profiles["alternative_external_power_kw_assumed"], check_names=False,
            )

    def test_infeasible_band_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({
                "Timestamp": pd.date_range("2024-01-01", periods=20, freq="15min"),
                "hospitalpower_kw": [100] * 20, "powerhospitalalone": [96] * 20,
                "powerev_kw": [4] * 20, "EnergyevKWh": [1] * 20,
            }).to_csv(root / "hospital_data_with_weekday_weekend.csv", index=False)
            pd.DataFrame([
                {"evseId": str(i), "Arrival Time": f"2024-01-01T0{i}:00:00Z",
                 "Departure Time": f"2024-01-01T0{i+1}:00:00Z", "kWhDelivered": energy,
                 "ev_type": "external"}
                for i, energy in enumerate([11.01, 11.5, 22, 22.01])
            ]).to_csv(root / "session_behaviour_clusters.csv", index=False)
            pd.DataFrame(columns=[
                "evseId", "chargingStart ISO-8601", "chargingStop ISO-8601",
                "energyConsumptionInKWh", "ev_type",
            ]).to_csv(root / "external.csv", index=False)
            _, flagged, groups, _, _, _ = analyze(root, root / "analysis")
            self.assertEqual(groups.set_index("power_group")["sessions"].to_dict(), {
                "above_11_through_11_5_kw": 2,
                "above_11_5_through_22_kw": 1,
                "above_22_kw": 1,
            })
            self.assertEqual(flagged["required_average_kw"].tolist(), [11.01, 11.5, 22, 22.01])


if __name__ == "__main__":
    unittest.main()

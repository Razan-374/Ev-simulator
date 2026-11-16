import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from investigate_stage2_data import identifier_overlaps, investigate, overlapping_pairs
from prepare_stage2_data import load_sessions, prepare, reconstruct_session


class ReconstructionTests(unittest.TestCase):
    def test_partial_slots_and_cross_midnight_conserve_energy(self):
        start = pd.Timestamp("2024-01-01 23:55:00", tz="UTC")
        stop = pd.Timestamp("2024-01-02 01:00:00", tz="UTC")
        slots, delivered, shortfall = reconstruct_session(start, stop, 4.0)
        self.assertAlmostEqual(delivered, 4.0, places=8)
        self.assertAlmostEqual(shortfall, 0.0, places=8)
        self.assertAlmostEqual(slots[start.floor("15min")], 11 / 12, places=8)
        self.assertEqual(len(slots), 3)
        self.assertAlmostEqual(sum(slots.values()), 4.0, places=8)

    def test_infeasible_session_is_capped_and_shortfall_reported(self):
        start = pd.Timestamp("2024-01-01 23:55:00", tz="UTC")
        slots, delivered, shortfall = reconstruct_session(start, start + pd.Timedelta(minutes=10), 4.0)
        self.assertAlmostEqual(delivered, 11 / 6, places=8)
        self.assertAlmostEqual(shortfall, 4 - 11 / 6, places=8)
        self.assertEqual(len(slots), 2)

    def test_zero_energy(self):
        start = pd.Timestamp("2024-01-01", tz="UTC")
        slots, delivered, shortfall = reconstruct_session(start, start + pd.Timedelta(hours=1), 0)
        self.assertEqual((slots, delivered, shortfall), ({}, 0, 0))

    def test_conflicting_duplicates_and_invalid_rows_are_not_reconstructed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"evseId": "A", "Arrival Time": "2024-01-01T00:00:00Z",
                 "Departure Time": "2024-01-01T01:00:00Z", "kWhDelivered": 4},
                {"evseId": "B", "Arrival Time": "2024-01-01T00:00:00Z",
                 "Departure Time": "2024-01-01T01:00:00Z", "kWhDelivered": -1},
            ]
            pd.DataFrame(records).to_csv(root / "session_behaviour_clusters.csv", index=False)
            pd.DataFrame([{
                "evseId": "A", "chargingStart ISO-8601": records[0]["Arrival Time"],
                "chargingStop ISO-8601": records[0]["Departure Time"], "energyConsumptionInKWh": 5,
            }]).to_csv(root / "external.csv", index=False)
            self.assertEqual(
                load_sessions(root)["status"].tolist(),
                ["conflicting_duplicate", "invalid_input", "conflicting_duplicate"],
            )

    def test_preparation_reports_missing_duplicate_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hospital = pd.DataFrame({
                "Timestamp": ["2024-01-01 23:45", "2024-01-02 00:00", "2024-01-02 00:00", "2024-01-02 00:30"],
                "hospitalpower_kw": [100, 100, 101, 100],
                "powerev_kw": [4, 4, 5, 4],
                "powerhospitalalone": [96, 96, 96, 96],
                "EnergyevKWh": [1, 1, 1.25, 1],
            })
            hospital.to_csv(root / "hospital_data_with_weekday_weekend.csv", index=False)
            record = {
                "evseId": "A", "Arrival Time": "2024-01-01 23:55:00+00:00",
                "Departure Time": "2024-01-02 00:55:00+00:00", "kWhDelivered": 4,
                "user_id": 1, "ev_type": "internal",
            }
            pd.DataFrame([record]).to_csv(root / "session_behaviour_clusters.csv", index=False)
            pd.DataFrame([{
                "evseId": "A", "chargingStart ISO-8601": record["Arrival Time"],
                "chargingStop ISO-8601": record["Departure Time"], "energyConsumptionInKWh": 4,
                "card number": 1, "ev_type": "external",
            }]).to_csv(root / "external.csv", index=False)
            out = root / "out"
            metrics = prepare(root, out)
            load = pd.read_csv(out / "stage2_load_separation.csv")
            sessions = pd.read_csv(out / "stage2_session_reconstruction.csv")
            self.assertEqual(len(load), 4)
            self.assertEqual(load["interval_status"].tolist(), ["present", "ambiguous_duplicate", "missing", "present"])
            self.assertTrue(pd.isna(load.loc[1, "measured_hospital_load_kw"]))
            self.assertTrue(pd.isna(load.loc[2, "measured_hospital_load_kw"]))
            self.assertEqual(sessions["status"].tolist(), ["reconstructed", "duplicate_excluded"])
            self.assertAlmostEqual(metrics["reconstructed_energy_all_sessions_kwh"], 4, places=8)
            self.assertEqual(metrics["cross_midnight_sessions"], 1)
            self.assertEqual(load.loc[0, "estimated_building_only_load_kw"], 96)
            self.assertEqual(load.loc[0, "measurement_verification_status"], "unverified_definitions_and_time_alignment")
            with (out / "stage2_metadata.json").open(encoding="utf-8") as file:
                self.assertEqual(json.load(file)["measurement_verification_status"], "unverified")
            monthly, _, _, overlaps, _ = investigate(root, out)
            self.assertAlmostEqual(monthly["reconstructed_internal_kwh"].sum(), 11 / 12, places=8)
            self.assertEqual(monthly["reconstructed_external_kwh"].sum(), 0)
            self.assertEqual(monthly["recorded_ev_energy_unverified_kwh"].sum(), 2)
            self.assertEqual(monthly["excluded_intervals"].sum(), 2)
            self.assertTrue(overlaps.empty)

    def test_overlap_detection_excludes_touching_intervals(self):
        sessions = pd.DataFrame([
            {"evse_id": "A", "source": "a", "source_row": 2,
             "start": pd.Timestamp("2024-01-01T00:00:00Z"), "stop": pd.Timestamp("2024-01-01T01:00:00Z")},
            {"evse_id": "A", "source": "b", "source_row": 3,
             "start": pd.Timestamp("2024-01-01T00:30:00Z"), "stop": pd.Timestamp("2024-01-01T01:30:00Z")},
            {"evse_id": "A", "source": "b", "source_row": 4,
             "start": pd.Timestamp("2024-01-01T01:30:00Z"), "stop": pd.Timestamp("2024-01-01T02:00:00Z")},
        ])
        pairs = overlapping_pairs(sessions)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.iloc[0]["overlap_seconds"], 1800)
        sessions["identifier"] = ["same", "same", "same"]
        identifier_pairs = identifier_overlaps(sessions)
        self.assertEqual(len(identifier_pairs), 1)
        self.assertNotIn("identifier", identifier_pairs.columns)


if __name__ == "__main__":
    unittest.main()

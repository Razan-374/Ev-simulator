# ev-simulator

Electric vehicle load simulator with unmanaged/managed charging modes.

## Stage 2 load separation

Run `python prepare_stage2_data.py` to read the hospital load and both observed
session CSVs and write load, session, validation and metadata files to
`stage2_data/`. Original datasets and optimizer scripts are not modified.
Run `python -m unittest -v test_prepare_stage2_data` for focused checks.

The 11 kW session profiles are reconstructed, not measured. Infeasible
sessions are capped at their recorded stop and their undelivered energy is
reported. The hospital timezone, timestamp convention, meter coverage and EV
measurement definitions are unverified. Repeated hospital clock times are
left blank, and comparisons and building-only estimates are provisional;
consult the generated metadata and validation report before using the data.
After preparation, `python investigate_stage2_data.py` generates monthly
category/source comparisons, EVSE and overlap checks, infeasible-session
details and diagnostic clock-shift comparisons in `stage2_data/`. These are
exploratory checks, not verified meter coverage or corrected timestamps.
`python analyze_stage2_alternative.py` creates a separate conditional
external-charging comparison. It does not boost unknown outlets; see the
generated focused analysis report for the optional documented-ratings format.

Run `python prepare_shared_stage2_data.py` to prepare a single provisional
input set for both models under `stage2_data/shared_inputs/`. Fixed load is
the provisional building load plus reconstructed external charging once.
Shared-outlet occupancy and controllable availability remain unknown, not
zero. The files are not ready for optimizer integration; see their metadata
and validation report.
See `stage2_data/shared_inputs/STAGE2_OUTLET_MAPPING_ASSESSMENT.md` for the
existing mapping evidence, missing infrastructure information and proposed
unimplemented sensitivity bounds.

## CSV input
- Provide a 24-row CSV with columns `hour`, `base_load_kw`, and optional `solar_kw`, `parking_duration_hours`.
- Example:
```
hour,base_load_kw,solar_kw,parking_duration_hours
0,950,0,8
1,920,0,8
...
23,1100,0,8
```
- Run with the CSV overrides:
```
python "unmanaged&managed charging.py" --profile-csv path/to/inputs.csv
```

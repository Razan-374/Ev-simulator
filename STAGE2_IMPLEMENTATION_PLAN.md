# Stage 2 implementation plan

**Status:** Provisional shared data preparation has been authorized before
Stage 1 sign-off. Do not integrate it into either optimizer until Stage 1 is
completed and validated for both V1G and V2G and the input uncertainties are
resolved. Do not change the original datasets or overwrite existing results.

## 1. Complete Stage 1 first

- Finish and validate both models, including their existing objectives,
  constraints, solver outcomes and complete 15-minute power and SoC exports.
- Complete the requested 10-seed comparisons for the applicable 11 kW and
  22 kW configurations against the original scenarios and results. Check
  service feasibility, delivered energy, SoC bounds, discharge reserve and
  energy conservation; report infeasible runs and solver failures.
- Resolve any outstanding validation issues and confirm Stage 1 completion
  before optimizer integration. The
  existing verification documents include historical findings and are not, by
  themselves, confirmation that both models have passed this gate.

## 2. Verify the input data and definitions

- Identify authoritative hospital, internal-session and public-session
  sources, their common historical coverage, units, timestamp conventions and
  time zones. Check duplicates, gaps, ambiguous times and session
  classification. `session_behaviour_clusters.csv` contains both internal and
  external records; select internal sessions by verified classification.
- Verify from source documentation or the data owner that
  `powerhospitalalone` excludes all EV charging. Checking its numerical
  relationship to `hospitalpower_kw` and `powerev_kw` is useful but does not
  establish the measurement definitions. Do not use it as verified building
  load until this is resolved.
- Establish what `powerev_kw` measures, including which chargers it covers
  and whether timestamps refer to interval starts or ends. Verify whether
  recorded session energy is charger-side energy and whether recorded stop
  times represent charging end, departure, or both.
- Document an authoritative EVSE-to-outlet mapping. The intended arrangement
  is 22 physical outlets: 14 internal-only, 3 shared, and 5 public-only; the
  three shared outlets are included in the 17 potentially controllable
  outlets. Confirm this mapping and reconcile it with the existing Stage 1
  assumptions before using it. Do not infer it from outlet counts alone.

## 3. Standalone preparation component

After the Stage 1 gate, implement `prepare_stage2_data.py` independently of
the optimizers. It should read the verified sources without altering them and
write prepared outputs to a separate Stage 2 directory:

| Output | Planned contents |
| --- | --- |
| `stage2_fixed_load.csv` | Continuous 15-minute timestamps, verified hospital-only load, reconstructed external load, and their sum in kW. |
| `stage2_internal_sessions.csv` | The same observed internal sessions for both models, with stable session IDs, arrival, departure, required energy and other verified model inputs. |
| `stage2_outlet_availability.csv` | Per-slot public occupancy of shared outlets and remaining controllable outlet capacity, plus the mapped public-only occupancy needed for checks. |
| `stage2_validation_report.csv` | Data-quality findings, reconstruction energy checks, occupancy issues and measured-versus-reconstructed comparisons. |
| Metadata | Source files, coverage, units, time zone, interval conventions, EVSE mapping, assumptions and unresolved issues. |

### External charging reconstruction

- For each public session, assume unmanaged charging begins at its recorded
  start at 11 kW and continues until its recorded energy is delivered, then
  becomes zero. The profile is **assumed, not measured**.
- Calculate each 15-minute interval's average power from the actual overlap
  with the charging period. This handles starts within a slot and partial
  final slots while preserving recorded session energy.
- Reserve the assigned outlet for the full recorded parking period, including
  time after charging ends, when actual departure is available. If departure
  is missing or its meaning is uncertain, flag it; do not silently treat
  charging completion as departure. Document any agreed fallback before use.
- Flag sessions whose energy cannot be delivered at 11 kW within the known
  connection period, as well as invalid times, unknown mappings, overlapping
  sessions on an outlet and occupancy exceeding physical capacity. Do not
  silently clip energy or invent capacity. Resolve these issues or explicitly
  exclude affected periods before using results.
- Shared public occupancy reduces the 17 controllable outlets; public-only
  occupancy does not. Define and document how partial-slot occupancy is
  represented conservatively so the models cannot double-allocate an outlet.

## 4. Validate preparation on a small multi-day period

- Check the continuous time grid, alignment, time zones, boundary-crossing
  sessions, completeness and all data-quality flags. Reconstructed
  per-session and aggregate energy must match the included recorded energy
  within a documented numerical tolerance.
- Verify slot by slot that fixed load equals hospital-only load plus external
  charging, and that simulated total load equals fixed load plus the
  optimizer's net internal EV power. Never add `hospitalpower_kw` or
  `powerev_kw` on top of these components.
- Compare reconstructed EV power with `powerev_kw` only after aligning their
  time intervals and coverage. If the measured series includes internal
  charging, an external-only reconstruction is not a like-for-like total:
  use a separately identified comparable internal reference if available,
  or report the comparison as partial. Do not add measured EV power to the
  fixed load. Investigate substantial differences in definitions, timing,
  coverage and assumptions before drawing economic conclusions.
- Test the preparation independently, including off-grid timestamps, partial
  intervals, zero energy, infeasible sessions, shared-outlet occupancy and
  cross-midnight sessions.

## 5. Integrate and compare models

- Once preparation has passed the multi-day checks, adapt V1G and V2G to read
  exactly the same prepared files and observed internal sessions. Keep V1G's
  service-first and V2G's peak-minimization objectives unchanged. Internal
  charging and V2G discharge remain optimizer decisions, not fixed 11 kW
  profiles.
- Support continuous multi-day operation: carry active sessions, outlet
  occupancy, SoC and relevant remaining energy across midnight rather than
  resetting them. Define boundary conditions for sessions already active at
  the beginning or still active at the end of the selected period.
- Validate the multi-day run before extending to the full verified historical
  overlap. Compare V1G and V2G with identical inputs and independently
  calculated maximum demand for each billing period and each strategy;
  document billing-period boundaries and any partial periods.
- Preserve original scenarios and results. Do not introduce BESS or new
  stochastic scenarios in this stage.

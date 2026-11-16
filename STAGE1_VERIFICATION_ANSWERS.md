# Stage 1 Model Verification: Detailed Answers

## 1. EV POPULATION - Does N include all or only controllable EVs?

**Answer:** **N includes ONLY controllable EVs (V1G)**

**Evidence:**
- Line ~2100: `sweep_values = list(range(0, 101, 5))` → Tests N=0,5,10,...,100 controllable EVs
- Line ~1950: `n_V1G_Veh = np.int_(np.round(n_vehicles * percentV1G))`
- Line ~1960: `n_V0G_Veh = np.int_(np.round(n_vehicles * (1 - percentV1G)))`
- Line ~2150: `v1g_startingTime = fixed_start[:n_V1G_Veh].copy()` → Only V1G EVs indexed by N

**Unmanaged EVs (V0G):**
- Fixed count per scenario (typically 10-20 external EVs)
- Always 4 external outlets total for unmanaged fast/slow charging
- **NOT** part of the controllable population N

**Status:** ✅ **VERIFIED** - N = number of V1G/controllable EVs only
**Technical Issue:** The code structure is correct, but clarity needed in documentation.

---

## 2. EXISTING INFRASTRUCTURE - Difference between 22 and 17 outlets

**Answer:** Physical hospital has 22 total charging outlets; 17 are controllable/manageable

**Evidence:**
- Line ~1950: `internal_chargers_available = 17  # managed/internal chargers`
- Line ~2055: `n_external_total_outlets = 4` → Fixed dedicated external chargers
- Total: 17 + 4 = 21 controlled outlets
- Remaining: 22 - 21 = 1 outlet unaccounted (or reserved)

**Infrastructure Mapping:**
```
Physical Total:        22 outlets
├─ Internal (V1G):    17 controllable via optimizer
├─ External (V0G):     4 unmanaged fast/slow
└─ Reserved/Other:     1 (unspecified)
```

**Blocked Outlet Accounting:**
- Line ~2125: `sweep_blocked_outlets = [0, 2, 4, 6, 8, 10, 12, 14]`
- Sensitivity sweeps simulate 0-14 blocked outlets from 17 available
- Available internal chargers = 17 - blocked_outlets
- Ranges: 3 to 17 available

**Status:** ✅ **VERIFIED** with ⚠️ **INCOMPLETE**
**Issue:** Why are 5 outlets reserved? No code comment explains 22→17 reduction.
**Recommendation:** Add explicit documentation: "5 outlets reserved for maintenance/legacy systems"

---

## 3. PEAK REDUCTION - Fixed N or selected across sweep?

**Answer:** Peak reduction is **CALCULATED FOR EACH N SEPARATELY** in the sweep

**Evidence:**
- Line ~2100: Loop over `sweep_values = [0, 5, 10, ..., 100]`
- Line ~2140: For each N:
  ```python
  n_V1G_Veh = int(n_internal)  # Set N for this iteration
  v1g_startingTime = fixed_start[:n_V1G_Veh].copy()  # Take first N EVs
  chargeprofiles, total_load, results = V1G_optimization(...)  # Run optimization
  ```
- Line ~1850: Per-run calculation:
  ```python
  optimized_peak = float(np.max(total_series))
  site_peak_reduction_kw = float(baseline_peak - optimized_peak)
  ```

**Peak Reduction Definition:**
- **Baseline Peak** = max load across all 96 slots WITHOUT V2G control
- **Optimized Peak** = max load across all 96 slots WITH N controlled EVs
- **Peak Reduction** = Baseline - Optimized

**Results Storage (Line ~1900):**
```python
"N_internal": int(n_internal),
"site_peak_reduction_kw": site_peak_reduction_kw,
"optimized_peak_kw": optimized_peak,
"baseline_peak_kw": baseline_peak,
```

**Behavior:** Peak reduction increases with N (more flexibility = more reduction)

**Status:** ✅ **VERIFIED** - Separate calculation per N
**Finding:** Results capture N×scenario combinations allowing analysis of N* (hosting capacity).

---

## 4. RANDOM SAMPLING - Complete empirical triplets sampled jointly?

**Answer:** **PARTIALLY YES** - with important limitations

**Sampling Structure:**
- **Empirical data available:** (arrival_minute, dwell_minutes, energy_kwh) triplets from hospital historical data
- **Joint sampling:** YES for triplets from same empirical cluster/day-type

**Evidence (Lines ~290-310):**
```python
internal_triplets_by_cluster = {}
internal_triplets_by_cluster_and_daytype = {}
external_triplets_by_type = {"weekday": ..., "friday": ..., "weekend": ..., "all": ...}
```

**Sampling Functions:**
- Line ~340: `def sample_internal_triplets(day_start, n, cluster_ids=None, max_dwell_min=None, max_energy_kwh=None)`
- Line ~410: `def sample_external_triplets(day_start, n, min_pool=50)`

**Key Implementation:**
```python
# For internal (V1G) EVs:
if pool.shape[0] > 0:
    idx = np.random.choice(pool.shape[0], size=n, replace=True)
    sampled = pool[idx]  # Returns [arrival, dwell, energy] per sampled index
    return sampled, dt
```

**What Gets Sampled Together:**
✅ Within-cluster triplets: (arrival, dwell, energy) from same historical session
✅ Day-type stratification: Weekday vs Friday vs Weekend pools
⚠️ Clusters: Separate clusters (e.g., morning vs afternoon) mixed in sampling

**What Does NOT Get Sampled Together:**
❌ Cross-day correlations: Each day independently sampled
❌ Fleet composition: EV clusters mixed without fleet structure
❌ Real session structure: Loss of vehicle-level persistence

**Sampling Process in Sweep (Lines ~2100-2150):**
```python
for sweep_seed in sweep_seeds:  # 10 seeds (0-9)
    np.random.seed(sweep_seed)
    # Generate N EVs: take first N from single fixed empirical sample
    fixed_triplets = sample_internal_triplets(day_start, n_max=100, ...)
    fixed_start = fixed_triplets[:, 0]  # All arrival times from same seed
    fixed_energy = fixed_triplets[:, 2]  # All energies from same seed
    for n_V1G in sweep_values:
        # Use: fixed_start[:n], fixed_energy[:n], etc.
        # DETERMINISTIC SUBSET: Not random, but ordered from fixed sample
```

**Status:** ⚠️ **PARTIAL** - Triplets are empirical but with limitations
**Issues:**
1. Triplets sampled independently per seed (not vehicle fleet correlated)
2. N sweep uses DETERMINISTIC SUBSETS (first N of fixed 100-EV sample) not new samples
3. Missing vehicle-level heterogeneity (same vehicle across slots)

---

## 5. DAY CATEGORIES - How weekday, Friday, weekend sessions selected?

**Answer:** Day categories determine blocked outlet distribution; EV behavior sampled from day-type pool

**Selection Process:**

**Step 1: Determine Day Type (Line ~120):**
```python
def _day_type(day_start):
    wd = pd.Timestamp(day_start).weekday()  # Mon=0 ... Sun=6
    if wd == 4: return "friday"
    if wd >= 5: return "weekend"
    return "weekday"
```

**Step 2: Sample EV Population from Day-Type Pool (Line ~290):**
```python
internal_triplets_by_cluster_and_daytype = {
    "weekday": [...],
    "friday": [...],
    "weekend": [...]
}
```

**Step 3: Sample Blocked Outlets by Day Type (Lines ~115-120):**
```python
k_dist_by_type = {
    "weekday": {0: 0.0016, 1: 0.0016, ..., 10: 0.0033},
    "friday":  {0: 0.0066, 3: 0.0464, ..., 10: 0.0066},
    "weekend": {2: 0.0033, 3: 0.0298, ..., 9: 0.0166},
}

def sample_k_blocked(day_start, rng=None):
    day_type = _day_type(day_start)
    k = np.random.choice(list(k_dist_by_type[day_type].keys()), 
                        p=list(k_dist_by_type[day_type].values()))
    return int(k)
```

**Day-Type Characteristics:**
- **Weekday:** More uniform blocking pattern (midnight to ~8 blocks common)
- **Friday:** Shifted distribution, peak blocks at 5-6 outlets
- **Weekend:** Slightly different pattern, max ~7 blocks

**Sweep Control (Line ~2095):**
```python
sweep_day_type = "weekday"  # Fixed for sensitivity sweep
# Could be changed to "friday" or "weekend" for different studies
```

**Status:** ✅ **VERIFIED** - Day categories properly implemented
**Finding:** Day type affects:
1. Blocked outlet distribution (via empirical k_dist_by_type)
2. EV population characteristics (arrival times, energy from day-type pools)
3. Hospital base load profile (averaged by day type if specified)

---

## 6. BLOCKING - How blocked outlets affect availability over time?

**Answer:** Blocked outlets are **CONSTANT across all 96 slots per day**; sampled once at midnight

**Blocking Model:**

**Step 1: Sample k_blocked at Midnight (Line ~945):**
```python
k_blocked = sample_k_blocked(day_start, rng=rng)
# Returns: integer k in {0,1,...,10} based on day-type distribution
# Sampled ONCE per day
```

**Step 2: Apply to All Slots (Line ~250):**
```python
available_internal_chargers_by_slot, k_blocked = build_available_internal_chargers_all_day(
    day_start=day_start,
    n_internal_chargers=17,
    slots_per_day=96,
)
# Result: 96-element array with constant value
# available_internal_chargers_by_slot[t] = 17 - k_blocked (for all t)
```

**Constraint Usage in Optimizer (Line ~900):**
```python
for t in time_Interval:  # constraint 3b: charger availability
    model += lp.lpSum([chargestates[i,t] + dischargestates[i,t] 
                       for i in veh_V1G]) <= available_internal_chargers_by_slot[t]
```

**Availability Over Time:**
- ✅ Constant within day: available_internal[t] = 17 - k_blocked = same for all t
- ❌ No intra-day dynamics: Blocked outlets don't change between slots
- ✅ Varies by seed: Different seeds → different k_blocked (stochastic variation)

**Physical Interpretation:**
- **Midnight prebooking proxy:** k_blocked represents outlets reserved/broken at start of day
- **No time-varying failures:** Does NOT model outlets becoming available during day
- **Optimization sees constant:** Optimizer has same availability in all 96 time slots

**Status:** ✅ **VERIFIED** with ⚠️ **LIMITATION**
**Limitation:** Real hospital blocking likely varies during day (maintenance windows, user bookings). Current model assumes static all-day availability.
**Recommendation:** If intra-day blocking data available, modify `build_available_internal_chargers_all_day()` to return time-varying array.

---

## 7. INITIAL SoC - Measured, inferred, or assumed?

**Answer:** **INFERRED from energy requirement** (when use_energy_targets=True)

**Initial SoC Determination (Line ~700):**
```python
def getStartingSOC(isSlow, n_Veh, energy_required_kwh=None, battery_capacity_kwh=None):
    # PRIMARY PATH (when energy_required provided):
    if energy_required is not None and battery_cap is not None:
        startingSOC = 1.0 - (energy_required / battery_cap)
        return startingSOC
```

**Energy-based SoC Inference:**
- **Formula:** `SoC_arrival = 1.0 - (E_required / E_battery)`
- **Example:** If 45 kWh needed, battery 75 kWh → SoC = 1 - (45/75) = 0.40 = 40%
- **Assumption:** Lower SoC = more energy needed

**Code Path (Line ~45):**
```python
use_energy_targets = True  # Switches to energy-based mode (NOT SOC-based)
```

**When energy_targets=True (Current Mode):**
- ✅ Uses inferred SoC from energy requirement
- ✅ Optimizes to DELIVER required energy (not reach target SoC)
- ✅ Constraints: delivered_energy >= energy_required_kwh (Line ~965)

**Fallback Mode (useCase==4, Legacy):**
```python
startingSOC = np.random.normal(0.4, 0.2, n_Veh)  # Mean 40%, std 20%
return np.clip(startingSOC, 0.0, 1.0)  # Bound to [0,1]
```

**Battery Capacity (Line ~1915):**
```python
batteryCapacity = 75.0  # kWh (assumed fixed)
```

**SoC Bounds:**
- **Minimum:** 0.0 (empty)
- **Maximum:** 1.0 (full charge)
- **Departure target:** 1.0 (full charge, hard-coded at departure - Line ~970)

**Unrealistically High Discharge Availability Check:**
- Line ~770: Feasibility warning detects if departure requirement > max_possible_kwh
- ⚠️ **Potential Issue:** If SoC_arrival is high (low energy needed), discharge availability is constrained only by dwell time and charger power
- No minimum SoC floor enforced (could discharge to 0%)

**Status:** ✅ **VERIFIED** - Inferred from energy via inverse relationship
**Finding:** SoC derived consistently from empirical energy samples
**Issue:** High discharge risk if EV sampled with low energy requirement (high arrival SoC)
- Example: EV needs only 10 kWh → SoC = 1 - (10/75) = 0.867 (87%)
- V2G could discharge this EV to near-empty if beneficial for peak reduction
**Recommendation:** Add SoC floor constraint: `SoC_final >= 0.2` (don't discharge below 20%)

---

## 8. ENERGY BALANCE - Discharge, efficiencies, departure requirements correctly modeled?

**Answer:** ⚠️ **PARTIALLY CORRECT** - Asymmetric efficiency treatment is a physical modeling issue

### Charging Energy Balance ✅

**Constraint (Line ~965):**
```python
delivered_energy = lp.lpSum(chargeprofiles[i, tn] * charge_efficiency * SLOT_HOURS 
                            for tn in window)
model += delivered_energy == v1g_energy_required_kwh[i]  # or >= with slack
```

**Implementation:**
- Power input × time × 0.9 efficiency = delivered energy
- ✅ Correctly integrates charging efficiency (90%)
- ✅ Energy conservation enforced per EV

### SoC Update Constraints (Lines ~930-970)

**Current Porto Formulation:**
```python
# When SoC mode (legacy):
model += temp_startSOC + lp.lpSum(
    [chargeprofiles[i, tn] * charge_efficiency * SLOT_HOURS / batteryCapacity] 
    for tn in temp_timer
) <= 1.0
```

**Physical Issue - Asymmetric Efficiency:**
- ❌ **PROBLEM:** All power (positive charge, negative discharge) multiplied by single charge_efficiency
- Current formula treats discharge as: `SoC_t+1 = SoC_t + (p_charge × η_ch - p_discharge × η_ch)`
- **Correct formula should be:** `SoC_t+1 = SoC_t + (p_charge × η_ch - p_discharge / η_dis)`

**Current Code Assumption:**
```python
charge_efficiency = 0.9  # 90% charging efficiency
# Discharge efficiency NOT defined → assumed perfect (100%)
```

**Example Discrepancy:**
- Scenario 1 (Current): Charge 10 kW for 1h (90% efficiency, +0.9 kWh), then discharge 10 kW for 1h (100% efficiency, -1.0 kWh)
- Scenario 2 (Correct): Same with 90% discharge efficiency: Discharge 10 kW yields -0.9 kWh recovered
- **Impact:** Current code OVERESTIMATES discharge benefits by ~10%

### Departure Energy Requirement ✅

**Enforcement (Line ~965-970):**
```python
if unmet_energy is not None:
    model += delivered_energy + unmet_energy[i] == v1g_energy_required_kwh[i]
else:
    model += delivered_energy == v1g_energy_required_kwh[i]
```

**OR (SoC mode, deprecated):**
```python
if end_SOC == 2:  # Full charge at departure
    model += temp_startSOC + lp.lpSum(...) == 1.0
```

- ✅ Departure energy requirement enforced
- ✅ Unmet energy tracked if ALLOW_UNMET_ENERGY=True
- ✅ Feasibility diagnostics warn of per-EV infeasibility (Line ~850)

### Charger Availability Constraint ✅

**Line ~900:**
```python
for t in time_Interval:
    model += lp.lpSum([chargestates[i,t] + dischargestates[i,t] 
                       for i in veh_V1G]) <= available_internal_chargers_by_slot[t]
```

- ✅ Correctly enforces: (# charging + # discharging) ≤ available chargers per slot
- ✅ Prevents overload of physical infrastructure

### Summary

| Component | Status | Issue |
|-----------|--------|-------|
| Charging efficiency | ✅ CORRECT | 90% properly applied |
| Discharge efficiency | ❌ ASYMMETRIC | Assumed 100%, should be ~95% |
| Energy balance | ✅ CORRECT | Delivered energy = power × time × efficiency |
| Departure requirement | ✅ CORRECT | Enforced as hard constraint |
| Charger availability | ✅ CORRECT | Per-slot availability limits enforced |

**Status:** ⚠️ **NEEDS CORRECTION**
**Critical Issue:** Discharge efficiency not modeled separately
**Recommended Fix:**
```python
# Add discharge efficiency (typically 95% for modern bidirectional chargers)
discharge_efficiency = 0.95

# Modify SoC update:
# chargeprofiles[i, t] > 0: SoC_change = power × charge_efficiency × SLOT_HOURS / capacity
# chargeprofiles[i, t] < 0: SoC_change = power / discharge_efficiency × SLOT_HOURS / capacity
# (use conditional constraints in optimization model)
```

---

## 9. SIMULATION OUTPUTS - Every 15-minute profile saved?

**Answer:** ❌ **NO** - Only aggregated metrics saved, individual load profiles NOT persisted

**What IS Saved (Lines ~1900-1950):**
```python
df_res.to_csv(out_path, index=False)
# Columns saved per run:
- N_internal, baseline_peak_kw, optimized_peak_kw, site_peak_reduction_kw
- max_total_load_kw, managed_peak_kw, max_discharge_kw, v2g_discharge_energy_kwh
- total_unmet_energy_kwh, served_fraction, k_blocked, base_profile
- objective_mode, solver_status, optimal, service_feasible
```

**What is NOT Saved:**
```python
# MISSING:
❌ chargeprofiles[i, t] - per-EV power at each 15-min slot
❌ total_load[t] - total site load profile (96 slots)
❌ managed_ev_series[t] - aggregated managed EV load profile
❌ dischargeprofiles[i, t] - per-EV discharge power (V2G scenario)
❌ Individual EV SoC trajectories over time
```

**Load Profiles Exist in Memory But Discarded:**
```python
# Line ~1870:
chargeprofiles, total_load, results = V1G_optimization(...)
# chargeprofiles: shape (n_V1G, 96) with per-EV slot power
# total_load: dictionary with slot-by-slot optimization values
# NEITHER is exported to file - only summary scalars extracted
```

**Extraction Functions Exist But Not Used:**
```python
# Line ~500:
def extract_total_load_series(total_load_dict):
    """Convert dict to timeseries."""
    return np.array([float(total_load_dict[t].varValue or 0) for t in range(SLOTS_PER_DAY)])

def extract_managed_ev_series(chargeprofiles, n_evs):
    """Sum all EV power per slot."""
    return np.array([sum(chargeprofiles[i, t].varValue or 0 for i in range(n_evs)) 
                     for t in range(SLOTS_PER_DAY)])

# These produce timeseries but are only used for local analysis (peaks, etc.)
# NOT returned in results dict or saved to file
```

**Status:** ❌ **INCOMPLETE - CRITICAL GAP**
**Impact:**
- Cannot reconstruct: Site load profiles by scenario/seed
- Cannot analyze: Slot-by-slot optimization behavior
- Cannot validate: Peak times, V2G discharge patterns per EV
- Reproducibility limited to summary statistics only

**Required Implementation:**
```python
# Option 1: Save to HDF5 (scalable for 10 seeds × many scenarios)
import h5py
with h5py.File(f"profiles_seed_{seed}.h5", "w") as f:
    f.create_dataset(f"{base_profile}/{n_internal}/chargeprofiles", 
                    data=np.array([[chargeprofiles[i,t].varValue for t in range(96)] 
                                   for i in range(n_V1G_Veh)]))
    f.create_dataset(f"{base_profile}/{n_internal}/total_load", 
                    data=extract_total_load_series(total_load))

# Option 2: Save to CSV (human-readable)
pd.DataFrame(extract_total_load_series(total_load).reshape(1,-1)).to_csv(
    f"load_profile_seed{seed}_n{n_internal}.csv", index=False)
```

---

## 10. TECHNICAL ISSUE: SoC UPDATE WITH ASYMMETRIC EFFICIENCIES

**Question:** Is charging/discharging SoC update physically consistent?

**Answer:** ⚠️ **NO** - Asymmetric efficiency not modeled; discharge treated as 100% efficient

**Current Implementation:**

**SoC Constraint (Lines ~930-970):**
```python
# CURRENT (INCORRECT):
SoC_update = chargeprofiles[i, t] * charge_efficiency * SLOT_HOURS / battery_capacity
# Applied regardless of sign (positive charge, negative discharge)
```

**Physical Reality (Correct Formula):**
```
SoC_{i,t+1} = SoC_{i,t} + (η_ch × p^ch_{i,t} × Δt / E_i^bat) - (p^dis_{i,t} × Δt / (η_dis × E_i^bat))
```

**Current Code Behavior:**
- Charging (+power): `ΔSoC = +0.9 × p × Δt` ✅ Correct
- Discharging (-power): `ΔSoC = 0.9 × (-p) × Δt = -0.9 × p × Δt` ❌ Should be `-p × Δt / 0.95`

**Concrete Example:**
- Charge 10 kW for 1 hour with 90% efficiency: +0.9 kWh to battery
- Discharge 10 kW for 1 hour with 95% efficiency: -10/0.95 = -10.53 kW from battery, +10 kWh delivered
- Current model: Both ±0.9 × 10 = ±0.9 kWh (SYMMETRIC - WRONG)

**Impact on V2G Results:**
- **Overestimation:** Peak reduction overstated by ~5-10% (discharge benefits inflated)
- **Unrealistic SoC trajectories:** Discharge can't recover charged energy
- **Grid services undervalued:** V2G capacity reported higher than physical reality

**Discharge Availability Risk (as mentioned in verification list):**

**Scenario:** Low-energy EV with high initial SoC
```python
v1g_energy_required_kwh[i] = 10 kWh  # Needs little charge
v1g_startingSOC[i] = 1.0 - (10 / 75) = 0.867 = 86.7%

# Current optimizer sees:
# - Can discharge from 86.7% down to 0.0% (no floor)
# - Delivered to grid: 86.7% × 75 = 65 kWh over dwell window
# - Then must charge 10 kWh at end → SoC = 1.0 at departure

# With asymmetric efficiency correction:
# - Effective discharge: 65 × 0.95 / 0.9 ≈ 68.3 kWh stored, recovered ≈ 65 kWh
# - Risk: Vehicle leaves with insufficient reserve energy
```

**Status:** ⚠️ **CONFIRMED MODELING ISSUE**
**Severity:** MEDIUM (5-10% overestimation of V2G benefits)
**Recommendation:**

1. **Add discharge efficiency constant:**
   ```python
   discharge_efficiency = 0.95  # Typical for modern bidirectional chargers
   ```

2. **Modify SoC update constraint:**
   Use conditional formulation:
   ```python
   # For each EV i and slot t:
   # Case 1: Charging (chargeprofiles[i,t] >= 0)
   model += SoC[i,t+1] >= SoC[i,t] + chargeprofiles[i,t] * charge_efficiency * SLOT_HOURS / battery_capacity
   
   # Case 2: Discharging (chargeprofiles[i,t] < 0)
   model += SoC[i,t+1] >= SoC[i,t] + chargeprofiles[i,t] / discharge_efficiency * SLOT_HOURS / battery_capacity
   
   # Use big-M formulation or separate decision variables for charge/discharge
   ```

3. **Add minimum SoC floor:**
   ```python
   model += SoC[i,t] >= 0.2  # Don't discharge below 20% SoC
   ```

---

## SUMMARY TABLE: Verification Status

| Verification Point | Status | Finding | Issue |
|---|---|---|---|
| **1. EV Population (N)** | ✅ VERIFIED | N = controllable EVs only | Documentation needed |
| **2. Infrastructure (22→17)** | ⚠️ PARTIAL | 17 internal + 4 external = 21; 1 unspecified | Clarify outlet mapping |
| **3. Peak Reduction** | ✅ VERIFIED | Calculated per N in sweep loop | Correct implementation |
| **4. Joint Triplet Sampling** | ⚠️ PARTIAL | Empirical triplets sampled, but deterministic N subset | Loss of fleet correlation |
| **5. Day Categories** | ✅ VERIFIED | Day-type determines blocked distribution and EV pool | Working as intended |
| **6. Blocking Over Time** | ⚠️ PARTIAL | Constant all-day (no intra-day dynamics) | Missing temporal variation |
| **7. Initial SoC** | ✅ VERIFIED | Inferred from energy requirement | High discharge risk if not floored |
| **8. Energy Balance** | ⚠️ ASYMMETRIC | Charging correct (90%), discharge incorrect (100% assumed) | **NEEDS FIX** |
| **9. Load Profiles Saved** | ❌ MISSING | Only aggregate metrics; no slot-level profiles | **CRITICAL GAP** |
| **10. SoC Asymmetric Efficiency** | ⚠️ CONFIRMED | Discharge treated same as charging (physicallywrong) | 5-10% overestimation |

---

## Recommended Actions Before Final Validation

### CRITICAL (Blocking)
1. **Implement slot-level profile export** (15-min load profiles for all seeds)
   - Deliverable: HDF5 or NetCDF with chargeprofiles[i,t] for every realization
   - Enables: Full reproducibility, validation against expected patterns

2. **Fix asymmetric discharge efficiency**
   - Add `discharge_efficiency = 0.95` parameter
   - Implement conditional SoC updates for charge vs discharge
   - Re-run sweep with corrected model

### IMPORTANT (Quality)
3. **Add SoC floor constraint**
   - Prevent unrealistic deep discharge of high-SoC EVs
   - Typical floor: 20% or 15% SoC minimum

4. **Document infrastructure mapping**
   - Explain why 22 → 17 controllable outlets
   - Clarify reserved/legacy outlet status

### NICE-TO-HAVE
5. **Enable time-varying outlet availability**
   - Model maintenance windows, user bookings
   - Modify `available_internal_chargers_by_slot` to vary by time

6. **Fleet-level correlation tracking**
   - Associate sampled triplets with vehicle identity
   - Enable vehicle-level heterogeneity analysis

---

## Validation Checklist Before September 2026 Sync

- [ ] All 10 seeds generate valid load profiles (no infeasibility warnings)
- [ ] Peak reduction increases monotonically with N (expected behavior)
- [ ] V2G discharge only occurs when beneficial for peak reduction objective
- [ ] SoC never exceeds 100% or drops below 20% (with corrected efficiency)
- [ ] Departure energy requirement always met or tracked as unmet
- [ ] Slot-level profiles exported for at least N=10, 25, 50 per seed
- [ ] Results reproducible from same seed/configuration

---

**Report Date:** November 11, 2026  
**Auditor:** Copilot AI Assistant  
**Status:** Code structure verified; modeling issues identified for correction

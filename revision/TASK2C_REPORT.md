# Task 2C — minimal USR power refit

**Outcome:** a seven-knot, nonnegative piecewise-linear USR power candidate is
supported for an explicitly labeled, small diagnostic controller pilot. It is
**not promoted to the default twin**. This establishes a usable correction to
negative power, not the validity of the submitted PPO comparisons. No RL training
or traffic final-test outcome evaluation was performed.

## Provenance and grouped protocol

The unchanged recovered CSV has 4,564 measurements, five complete sweeps, 110 case
blocks and 22 offered-load levels (0.01–5,000 Mbps), with 1,250-byte packets.
Its SHA-256 remains
`72e8772f555babfec66b695fe0b31a7366ebc13b3b125413f8d7823b000f5d20`.

New evidence improves the Task 2B provenance assessment: **all 4,564 power labels
match Scaphandre process-power exports divided by 10⁶**, within 2.258 seconds.
The process is PID 525735, `/openair-upf/bin/oai_upf`; the raw metric is explicitly
microwatts. Thus the target is **process-attributed watts**, with no idle subtraction
introduced by this refit. All case/sweep, packet-size/rate and load identifiers
match the recovered campaign catalog. Packet size × packet rate × 8 / 10⁶ matches
the recorded Mbps exactly. Paths, hashes and row matches are in
[measurement provenance](reports/task2c/measurement_provenance.json) and
[raw-label matches](reports/task2c/power_label_raw_matches.csv).

Uncertainties remain: 31 labels match a nearby value rather than the nearest
timestamp's value; zeros sometimes match several samples. The exact historical
join/trimming procedure is unauthenticated. The campaign labels the input as a
DL/UL objective (per direction), not a verified aggregate DL+UL measurement. We
retain the historical numeric load mapping without silently doubling it. The
aggregated exports questioned in Task 2B were not substituted. There is no true
idle measurement, calibrated meter uncertainty, packet-size diversity or new
physical campaign.

All groups are on **13 January 2025**:

| Role | Complete sweeps | Rows / case blocks | Time interval |
|---|---|---:|---|
| Training | 1, 2, 3 | 2,736 / 66 | 13:09:06–17:50:51 |
| Validation | 4 | 915 / 22 | 18:21:03–19:40:06 |
| Internal test | 5 | 913 / 22 | 19:44:48–21:03:42 |

Each role contains all 22 load levels. This preserves whole correlated measurement
blocks while allowing load interpolation checks. The internal test is from the
same day/device, and these data were already inspected historically and in Task
2B; it is not a new independent or fully blind physical test.

## Candidates and frozen selection

[The configuration](configs/usr_power_refit_v1.json) fixes three candidates:
nonnegative affine regression, seven-knot piecewise-linear regression, and a
22-knot model using training load levels. All minimize unweighted training-row
squared error in **watts** using nonnegative least squares. There is no target
standardization or output clipping. The affine input scale is the training maximum;
PWL uses original Mbps coordinates.

Selection used **profiling-validation RMSE only**, before internal-test scoring,
original-model comparison or traffic diagnostics. The seven-knot model was frozen
and was not refitted on training plus validation. No neural comparator was trained.

| Model | Validation RMSE W | Internal-test RMSE W | Validation RMSE W, ≤20 Mbps | Internal-test RMSE W, ≤20 Mbps |
|---|---:|---:|---:|---:|
| Nonnegative affine | 1.078557 | 1.033190 | 0.782680 | 0.783225 |
| **Seven knots — selected** | **0.266462** | **0.309299** | **0.009086** | **0.010058** |
| 22 knots | 0.270919 | 0.295320 | 0.007626 | 0.008774 |
| Original MLP — descriptive only | 0.264858 | 0.302618 | 0.044541 | 0.045022 |

The 22-knot candidate performs better on internal test and at low loads; this does
not override the prespecified validation selection. The original model's historic
training split included rows from all five sweeps, so its numbers are **not an
unbiased held-out comparison**. Overall errors remain similar, while low-load
errors improve substantially. No claim of global accuracy superiority is warranted.

Selected-model validation/internal-test MAE is 0.079240/0.105602 W, R² is
0.969316/0.956931, and bias is +0.009571/+0.039595 W. Across sweeps 1–5 its RMSEs
are 0.289613, 0.289228, 0.292169, 0.266462 and 0.309299 W. Full metrics and residual
breakdowns are saved [by load](reports/task2c/errors_by_load.csv),
[by sweep](reports/task2c/errors_by_sweep.csv) and
[by source row](reports/task2c/profiling_predictions.csv).

## Low-load behavior, physical constraints and uncertainty

All values below are watts per USR instance; measured means pool the five sweeps
for description, not fitting or selection.

| Mbps | Measured mean | Original MLP | Replacement |
|---:|---:|---:|---:|
| 0.01 | 0.000195 | 0.060257 | 0.000147 |
| 0.1 | 0.001156 | 0.052894 | 0.001234 |
| 1 | 0.009007 | 0.005958 | 0.012106 |
| 2 | 0.017470 | −0.021882 | 0.019097 |
| 4 | 0.034704 | −0.053532 | 0.033080 |
| 10 | 0.085635 | 0.102490 | 0.075027 |
| 20 | 0.172508 | 0.207182 | 0.168777 |

The declared **interpolation** domain is 0.01–5,000 Mbps per instance, preserving
the measured range. Seven fixed knot loads are 0.01, 1, 10, 100, 400, 1,000 and
5,000 Mbps. Every segment is a convex combination of nonnegative knot powers:
nonnegativity therefore holds continuously, not just at checked points. A 22,002-point
diagnostic grid is finite and nonnegative, with minimum 0.000146960 W. Zero load
and out-of-domain inputs raise errors; no extrapolation or clamping is permitted.
This mathematical property does not validate physical accuracy between load levels.

[Training-only uncertainty diagnostics](reports/task2c/training_uncertainty.csv)
use three leave-one-training-sweep-out refits of the selected complexity. Their
envelope is **not a calibrated confidence or prediction interval**. At 4 Mbps it
is 0.032770–0.033302 W. At 1 Mbps, the 0.011399–0.012630 W model envelope misses
the training sweep means (0.008900–0.009069 W), illustrating systematic approximation
error that small resampling variation does not capture. Internal-test low-load
bias is −0.002703 W. Measurement uncertainty and hardware transfer remain unresolved.

See the [comparison plot](reports/task2c/power_refit_comparison.png)
([PDF](reports/task2c/power_refit_comparison.pdf)).

## Isolated twin and validation-only consequences

The opt-in `DiagnosticPowerTwin` changes **only USR power**. It preserves the
original DPDK and both QoS models, balanced division by instance count, and
`total power = instances × per-instance power`. The original twin stays available
unchanged. The replacement uses explicit float64 Mbps inputs and JSON coefficients;
the legacy float32 inference fields continue to identify the unchanged QoS path.

The complete 15,120-outcome training/validation input census has **zero invalid
outputs for every action and partition**:

| Workload / role | Source rows per action | Scored rows per action | DPDK / USR(1) / USR(2) invalid |
|---|---:|---:|---:|
| Netflix training | 2,016 | 1,920 | 0 / 0 / 0 |
| Netflix validation | 504 | 504 | 0 / 0 / 0 |
| YouTube training-date block | 2,016 | 1,920 | 0 / 0 / 0 |
| YouTube validation-date block | 504 | 504 | 0 / 0 / 0 |

Previously USR(2) failed at one Netflix and two YouTube training slots; those
negative powers are resolved. YouTube remains transfer-only and was not used to
fit/select this model. The census is input checking, not controller performance.

All static evaluations use the identical **Netflix validation** horizon: 504
quarter-hour slots, 22 April 00:00–27 April 2019 05:45, totaling 126 hours.

| Static configuration | Task 2A energy Wh | Replacement Wh | Change Wh (%) | QoS violation rate |
|---|---:|---:|---:|---:|
| DPDK | 103.344021 | 103.344021 | 0 (0%) | 0 |
| USR(1) | 150.060349 | 145.257591 | −4.802757 (−3.2006%) | 0 |
| USR(2) | 151.085224 | 149.924831 | −1.160393 (−0.7680%) | 0 |

Energy is `sum(power_W × 0.25 h)` in Wh. Mean SEC changes from
0.00901118 to 0.00879901 J/Mbit for USR(1), and 0.00919883 to 0.00886988 for
USR(2); DPDK is unchanged. Every QoS value and every DPDK power value matches
Task 2A exactly. Repeated evaluations are exactly equal.

The initial configuration is eligible DPDK. Type-switch counts are 0/1/1 and
USR launches 0/1/2; there are no scale events or blocked requests. The existing
cooldown remains unchanged. Launch penalties affect reward, not integrated Wh;
these results add no physical startup energy. Static results cannot establish
learned-policy rankings or the submitted claims.

## Artifacts, verification and recommendation

The [selected artifact](models/task2c/selected_usr_power.json) has SHA-256
`3fc2b6b3b9cedb0c8abd17ea0740a6f721628c93c607e4397bb5fb1155f5c7f8`.
It records parameters, source/configuration/implementation hashes, exact training
rows, split and dependencies. The [frozen selection](reports/task2c/selection_frozen.json),
[twin manifest](reports/task2c/diagnostic/diagnostic_twin_manifest.json) and
[static summary](reports/task2c/diagnostic/static_validation_summary.json) link
the fitted model to its diagnostics. External measurement files still require
separate backup; hashes do not make this repository self-contained.

**Tests: 124 passed in 13.67 seconds**, comprising 101 existing and 23 new cases.
Checks cover grouped splits, forbidden fitting/selection roles, future-label
poisoning, deterministic fitting/reloading, continuous convex interpolation,
domain rejection, artifact tampering, balanced scaling and unchanged QoS/DPDK.
The [test log](reports/task2c_tests.log) and
[file manifest](reports/task2c/file_manifest.json) list execution evidence and all
new files. No preexisting revision file was modified; original tracked files,
models, datasets and Git references were preserved.

**Recommendation:** retain this frozen candidate for a small diagnostic controller
pilot after explicit authorization, using only training/validation traffic. It
resolves a demonstrated physical-sign defect with a modest surrogate change and
retains the three configurations. Do not treat it as independently certified or
begin the full claims-validation campaign yet. Before that campaign, resolve or
explicitly scope the DL/UL-to-traffic mapping and label alignment, assess QoS
threshold reliability, and verify causal forecast/controller integration under
the corrected contract. Confirmatory deployment claims require new physical
measurements. This refit neither confirms nor refutes the original PPO advantage;
all comparative results must ultimately be regenerated under one declared twin.

**Stopped after Task 2C.** No default promotion, controller modification, RL
training, reward change, final-test controller outcome access, manuscript edit,
commit or push occurred.

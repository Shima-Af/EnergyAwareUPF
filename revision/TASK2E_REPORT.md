# Task 2E — QoS surrogate reliability audit

**Verdict: neither original QoS surrogate currently supports a scientifically
validated controller pilot at τ=0.90.** Preserve both as historical references.
Resolve the label definition first, then refit/revalidate both using complete
sweeps; assess conditional uncertainty rather than applying a global bias shift.
No model was modified, fitted, recalibrated or retrained in this audit.

## Provenance and historical procedure

Recovered sources are `DDT_UPF_Selection_DRL/data/processed/dpdk_processed.csv`
(4,594 rows, 3 January 2025) and `oai_processed.csv` (4,564 rows, 13 January 2025).
Each has five complete sweeps, 110 case blocks, 22 offered-load levels
(0.01–5,000 Mbps) and 1,250-byte packets. Identical copies in `xRL_UpfSelection`
and `dynamic-upf-selection-rl` were hash-verified. All case, sweep, packet-rate,
packet-size and offered-load identifiers match the recovered campaign catalog.
Exact paths/hashes, dependencies and all model/scaler identities are in
[artifact provenance](reports/task2e/artifact_provenance.json).

Both saved QoS architectures are Dense(64, ReLU) → BatchNorm → Dropout(.20) →
Dense(32, ReLU) → BatchNorm → Dense(16, ReLU) → Dense(1, linear). Inference is
float32 per-instance offered load → `log1p` → saved input StandardScaler → neural
output → saved target StandardScaler inverse. The target is stored **`QoS_w`**,
dimensionless; it is not a measured violation probability. The output is unbounded
by architecture. No clipping was applied; all predictions at measurement loads are
finite and within [0,1]. This is not a guarantee for every possible input.

The recovered candidate training script consumes `QoS_w` directly. It splits rows
80/20 with seed 42, fits both scalers on training only, and uses the remaining rows
for early stopping, learning-rate reduction and reported accuracy. Its associated
configuration specifies Adam 0.001, 300 epochs, batch 8, stopping patience 25 and
LR patience 10. Saved architectures and **all reconstructed scaler means, scales
and training counts match**. This strongly supports the reconstruction but does
not authenticate an exact training run, model-initialization seed or epoch log.

| Historical split | DPDK | USR |
|---|---:|---:|
| Training / early-stopping rows | 3,675 / 919 | 3,651 / 913 |
| Case blocks shared across both sides | 110 / 110 | 110 / 110 |
| Sweeps shared across both sides | 5 / 5 | 5 / 5 |
| Validation rows adjacent to training in the same case | 887 / 919 | 865 / 913 |
| Median nearest same-case training observation | 3 seconds | 3 seconds |

Thus reported accuracy reuses model-selection data and correlated samples.
**No genuinely independent profiling evaluation is available for either saved
network.** Complete-sweep summaries below are descriptive: every sweep contributed
to historical training. Calling sweeps 4/5 validation/test now cannot retroactively
make these original models independent.

## QoS-label definition remains a blocking uncertainty

The submitted manuscript states latency/jitter/loss weights .40/.40/.20 and
thresholds 300 ms / 100 ms / 10⁻⁶. It does not fully specify the component-scoring
implementation. The recovered exploratory notebook instead uses .60/.20/.20,
downlink delay, `RAN_jitter`, and `RAN_PLR_interval`, with caps 300 / 30,000 / 1.
Its bounded linear formula **does not reproduce the stored labels**: mean absolute
differences are 0.045327 (DPDK) and 0.148794 (USR); maximum differences are 0.600102
and 0.715502. Only 93/4,594 and 1/4,564 labels match to 10⁻¹⁰.

Applying the notebook-style linear mapping with the manuscript's weights/thresholds,
explicitly assuming raw jitter is microseconds, also fails (MAE 0.044460 / 0.144039).
This rejects that particular reconstruction, **not every possible bounded scoring
function**. A different unused helper in the candidate training script combines
throughput, delay, jitter and loss using pooled MinMax scaling; its calls are
commented out. None is an authenticated original label builder. We did not
substitute another score or regenerate labels.

Even **identical recorded load, downlink delay, RAN jitter and RAN interval loss**
can have stored labels differing by 0.40. For DPDK, at 2 Mbps with delay 0,
jitter 2 and loss 0, labels span 0.56–0.96. For USR, at 200 Mbps with delay
0.002 ms, jitter 2 and loss 0, labels span 0.55999–0.95999. These discrepancies
suggest omitted components, hidden precision or a pipeline mismatch; they do not
identify the cause or establish that measurements are fabricated.

[Label checks](reports/task2e/label_consistency.json),
[formula comparisons](reports/task2e/label_formula_checks.csv) and
[source-row examples](reports/task2e/label_discrepancy_examples.csv) preserve the
evidence. Each mode has 110 recovered LoadCore ZIP exports. Representative raw
delay exports contain time-binned, cumulative-looking delay distributions; the
exact aggregation, differencing, missing-value treatment and KPI-to-label join
remain unauthenticated. A newer campaign's different validation pipeline was not
treated as the original procedure.

Other material limitations persist: recorded offered load is a per-direction
DL/UL objective, not an authenticated aggregate traffic variable; raw jitter units
and direction-to-score mapping need confirmation; packet-loss counters and interval
denominators/reset handling need verification. Three-second profiling observations
also do not directly validate 15-minute interval compliance. Balanced USR(2)
single-instance QoS reuse does not validate physical multi-instance QoS interaction.

## Performance and threshold errors

The first two rows reproduce the manuscript's rounded profiling metrics. They
are historical early-stopping scores, **not independent test evidence**.

| Model / scope | N | MAE | RMSE | R² | Bias, prediction − label |
|---|---:|---:|---:|---:|---:|
| DPDK / historical early-stopping validation | 919 | 0.042231 | 0.084657 | 0.152852 | −0.003636 |
| USR / historical early-stopping validation | 913 | 0.043053 | 0.078478 | 0.906810 | −0.005510 |
| DPDK / all rows, descriptive | 4,594 | 0.043544 | 0.087843 | 0.140183 | −0.002992 |
| USR / all rows, descriptive | 4,564 | 0.043453 | 0.083182 | 0.898526 | −0.003326 |
| DPDK / historical validation, load ≤400 Mbps | 629 | 0.038820 | 0.079200 | 0.186331 | −0.004329 |
| USR / historical validation, load ≤400 Mbps | 620 | 0.035257 | 0.069786 | 0.344111 | −0.002669 |

Residual distributions are asymmetric: historical-validation median / 95th / 99th
percentiles of prediction minus label are −0.024685 / +0.099812 / +0.383231 for
DPDK, and −0.012796 / +0.076744 / +0.382310 for USR. Small global bias therefore
does not imply reliable positive-tail errors.

False-safe means prediction ≥0.90 and label <0.90; false-unsafe reverses those
inequalities. Equality is safe. Denominators below are explicit.

| Historical validation | DPDK | USR |
|---|---:|---:|
| False-safe / all rows | 41 / 919 (4.46%) | 52 / 913 (5.70%) |
| False-safe / predicted-safe rows | 41 / 800 (5.13%) | 52 / 396 (13.13%) |
| False-safe / measured-unsafe rows | 41 / 103 (39.81%) | 52 / 515 (10.10%) |
| False-unsafe / all rows | 57 / 919 (6.20%) | 54 / 913 (5.91%) |
| False-safe measured shortfall: mean / maximum | 0.260900 / 0.380075 | 0.094569 / 0.420415 |
| False-safe optimism (prediction − label): mean / maximum | 0.316838 / 0.419736 | 0.108710 / 0.441273 |

These errors are not confined to tiny threshold crossings or extrapolation.
Within ≤400 Mbps, historical-validation false-safe counts remain **27/629 DPDK**
and **52/620 USR**. DPDK failures occur at 1–400 Mbps as well as higher loads.
USR false-safe cases occur at 6–400 Mbps, including 16 at 6 Mbps, 10 at 20 Mbps
and 9 at 200 Mbps. USR false-unsafe cases concentrate at 2/4 Mbps (21/33).
For USR(2), these are **per-instance** loads; total load is doubled.

For the descriptive band `|measured QoS−0.90|≤0.05`, DPDK has 91 validation rows,
RMSE 0.036784, 5 false-safe and 30 false-unsafe; USR has 358 rows, RMSE 0.031696,
34 false-safe and 53 false-unsafe. Conditioning on labels near the threshold
excludes severe errors. The separate **prediction-centered** band contains
300/538 rows and 16/52 false-safe cases. See
[band definitions and results](reports/task2e/near_threshold.json).

Across complete sweeps, false-safe fractions among predicted-safe rows range
**4.04–6.02% DPDK** and **7.23–23.43% USR**. These are descriptive between-sweep
ranges, not confidence intervals or deployment probabilities. All-row false-safe
totals are 190/4,594 and 343/4,564; correlation prevents interpreting rows as
independent trials. [Load-level errors](reports/task2e/errors_by_load.csv),
[sweep-level errors](reports/task2e/errors_by_sweep.csv),
[row predictions](reports/task2e/predictions.csv) and the
[plot](reports/task2e/qos_reliability.png) ([PDF](reports/task2e/qos_reliability.pdf))
retain the complete results.

## Why DPDK R² is low

The DPDK target is usually high (median 0.979875), but spans 0.16–1.00, with
standard deviation 0.094734; its 5th percentile is 0.757681. It is not simply a
constant target with harmless rounding variation. USR spans 0–1 with standard
deviation 0.261126. Similar absolute prediction errors therefore consume a much
larger fraction of DPDK's variation. USR's high global R² is helped by the large
load-dependent degradation above 400 Mbps; within the study range its historical
validation R² falls to 0.344111.

An exact, descriptive sum-of-squares decomposition is more informative:

| All-row decomposition | DPDK | USR |
|---|---:|---:|
| Target variation within the same offered-load level | 84.69% | 9.85% |
| Model squared error attributable to same-load variability | 98.50% | 97.05% |
| Best in-sample load-mean R² | 0.153092 | 0.901517 |
| Actual model R² | 0.140183 | 0.898526 |
| Global squared bias contribution to model squared error | 0.12% | 0.16% |

The in-sample load means are an arithmetic diagnostic, **not a new fitted
evaluation model or an independent performance ceiling**. For DPDK, 87.07% of
same-load variation is within individual sweep/load blocks, rather than between
sweeps. Historical-validation-only decomposition gives the same conclusion:
97.21% of DPDK model squared error is within-load variability.

Thus global bias and inadequate fitting of the average load curve are not the
dominant explanations. Most DPDK target variation is unavailable to a load-only
conditional-mean model. It could reflect real unobserved state, traffic/counter
effects, measurement variability or label-processing problems; **measurement
noise is not identified as the cause**. A larger mean-regression network alone
would not establish reliable per-observation safety. Full decomposition is in
[the machine-readable summary](reports/task2e/summary.json).

## Recommendations and gate before a pilot

| Model | Recommendation | Reason |
|---|---|---|
| DPDK QoS | **Refit/revalidate after label recovery**; retain original for history | Low-load-curve bias is small, but severe unpredictable unsafe labels occur at nominally safe loads. Mean recalibration cannot resolve this variability. |
| USR QoS | **Refit/revalidate after label recovery**; retain original for history | Global R² masks threshold errors and weaker study-range fit; false-safe risk varies substantially across sweeps. |

Neither should be kept unchanged as a validated QoS safety signal. A constant
offset or post-hoc clipping is not an adequate correction. Recalibration of
conditional uncertainty may later complement a verified model; it cannot repair
an unauthenticated target definition. Pure software smoke checks could still use
the original models as historical fixtures, but no scientific controller pilot
or RL training is recommended on the present QoS evidence.

Required next validation steps, **proposed only**:

1. Authenticate scoring code and units, component directions, loss denominators,
   cumulative/interval counter treatment and timestamp joins. Explain identical-KPI
   label discrepancies. Freeze either the submitted score or an explicitly revised
   definition; do not silently treat `QoS_w≥0.90` as a standardized service guarantee.
2. Reuse complete sweeps 1–3/4/5 for training/validation/internal-test organization,
   with scalers on training only and checkpoint/hyperparameter selection on
   validation only. This is internal redevelopment on already inspected data.
   Seek fresh held-out campaigns for confirmation. Group-based evaluation follows
   [the official guidance on correlated observations](https://scikit-learn.org/stable/modules/cross_validation.html).
3. Prespecify a small bounded-output neural candidate set retaining the original
   hidden architecture where useful, using a sigmoid head directly on authenticated
   [0,1] scores. Compare against training-mean and simple load-curve references.
   Do not impose global monotonicity without evidence. If timing/state matters,
   add features only if causally available to every controller and supported by
   new measurements. Do not run a model search in this audit.
4. Prespecify acceptable false-safe frequency **and severity**, by load and sweep,
   before fitting. Assess conditional lower-tail error or unsafe-event probability
   separately from mean QoS. Those are different prediction targets and must not
   silently replace the reward's QoS score. Use independent calibration groups
   if claiming calibrated uncertainty; five exposed sweeps do not justify naive
   row-wise intervals. Standard conformal guarantees depend on exchangeability,
   a substantive issue for correlated/heterogeneous data; see
   [Plassier et al., AISTATS 2024](https://proceedings.mlr.press/v238/plassier24a.html).
5. Freeze the resulting twin, repeat training/validation input checks and aligned
   diagnostics, and obtain pilot authorization. Keep traffic final testing untouched.
   Any paper claims must remain conditional on the validated score, operating domain
   and balanced-load approximation until physical evidence supports more.

## Files and verification

All additions are under `revision/`: the read-only audit, label checks, plotting
module, diagnostic tests, this report and evidence outputs. The
[file manifest](reports/task2e/file_manifest.json) lists every new path/hash;
[integrity evidence](reports/task2e/execution_and_integrity.json) verifies prior
artifacts. All original models/scalers, datasets, Task 2C/2D files, manuscripts and
Git references are unchanged; work remains on `revision/comcom-clean-evaluation`.

**11 focused tests passed in 0.90 seconds.** They check threshold equality,
denominators, false-safe magnitude, residual signs, variance identities, historical
split reconstruction, distinct threshold bands and label-evidence limitations.
The prior 139-test suite was not rerun: the foundation was unchanged and this task
required no fitting. [Test log](reports/task2e_tests.log),
[audit log](reports/task2e_audit.log). The audit log includes pandas timestamp-format
inference warnings; parsed nearest-neighbor spacing was explicitly verified as
three seconds. No model failure was suppressed or repaired.

**Stopped after Task 2E.** No profiling-model training, power search, default-twin
change, controller/RL training, traffic final-test evaluation, manuscript edit,
commit or push occurred.

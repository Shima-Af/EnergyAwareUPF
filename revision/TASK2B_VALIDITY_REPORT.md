# Task 2B — profiling-model validity audit

**Verdict:** refit and independently validate the USR power surrogate before PPO
training under the full three-action contract. The original model returns negative
power at an actual Netflix training slot. No correction or refitting was implemented.

## Root cause and measurement evidence

The recovered original USR CSV contains **4,564 rows**, 22 offered-load levels,
five repeated test sets and 110 case blocks, all with 1,250-byte packets. There are
**no negative measured power values**, 352 exact zeros, and no zero-offered-load
experiment. The 0.01–20 Mbps region contains 1,865 rows at nine levels across 45
blocks. These are process-attributed estimates, not whole-server electrical power.

| Per-instance offered load | Measured mean power | Original-model power |
|---:|---:|---:|
| 0.01 Mbps | 0.000195 W | 0.060257 W |
| 0.1 Mbps | 0.001156 W | 0.052894 W |
| 1 Mbps | 0.009007 W | 0.005958 W |
| 2 Mbps | 0.017470 W | **−0.021882 W** |
| 4 Mbps | 0.034704 W | **−0.053532 W** |
| 6 Mbps | 0.051405 W | 0.045395 W |
| 8 Mbps | 0.068602 W | 0.092808 W |
| 10 Mbps | 0.085635 W | 0.102490 W |
| 20 Mbps | 0.172508 W | 0.207182 W |

[Measurement statistics](reports/task2b/measurement_by_load.csv) retain sample
ranges, variability and zeros; the [plot](reports/task2b/usr_power_low_load.png)
([PDF](reports/task2b/usr_power_low_load.pdf)) shows the original curve without
clipping. Error bars are sample ranges, not confidence intervals. Positivity
alone would not fix the substantial positive bias at the smallest loads.

**The observed mechanism is unconstrained regression in standardized target space.**
The saved target scaler has mean 1.4193139895 W and scale 1.5199246406 W. Physical
zero corresponds to standardized output **−0.9338055003**. The network's linear
head produces −0.9482024312 at 2 Mbps and −0.9690257311 at 4 Mbps, below that bound;
inverse scaling therefore correctly returns negative watts.

- Input `log1p` is applied once, with finite scaled inputs inside the measured
  load range. These failures are interpolation errors, not low-load extrapolation.
- Saved input/target means, scales and sample counts exactly match the reconstructed
  3,651 training rows from the documented 80/20 seed-42 split. A float64 target
  transform/inverse round trip has maximum error 4.44e−16 W. There is no evidence
  of a swapped scaler or erroneous inverse transform.
- The recovered training script uses `power_consumption_watts` directly and has
  no baseline subtraction. The corrected adapter also subtracts no idle term.
  StandardScaler centering is reversed at inference; it is not physical idle-power
  subtraction. Upstream label-generation/attribution provenance is still incomplete,
  so subtraction before these CSVs were created cannot be categorically excluded.
- The unbounded output and global MSE objective provide no nonnegativity guarantee.
  The low-load region is 40.9% of rows but contributes only about 1.01% of the
  current model's total squared residuals. This descriptive calculation is consistent
  with poor low-load fitting being obscured by global error; it is not independent
  accuracy validation or proof of the optimizer's causal history.

[Inference-stage evidence](reports/task2b/power_inference_stages.csv) records each
transformation. Task 2A's version checks produced much smaller changes and already
found negative archived predictions. This is not explained by rounding noise.
A fixed 0.01-Mbps diagnostic grid found negative power at samples **1.18–5.09 Mbps**
(minimum −0.054541 W). These sampled limits are not validated operating boundaries.
The grid's 0-Mbps point is explicitly unmeasured extrapolation.

## Invalid outcomes on training and validation inputs

All three configurations were queried at every source load, with balanced division
for USR(2). This is an input-validity census, **not controller rollouts or rewards**.
"Invalid" means negative/nonfinite power or QoS outside [0,1]; ordinary QoS below
the 0.90 target is not classified as an invalid prediction.

| Workload / chronological block | Rows per action | DPDK invalid | USR(1) invalid | USR(2) invalid |
|---|---:|---:|---:|---:|
| Netflix training, all source rows | 2,016 | 0 | 0 | 1 (0.0496%) |
| Netflix training, scored after warmup | 1,920 | 0 | 0 | **1 (0.0521%)** |
| Netflix validation | 504 | 0 | 0 | 0 |
| YouTube training-date block, all rows | 2,016 | 0 | 0 | 2 (0.0992%) |
| YouTube training-date block, after warmup | 1,920 | 0 | 0 | **2 (0.1042%)** |
| YouTube validation-date block | 504 | 0 | 0 | 0 |

YouTube remains transfer-only: its chronological block names do not authorize
parameter fitting or tuning. No final-test inputs were passed to the models.
All failures here are negative power; there were no nonfinite or out-of-range QoS
outputs in these blocks. Numeric validity does not establish QoS calibration.

The Netflix failure occurs on **16 April 2019 05:45**, source row 1463: total load
8.720405 Mbps becomes 4.360202 Mbps per USR instance, producing **−0.078472 W total**.
The two YouTube failures occur on 15 April at 12:00 and 12:15. The
[three failing records](reports/task2b/invalid_workload_predictions.csv) and
[complete 15,120-row census](reports/task2b/workload_prediction_census.csv) preserve
the inputs, timestamps, outputs and failure reasons. A low frequency does not make
negative energy an acceptable reward signal.

## Independence and support for refitting

The recovered power script randomly splits rows 80/20, fits scalers on training,
uses the other 20% for both `EarlyStopping(val_loss)` and learning-rate reduction,
and then reports accuracy on those same rows. Thus, scaler fitting is training-only,
but there is **no independent final profiling test** in that procedure.

For both USR and DPDK, **all 110 case blocks and all five test sets appear in both
sides** of the reconstructed split. Of 913 USR validation rows, 865 have an adjacent
training row from the same case; the median nearest training sample is three seconds
away. DPDK has 887/919 and the same median spacing. These are correlated within-run
samples, not independent campaign replicates. Saved scaler statistics strongly
support this reconstruction, although an authenticated original training manifest
is still missing. [Block-level evidence](reports/task2b/measurement_by_block.csv)
and [strict timestamp verification](reports/task2b/timestamp_verification.json)
are saved; the recovered QoS training script likewise reuses early-stopping data
for its reported accuracy.

**Available data support a candidate refit, not an independently certified model.**
Five sweeps and dense low-load sampling are sufficient to try a small nonnegative
model. There is no true idle measurement, no packet-size diversity, and only one
USR measurement day. Zero attributed values need attribution/resolution checks;
they do not establish zero whole-server power. The external `raw/*grouped_limited`
files are already aggregated exports: 100 groups rather than 110, and some means
do not match processed data (USR maximum difference 0.834581 W; at ≤20 Mbps,
0.000586 W). Do not substitute or merge them as authenticated raw measurements.

## Correction options and paper implications

| Option | Advantages | Disadvantages and implications |
|---|---|---|
| Nonnegative constrained regression | Nonnegative coefficients on nonnegative bases, or nonnegative fitted knot values, can guarantee nonnegative power in the declared domain. Retains the three-action study. | Requires refitting, regularization/complexity selection and independent grouped validation. Constraints must apply in **watts**, not by forcing the current standardized output ≥0, which would incorrectly impose power ≥1.419 W. |
| Simple interpretable power model | A nonnegative intercept/slope or a small segmented/saturating curve is transparent, data efficient and easier to audit. Low-load measured means support considering it. | A single global line is inappropriate for the 0.01–5,000 Mbps range with saturation/nonmonotonic high-load behavior. Declare the study domain, validate residuals and avoid assuming global monotonicity. This changes the paper's MLP power-model methodology. |
| Explicit output constraints | A declared physical nonnegative output parameterization can prevent violations; post-hoc projection `max(0,P)` is transparent if separately justified. | Projection alone creates zero-cost positive-traffic regions and leaves positive low-load bias intact. It changes controller incentives and does not establish calibration. A constrained neural head generally needs refitting; forcing the current standardized output ≥0 would impose the wrong physical lower bound. Silent clipping is unacceptable. |
| Restrict operating domain | Appropriate when measurements cannot support a region; explicit rejection prevents exploitation. | Selecting a cutoff merely where predictions become positive is not physical validation. Removing affected rows changes the workload/horizon; excluding USR(2) changes action availability. These would narrow the submitted study and require a new declared contract, not repair the current three-action problem. |

**Recommendation:** after resolving label provenance, refit **USR power** with a
small interpretable, physically nonnegative model, considering a regularized
nonnegative piecewise-linear model and a simple constrained affine/segmented
baseline. Select complexity using profiling validation groups, not traffic-controller
performance. No candidate was fitted or ranked in this audit. This recommendation
does not require retraining every surrogate solely to fix positivity, but all
surrogates still need an independent validity check.

Refitting is necessary for this recommended correction while preserving the full
workloads/actions. Explicitly justified projection/domain restrictions are logical
alternatives, not equally validated substitutes. The revised paper must describe
the changed surrogate and grouped validation and regenerate all affected controller
comparisons under one twin. Historical results remain conditional on the old twin.

## Minimum gate before controller-training authorization

1. Authenticate the power label/attribution and offered-load definitions, grouping,
   zero readings and available acquisition logs; resolve aggregated-export mismatch.
   Add true idle/near-idle measurements if claiming behavior there. Keep attributed
   power and whole-server electricity distinct.
2. Prespecify a per-instance domain covering all three actions on the intended
   workloads, model families, complexity limits and error/decision-robustness
   criteria tied to measurement uncertainty. Do not define the domain from the
   negative prediction's zero crossings or tune on controller final-test outcomes.
3. After approval, fit only within new grouped profiling splits: whole repeated
   sweeps/campaigns must stay together; scalers fit on training only; early stopping
   and model selection use validation only. A three-sweep/train, one/validation,
   one/test split is an internal-check option, not a newly untouched dataset after
   this audit. Seek a fresh held-out campaign for confirmatory physical claims.
4. Require nonnegativity by construction across the declared interpolation domain,
   finite outputs and appropriate boundary behavior. Check held-out error by load
   and sweep, low-load bias, uncertainty and controller configuration-ranking
   robustness; a single global RMSE is insufficient. Resolve QoS threshold reliability
   as a separate requirement.
5. Save new revision-specific models/scalers and provenance, repeat the full input
   validity census and Task 2A validation-only static checks, and approve those
   results before starting controller training. Preserve historical artifacts.

Diagnostics: [summary](reports/task2b/validity_summary.json),
[artifact manifest](reports/task2b/artifact_manifest.json),
[execution log](reports/task2b_execution.log), and
[seven passing diagnostic tests](reports/task2b/diagnostic_test_results.txt).
The earlier 94-test suite was not rerun because the evaluation foundation was not
changed. All additions are listed in the [file manifest](reports/task2b/file_manifest.json).
No existing model, dataset, Task 1/2A file, manuscript, or Git reference was changed.
No correction, training, commit or push was performed. **Stop after Task 2B.**

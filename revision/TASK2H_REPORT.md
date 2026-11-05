# Task 2H — grouped QoS training and validation

**Outcome:** eight bounded neural runs completed under the approved protocol.
Validation selected `original_hidden_sigmoid_seed_123` for DPDK and
`original_hidden_sigmoid_seed_42` for USR. Both retain the original hidden
architecture. They are useful conditional-mean candidates, **not validated QoS
safety signals**. No risk tolerance is invented, no model is promoted, and no
controller training is authorized by this report.

## Protocol and reproducibility

The unchanged [score specification](specifications/qos_score_v1.json) and
[Task 2G protocol](configs/qos_validation_v1.json) were checked against their
recorded hashes before training. Historical `QoS_w` targets and source CSVs are
unchanged. Complete sweeps 1–3 trained parameters/scalers, sweep 4 selected
checkpoints/candidates and sweep 5 supplied internal evaluation. Group sizes are
DPDK 2,760/916/918 and USR 2,736/915/913. No random row split or traffic dataset
was used.

Each mode received the prescribed original 64/32/16 ReLU architecture with
BatchNorm/dropout and its plain dense variant, seeds 42/123, a sigmoid head in
original score units and training-only log1p-load standardization. The approved
Adam/batch/stopping schedule was used. Total training was **1,430 epochs out of
the 4,800-epoch cap**. All eight runs succeeded; no retry, extra scientific fit,
search extension or train-plus-validation refit occurred. The
[run ledger](reports/task2h/candidate_runs.jsonl) records every start/completion;
each model directory contains its selected weights, manifest and epoch history.
All eight neural candidates and four simple references return finite [0,1]
outputs on a 22,000-point interpolation-domain grid. The sigmoid/convex-mean
construction supplies boundedness between samples; the grid does not validate
physical accuracy there. [Bounded-output checks](reports/task2h/bounded_output_checks.json)
record the ranges for every model.

Both mode selections were written to [frozen selection](reports/task2h/selection_frozen.json)
**before any internal-test scoring**. Neural selection used validation RMSE only.
Unselected networks have training/validation results but intentionally **no
internal-test scores**, as required by the approved protocol. Simple references
use only training means: one constant and one linear interpolation through the
22 training load-level means. Their coefficients are saved alongside the neural
artifacts. Original-network comparisons reuse hash-linked Task 2E predictions on
identical source rows and are explicitly non-independent.

| Architecture / seed | DPDK validation RMSE | USR validation RMSE | DPDK epochs run / selected | USR epochs run / selected |
|---|---:|---:|---:|---:|
| Original hidden / 42 | .080463 | **.078010** | 111 / 31 | 121 / 41 |
| Original hidden / 123 | **.080406** | .078081 | 165 / 85 | 167 / 87 |
| Plain hidden / 42 | .080411 | .078114 | 133 / 53 | 309 / 229 |
| Plain hidden / 123 | .080456 | .078171 | 137 / 57 | 287 / 207 |

The small RMSE differences select candidates; they do not establish statistical
superiority of an architecture or seed. All four neural candidates within each
mode have identical validation safe/unsafe classifications at the measured loads.

## Comparative prediction quality

V = validation; T = internal test. All values are dimensionless composite-score
units. “Historical” models trained on rows from all five sweeps, so those rows
are descriptive comparisons, **not independent benchmark estimates**.

| Mode / model | Role | MAE | RMSE | R² | Prediction − label bias |
|---|---|---:|---:|---:|---:|
| DPDK selected NN | V | .036185 | .080406 | .147794 | −.001208 |
| DPDK constant | V | .054015 | .087565 | −.010724 | −.009020 |
| DPDK load-mean curve | V | .041085 | .084049 | .068820 | −.008837 |
| DPDK historical NN* | V | .042525 | .081410 | .126371 | −.008085 |
| DPDK selected NN | T | .038030 | .082881 | .056080 | −.002544 |
| DPDK constant | T | .051601 | .085939 | −.014859 | −.010399 |
| DPDK load-mean curve | T | .043908 | .087281 | −.046812 | −.010272 |
| DPDK historical NN* | T | .043032 | .083176 | .049337 | −.009446 |
| USR selected NN | V | .036183 | .078010 | .908083 | +.006717 |
| USR constant | V | .226772 | .257309 | −.000006 | +.000651 |
| USR load-mean curve | V | .042434 | .080354 | .902476 | −.000579 |
| USR historical NN* | V | .038483 | .078698 | .906456 | −.005274 |
| USR selected NN | T | .040726 | .083342 | .904454 | +.014718 |
| USR constant | T | .237110 | .269742 | −.000869 | +.007948 |
| USR load-mean curve | T | .044115 | .082616 | .906113 | +.007384 |
| USR historical NN* | T | .040147 | .080449 | .910972 | +.002616 |

The simple USR load-mean curve has slightly better internal-test RMSE/R² than
the selected NN, while the NN has lower MAE. The historical USR network also has
lower internal-test RMSE, with the substantial caveat that it saw those sweeps.
Neither result changes the frozen neural selection or establishes that a neural
model must win. The two simple references remain appropriate sensitivity controls.

All models' available training/validation metrics and selected/baseline internal
metrics are in [metrics.json](reports/task2h/metrics.json). Full residuals,
load/sweep breakdowns and macro load summaries are in
[predictions](reports/task2h/predictions.csv),
[load errors](reports/task2h/errors_by_load.csv),
[sweep errors](reports/task2h/errors_by_sweep.csv) and
[macro metrics](reports/task2h/macro_load_metrics.csv).

Within the **≤400 Mbps** per-instance range, selected DPDK V/T RMSE is
.082123/.080278 and R² .194652/.079436. Selected USR V/T RMSE is .077002/.077997
and R² .249744/.280871. Thus USR's excellent full-range R² is helped by large
load-dependent degradation above the study range. Within USR(2)'s **≤200 Mbps**
per-instance range, V/T RMSE is .075433/.077295 and R² .261442/.300388.
[Domain metrics](reports/task2h/study_domain_metrics.json) contain the same scopes
for every evaluated model; none uses traffic final-test performance.

## Threshold reliability at 0.90

“Safe” here means the **composite score** meets .90, not a 5QI guarantee.
False-safe (FS) is prediction ≥.90 with measured label <.90; false-unsafe (FU)
is the converse. All-row frequencies, predicted-safe coverage and conditional
fractions must be read together.

| Selected model / role | FS / all | FS / predicted safe | FU / all | Predicted-safe coverage | FS measured shortfall mean / p95 / max |
|---|---:|---:|---:|---:|---:|
| DPDK V | 37/916 (4.04%) | 37/834 (4.44%) | 36/916 | 91.05% | .297534 / .340069 / .340165 |
| DPDK T | 43/918 (4.68%) | 43/836 (5.14%) | 51/918 | 91.07% | .278458 / .340990 / .341950 |
| USR V | 62/915 (6.78%) | 62/417 (14.87%) | 57/915 | 45.57% | .110119 / .380396 / .404195 |
| USR T | 60/913 (6.57%) | 60/415 (14.46%) | 51/913 | 45.45% | .102732 / .380201 / .500495 |

Of measured-unsafe rows, the selected models miss 37/83 and 43/74 for DPDK V/T;
USR misses 62/503 and 60/507. Maximum false-safe **prediction optimism** is
.402219/.407474 for DPDK and .428343/.543916 for USR. These are substantial errors,
not only rounding or tiny threshold crossings.

On internal test, the constant DPDK model predicts every row safe: 74/918 FS,
zero FU and 100% coverage. The DPDK load-mean reference gives 40/794 FS among
predicted-safe rows, 90 FU and 86.49% coverage. The USR constant predicts **no
rows safe**, with zero FS but 406 FU and zero coverage; its FS/predicted-safe rate
is **undefined**, not evidence of zero operational risk. The USR load-mean reference
has 59/374 FS among safe predictions, 91 FU and 40.96% coverage. Full corresponding
validation and training statistics remain in the metrics file.

Within ≤400 Mbps on internal test, DPDK has **24/543** FS among predicted-safe
rows and USR **60/415**. USR(2)'s ≤200 Mbps input domain has **49/373**. Errors
therefore persist in the intended input ranges, not only at saturation.

In the measured-centered ±.05 threshold band, selected DPDK V/T RMSE is
.050482/.053307 (59/100 rows, FS 1/3, FU 35/51). USR RMSE is .017942/.022193
(420/389 rows, FS 39/43, FU 57/51). Prediction-centered bands retain severe
false-safe errors: USR includes all 62/60 FS there. Conditioning on measured
labels near .90 hides the larger errors. [Band metrics](reports/task2h/near_threshold_metrics.json)
retain both definitions for all models.

For USR, every validation/internal safe/unsafe decision at measured load levels
is unchanged from the historical NN. DPDK changes only the classification at
5,000 Mbps, outside the controller study range; internal-test FS rises 40→43 and
FU falls 90→51 overall. Bounded outputs alone do not resolve safety errors.

## DPDK: what load can explain

The selected NN improves RMSE over the training constant by **8.18% V / 3.56% T**
(MSE reductions 15.68%/6.99%). This is useful but modest empirical predictive
information; it is not an inferential significance result from independent
campaigns. Internal-test R²=.056080 leaves **94.39%** of that sweep's target
variation unexplained.

In-sample arithmetic load means show that **80.69% V / 89.48% T** of DPDK target
variation occurs at identical offered loads. This contributes approximately
**94.69% / 94.79%** of selected-network squared error. Even per-load means computed
on those same evaluation labels achieve only R² .193076/.105224; they are a
descriptive reference, not a new fitted benchmark or an independent ceiling.
The selected network approximates the mean curve reasonably well, but load alone
does not explain realized throughput/quality fluctuations. Task 2F established
that the score's received-throughput component explains many formerly unexplained
same-DL-KPI label jumps. That is real omitted information for a load-only input,
not evidence that the labels must be replaced or that a larger network will fix it.

[Selected-model variance decomposition](reports/task2h/selected_variance_decomposition.json)
uses actual predictions. A 5.96×10⁻⁸ same-load float32 batch-position variation
in USR validation creates a small cross term; the diagnostic retains that term
instead of altering predictions. Identical ordered inference calls and reloads
are exactly repeatable. This does not assert bitwise invariance across different
batch shapes or hardware. The smallest selected-model measured-load threshold
margin is approximately .00255, far larger than that observed numerical variation.

## Decisions and smallest next correction

1. **Can DPDK QoS be predicted meaningfully from load alone?** Modestly as a
   conditional mean, demonstrated by improvement over the constant. Not with
   strong per-observation threshold reliability: most within-load variation is
   inaccessible to that input, and severe FS outcomes remain.
2. **Is USR reliable near .90?** It predicts the broad mean curve well, but the
   present results do not support calling it a reliable safety classifier.
   Internal FS/predicted-safe is 60/415 with maximum measured shortfall .500495.
   No approved tolerance exists with which to declare formal acceptance/rejection.
3. **Which models enter a pilot?** Preserve the two validation-selected bounded
   networks as candidates for a later, explicitly scoped exploratory comparison,
   with the simple load-mean models as sensitivity references. **No model is
   approved here as a sole safety gate or promoted to the twin, and no controller
   training is automatically authorized.** A purely model-conditional pilot would
   require explicit acceptance of these limitations; it would not validate actual
   QoS guarantees.
4. **What risks remain?** The five sweeps are exposed same-campaign data, not a
   new physical test; rows are correlated; repeated seeds do not create physical
   replication. DN/RAN direction, packet-loss denominator and workload mapping
   retain Task 2G limitations. Conditional means can hide lower-tail failures;
   USR(2) QoS still assumes balanced independent instances. Operational 5QI and
   service-continuity claims remain unsupported.

**Smallest next correction proposed:** retain the fixed score and these frozen
point models; add an explicitly separate, training/validation-only per-load
lower-tail/error-risk diagnostic using complete sweeps. First agree acceptable
FS frequency, severity and useful coverage before defining any guard or risk
threshold. Do not silently replace mean QoS in the reward with a conservative
quantile, reject-all rule or unsafe-event probability. Fresh grouped calibration
measurements and clarified counter semantics are needed for calibrated physical
claims. If better state prediction is pursued later, use only causally available
telemetry; same-interval realized throughput cannot be fed into start-of-interval
decisions as if known in advance. No further neural search is proposed here.

## Artifacts, tests and stop

Models are under `revision/models/task2h/{dpdk,usr}/`. Each neural candidate has
`model.keras`, `manifest.json` and `history.csv`; both simple models are serialized
separately. Manifests pin source/config/score/code hashes, exact training rows,
training-only scaler, seed and best epoch. [Runtime/protocol evidence](reports/task2h/prespecified_protocol.json),
[training log](reports/task2h_training.log), [comparison plot](reports/task2h/grouped_qos_comparison.png)
([PDF](reports/task2h/grouped_qos_comparison.pdf)), and
[file manifest](reports/task2h/file_manifest.json) complete the record.

**50 tests passed in 2.84 seconds:** 18 new grouped-QoS tests plus 32 existing
data/selection tests. No extra neural training was performed by these tests.
Checks cover deterministic reload/inference, bounded outputs, invalid domains,
artifact tampering, training-only scalers/baselines, exact disjoint sweep membership,
the eight-run ledger, validation selection and withheld losing-candidate test scores.
The [test log](reports/task2h_tests.log) includes the results. A post-processing
variance assertion initially assumed exact same-load floating predictions; it was
generalized to retain the numerical cross term and tested. No model was retrained
or selected in response.

[Integrity evidence](reports/task2h/execution_and_integrity.json) verifies all prior
revision files and original profiling artifacts unchanged. Work remains on
`revision/comcom-clean-evaluation`. **Stopped after Task 2H:** no score changes,
default-twin promotion, PPO/recurrent PPO/DQN training, traffic final-test outcomes,
manuscript edits, commit or push.

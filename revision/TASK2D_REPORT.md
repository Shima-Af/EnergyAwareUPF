# Task 2D — constrained neural USR power comparison

**Recommendation: retain the frozen Task 2C piecewise-linear candidate for the
current diagnostic foundation.** A physically nonnegative network can retain the
original hidden architecture, but the validation-selected network has substantially
more low-load bias than Task 2C. Its small overall RMSE advantage does not establish
equivalent quality for configuration comparisons. Neither candidate is promoted
to the default twin. This result does not prove neural surrogates cannot work;
it describes the completed, limited search.

## Prespecified experiment and independence

[The frozen protocol](configs/usr_power_nn_v1.json) was written before fitting.
Exactly four scientific candidate runs were trained: two architectures × seeds
42 and 123. Each had a maximum of 600 epochs, Adam learning rate 0.001, batch size
64, validation early-stopping patience 80, learning-rate reduction after 30
unimproved epochs (factor 0.5, minimum 0.00001), and restoration of the best
validation weights. Total scientific training was 1,093 epochs out of a 2,400-epoch
cap. Two additional two-epoch **synthetic unit-test fits** checked reproducibility;
they were not candidate runs or part of selection.

We reused the exact Task 2C source hash, source-row assignments and complete sweep
split: sweeps 1–3 train (2,736 rows), sweep 4 validation (915), sweep 5 internal
test (913). Input `log1p(Mbps)` mean/std were fitted on training only. The loss
was unweighted row MSE in watts, matching Task 2C. Checkpoint, architecture and seed
selection used profiling-validation RMSE only. No traffic/controller results were
used. The selected manifest was frozen before internal-test scoring; unselected
networks were not scored on internal-test outcomes. No search expansion or
train-plus-validation refit occurred.

This is an evaluation on disjoint measurement sweeps, **not a new independent
physical campaign or fully blind dataset**. All sweeps are from the same device/day,
with 1,250-byte packets and 22 load levels. Task 2C already inspected sweep 5. Its
provenance limitations remain: exact timestamp join, per-direction DL/UL objective
versus aggregate workload mapping, attribution uncertainty and absent true idle
measurements. See [Task 2C](TASK2C_REPORT.md) and the
[execution protocol](reports/task2d/prespecified_protocol.json).

## Minimal architecture changes and physical validity

The saved original USR power network is Dense(64, ReLU) → BatchNorm → Dropout(.15)
→ Dense(32, ReLU) → BatchNorm → Dense(1, linear). Its output was standardized power.

The first variant retains every hidden layer and replaces only the linear output
activation with **softplus in physical watts**. The centered target transform is
removed: applying softplus before the original centered inverse transform would
impose the wrong physical lower bound. The second variant removes BatchNorm and
Dropout, retaining the same 64/32 dense widths and softplus head. Those changes
are explicit alternatives, not modifications of the historical artifacts. Training
budget and grouped scaling are new; historical training is not reproduced.

For every finite latent output, `softplus(z) = log(1 + exp(z)) ≥ 0`; Keras uses its
stable implementation. No inverse target centering, clipping, projection, or output
repair is applied. Hidden layers may vary between measured loads, but the physical
head remains nonnegative. The declared interpolation domain is **0.01–5,000 Mbps
per instance**; zero, nonfinite and out-of-domain loads raise errors. All saved
parameters and outputs are finite. All four networks pass a 22,000-point domain
grid, including between measured levels. The selected network's grid minimum is
0.027130 W. Nonnegativity is guaranteed by the head; physical accuracy between
measurement levels is not guaranteed by this construction or grid check.

## Validation selection and comparison

| Candidate | Epochs run / selected | Validation RMSE W | Validation low-load RMSE W, ≤20 Mbps |
|---|---:|---:|---:|
| **Original hidden layers + softplus, seed 42** | **311 / 231** | **0.263503** | **0.020966** |
| Original hidden layers + softplus, seed 123 | 387 / 307 | 0.264375 | 0.034579 |
| Plain 64/32 + softplus, seed 42 | 108 / 28 | 0.264763 | 0.009646 |
| Plain 64/32 + softplus, seed 123 | 287 / 207 | 0.263855 | 0.009771 |

Selection retains the original hidden architecture (2,625 total parameters),
seed 42. The plain variants have 2,241 parameters. The better low-load results of
the plain variants do not override the frozen overall-RMSE selection rule. No
post-hoc switch of selection objective or winning seed was made.

| Model / role | RMSE W | MAE W | R² | Bias W | Low-load RMSE W | Low-load bias W |
|---|---:|---:|---:|---:|---:|---:|
| Selected NN / validation | 0.263503 | 0.080958 | 0.969994 | +0.010838 | 0.020966 | +0.018820 |
| Frozen Task 2C / validation | 0.266462 | 0.079240 | 0.969316 | +0.009571 | 0.009086 | −0.003069 |
| Original NN / validation* | 0.264858 | 0.094256 | 0.969684 | −0.003256 | 0.044541 | +0.005241 |
| Selected NN / internal test | 0.306927 | 0.109318 | 0.957589 | +0.040959 | 0.021716 | +0.019175 |
| Frozen Task 2C / internal test | 0.309299 | 0.105602 | 0.956931 | +0.039595 | 0.010058 | −0.002703 |
| Original NN / internal test* | 0.302618 | 0.117502 | 0.958772 | +0.026958 | 0.045022 | +0.005528 |

*Original-network scores are **descriptive only**: its historical training split
included rows from all five sweeps. They are reused from the hash-linked Task 2C
predictions under the same deterministic inference profile. They are not unbiased
held-out estimates. Task 2C coefficients remain frozen and its newly computed
metrics match the prior results exactly.

The selected NN lowers overall RMSE relative to Task 2C by 0.002958 W on validation
and 0.002372 W on internal test. Its MAE is slightly worse on both. Its low-load
RMSE is **2.31× / 2.16×** Task 2C's validation/internal-test errors. Low-load MAE
is 0.018820/0.019175 W versus 0.005953/0.006106 W. No inferential significance
claim is made from one validation and one same-day internal-test sweep.

Selected-network RMSE by sweeps 1–5 is 0.284614, 0.287106, 0.289930, 0.263503 and
0.306927 W. Complete train/validation/test metrics, per-load/per-sweep biases and
residuals are saved in [metrics](reports/task2d/metrics.json),
[load errors](reports/task2d/errors_by_load.csv),
[sweep errors](reports/task2d/errors_by_sweep.csv) and
[row predictions](reports/task2d/predictions.csv). The corresponding original-model
residuals remain in [Task 2C](reports/task2c/errors_by_load.csv).

## Low-load bias and implications for configuration comparisons

All following values are watts per instance; measured means use validation only.

| Mbps | Measured mean | Selected constrained NN | Frozen Task 2C | Original NN* |
|---:|---:|---:|---:|---:|
| 0.01 | 0.000196 | 0.027130 | 0.000147 | 0.060257 |
| 0.1 | 0.001087 | 0.029340 | 0.001234 | 0.052894 |
| 1 | 0.009038 | 0.033284 | 0.012106 | 0.005958 |
| 2 | 0.017610 | 0.032830 | 0.019097 | −0.021882 |
| 4 | 0.034775 | 0.048109 | 0.033080 | −0.053532 |
| 10 | 0.084578 | 0.101663 | 0.075027 | 0.102490 |
| 20 | 0.176787 | 0.189104 | 0.168777 | 0.207182 |

Positivity fixes the original sign defect but does not remove low-load bias.
At 0.01 Mbps the selected network overpredicts by approximately 138×; the lowest
loads are especially sensitive to small absolute errors. The original-hidden
network's two-seed range at that point is 0.027130–0.037255 W; both miss the measured
mean. Seed spread therefore misses common systematic error and is **not a
calibrated confidence or prediction interval**. Task 2C also retains approximation
error; it is not physical ground truth. See the
[comparison plot](reports/task2d/neural_power_comparison.png)
([PDF](reports/task2d/neural_power_comparison.pdf)).

Post-selection diagnostics use a **synthetic 2,010-point total-load grid** from
0.02 to 400 Mbps, balanced USR division and unchanged original DPDK/QoS predictions.
No traffic files, controller rollouts, cooldown policy or integrated energies enter
these diagnostics. At 20 Mbps total:

| Power model | USR(1) W | USR(2) W | Lower predicted USR power |
|---|---:|---:|---|
| Selected NN | 0.189104 | 0.203326 | USR(1) |
| Task 2C | 0.168777 | 0.150054 | USR(2) |

USR(1)/USR(2) ordering differs between NN and Task 2C at 1,109 grid points; the
lowest-power action among all three differs at 361 points (also 361 when filtering
by predicted QoS ≥0.90). The two original-hidden neural seeds disagree on USR
ordering at 439 points. These grid counts depend on sampling; they are **not
workload frequencies or controller-performance estimates**.

A conservative stress scenario propagates maximum absolute **validation** residuals
from bracketing load levels, multiplying by instance count. Its combined error
budget exceeds the predicted USR(1)/USR(2) power margin at all grid points. This
demonstrates lack of robustness under that scenario, not a calibrated probability
of reversed rankings. It allows worst-case independent error signs even though
both actions share one fitted function; those signs need not be jointly attainable.
DPDK/QoS uncertainty is excluded, so predicted feasibility is not a safety claim.
Details are in the [scenario summary](reports/task2d/comparisons/configuration_comparison_summary.json)
and [per-load comparisons](reports/task2d/comparisons/configuration_power_comparisons.csv).

## Decision, artifacts and limits

**The original hidden architecture can be retained with guaranteed nonnegative
watts. However, this budget does not support choosing that network over Task 2C
without sacrificing low-load prediction quality.** Retain Task 2C as the current
diagnostic reference, preserve all neural candidates, and describe a surrogate
change accurately if that reference is ultimately adopted. Do not claim that all
four revised profiling models remain neural networks if USR power uses Task 2C.

This is a scientific tradeoff recommendation, not a formal threshold-based
rejection: no equivalence/noninferiority margin was prespecified. Neither model
is independently physically certified. Any further neural search or different
multi-objective selection would need a newly declared protocol and honest reuse
disclosure; the present search is closed. The plain networks show that future
work may be viable, but were not retrospectively promoted. More physical evidence
and QoS-threshold validation remain necessary before strong operational claims.
These experiments neither confirm nor refute the submitted PPO advantage.

New files are isolated under `revision/`: one configuration; neural training and
inference modules; comparison/plot diagnostics; tests; four model directories
(each containing `model.keras`, `manifest.json`, `history.csv`); and reports.
The selected candidate is linked by [frozen selection](reports/task2d/selection_frozen.json)
to its [manifest](models/task2d/original_hidden_softplus_seed_42/manifest.json).
Manifests record seeds, architecture, training rows, scaler, source/config/code
hashes, selected epochs and artifact hashes. The protocol records inference
dependencies and deterministic CPU settings. The
[file manifest](reports/task2d/file_manifest.json) lists all additions and hashes;
[integrity evidence](reports/task2d/execution_and_integrity.json) verifies prior files.

**Automated tests: 139 passed in 16.28 seconds** (124 previous + 15 new cases).
Coverage includes exact Task 2C groups, training-only scaling, validation selection,
no internal-test scores for losing candidates, retained hidden architecture,
nonnegative physical head with negative latent inputs, domain rejection, artifact
tampering, deterministic reloads and same-seed synthetic training. Keras emitted
28 NumPy array-copy deprecation warnings during the new tests; none were invalid
predictions or failed tests. [Test log](reports/task2d_tests.log),
[training log](reports/task2d_training.log), and
[comparison log](reports/task2d_comparisons.log) are preserved.

**Stopped after Task 2D.** Historical/Task 2C models, data and results, the other
three profiling models and the default twin remain unchanged. No PPO, recurrent
PPO or DQN training, traffic final-test outcome access, manuscript edit, commit or
push occurred. Work remains on `revision/comcom-clean-evaluation`.

# Task 2A — original fitted digital twin and validation-only static integration

Completed on `revision/comcom-clean-evaluation`, based on commit
`c0734715418d3f6d10785e5174aebb4bac24d0da`. The author's provisional approval of
the 60/15/25 split and causal interval-start decision convention applies here.
Task 1 files/configuration remain unchanged. This report supplements
[PROTOCOL.md](PROTOCOL.md); it does not change the manuscript or historical results.

## Implemented adapter

[FittedUPFTwin](evaluation/twin.py) implements the existing `Twin.outcome` contract.
It loads the original four Keras files, with their eight original input/target
scalers, using the pinned [artifact configuration](configs/twin_artifacts_v1.json).
It verifies all twelve SHA-256 hashes and serialization metadata **before** model
loading/unpickling. Missing/mismatched artifacts cause failure; there is no fallback
to polynomial models, another directory, a lookup grid, or a zero-valued prediction.

All artifacts reside in `saved_models/prediction_models/`:

| Mode/target | Model | Input scaler | Target scaler |
|---|---|---|---|
| DPDK QoS | `dpdk_upf_performance_model.keras` | `dpdk_perf_scaler_X.pkl` | `dpdk_perf_scaler_y.pkl` |
| DPDK power | `dpdk_upf_power_model.keras` | `dpdk_scaler_X.pkl` | `dpdk_scaler_y.pkl` |
| USR QoS | `oai_upf_performance_model.keras` | `oai_perf_scaler_X.pkl` | `oai_perf_scaler_y.pkl` |
| USR power | `oai_upf_power_model.keras` | `oai_scaler_X.pkl` | `oai_scaler_y.pkl` |

For DPDK, input load is the total interval load. For USR(k), input load is total
load/k, attributed power is k times predicted single-instance power, and QoS is
the predicted **per-instance** QoS. The only actions remain 0=DPDK, 1=USR(1),
2=USR(2). This reproduces the requested balanced-load approximation; it does not
validate load imbalance, interference, or physical multi-instance interactions.

Inference preserves the historical transform chain: cast per-instance Mbps to
float32, apply `log1p`, transform with the corresponding saved input scaler,
infer the corresponding network, and invert the saved target scaler. The float32
model input and the original per-instance load are both retained in diagnostics.
The adapter performs no fitting or parameter updates.

## Inference and compatibility profile

The [execution manifest](reports/task2a/validation/artifact_manifest.json) records
absolute and relative artifact paths, bytes, hashes, serializer metadata, adapter
hash, twin identity, runtime versions, TensorFlow build, and compatibility warnings.
[Host details](reports/task2a/inference_host.json) and the
[existing environment package inventory](reports/task2a/environment_packages.json)
record hardware and transitive package versions. The package inventory is not an
instruction to install every development package.

- Saved model serializer: Keras **3.11.3** (30 September 2025).
- Saved scaler serializer: scikit-learn **1.7.2**.
- Active profile: Python **3.12.3**, TensorFlow **2.20.0**, Keras **3.13.2**, NumPy
  **2.4.2**, scikit-learn **1.8.0**, joblib **1.5.3**, h5py **3.15.1**.
- CPU only; oneDNN disabled; deterministic TensorFlow operations; intra/inter-op
  and OMP threads fixed at one; TF32 disabled.
- Models load with `compile=False` and execute with `training=False`. Dropout is
  inactive and batch-normalization statistics are not updated.
- Each forward pass has exactly one sample. This avoids caller-batch-dependent
  arithmetic. An exact float32 input cache improves repeated evaluation without
  changing the forward-pass policy. Cache-cleared, reloaded-model, and independent
  process checks all agree exactly on the tested inputs.

All eight cross-version scaler warnings are retained in the manifest. The exact
TensorFlow version that originally serialized these model artifacts is not in
their `.keras` metadata; it remains explicitly unknown. Using this tested profile
does not establish compatibility with arbitrary library/hardware versions. A
dependency/backend change requires a new identity and verification.

The adapter must initialize TensorFlow's inference profile before another module
imports TensorFlow; otherwise it fails with instructions to use a fresh process.
It does not import the historical `src.utils` runner or change `.venv`. The CUDA
build may log a failed GPU initialization when GPU visibility is deliberately
disabled; the production manifest confirms that only CPU is visible.

## Invalid predictions

`raw_prediction` retains unaltered numerical outputs for diagnostics. `outcome`
rejects nonfinite/negative attributed power and QoS outside [0,1] before accounting.
The exception identifies the twin, action, total/per-instance load, actual outputs,
and failed checks. Validation preflight records every invalid slot; a failed
configuration receives no partial/"repaired" energy total. Nonfinite failures are
encoded explicitly in JSON diagnostics, not converted to physical output values.

The original USR power surrogate produces negative values at low load:

- At 2 Mbps per instance, current fixed-profile power is **−0.0218821764 W**.
- At 4 Mbps per instance, it is **−0.0535320044 W**.
- In the profiling reference CSV, **100 of 913** USR power rows have negative
  predictions, covering these two unique offered-load values. The same rows are
  negative in the new CPU probe. This is already present in archived predictions,
  not merely a new cross-version warning.

The power network has an unconstrained linear output head; nonnegative physical
power is not guaranteed by that architecture. These loads lie within the recovered
profiling range. The negative magnitudes greatly exceed the numerical differences
studied below. No clipping, replacement by DPDK, extrapolation restriction, model
repair, or retraining was introduced. Further use at low load requires an explicit
scientific handling/model-validation decision.

## Numerical investigation

Reference: `profiling_model_predictions_vs_target.csv`, using **identical input
loads and sample identifiers**, not historical traffic-controller outcomes. The
comparison covers 919 rows per DPDK target and 913 per USR target, with 22 unique
offered loads per mode. Each profile records the same model/scaler hashes and
reference-file hash. No forecast/controller selection uses these comparisons.

The canonical summary is
[controlled_comparisons.json](reports/task2a/numerics/controlled_comparisons.json).
Per-profile CSVs retain every archived/fresh pair and difference. An earlier
analysis snapshot, `numerical_findings.json`, is also retained for traceability.

| Target | Maximum absolute difference, fixed CPU scalar vs archive | Exact matches |
|---|---:|---:|
| DPDK attributed power | 0 W | 919/919 |
| DPDK QoS | 1.68085098e−5 | 0/919 |
| USR per-instance power | 3.10063362e−4 W | 0/913 |
| USR QoS | 1.06036663e−4 | 0/913 |

These are measured differences, not a "close enough" acceptance criterion. New
replays on the same canonical profile are required to be exactly repeatable;
historical numerical reproduction is not claimed.

Controlled comparisons against current CPU `model.predict(batch_size=32)` found:

1. **Batch size/call path:** single-sample direct calls change USR power by at
   most 4.76837158e−7 W. Batch vs single-sample `log1p` inputs match exactly here.
2. **oneDNN:** enabling it changes USR power by at most 3.57627869e−7 W.
3. **scikit-learn serializer match:** temporarily using 1.7.2 changes USR power by
   at most 4.76837158e−7 W, DPDK power by 5.96046448e−8 W, and USR QoS by
   1.49011612e−7. Source inspection explains this small difference: 1.8 explicitly
   casts saved mean/scale coefficients to input dtype before arithmetic; 1.7.2
   uses the saved coefficients in the in-place operations. The 1.7.2 probe agrees
   bitwise with that explicit saved-statistics formula. This establishes a small
   version effect; it does not explain the larger archival discrepancy.
4. **Keras serializer match:** Keras 3.11.3 gives identical profiling predictions
   to 3.13.2 in the controlled CPU batch comparison. Combining old Keras with old
   sklearn still leaves the larger gap.
5. **GPU precision control:** the host lists a GRID A100D-40C with driver
   580.82.07, but TensorFlow fails during loading with
   `cudaSetDevice() on GPU:0 failed. Status: operation not supported`.
   [The failure log](reports/task2a/numerics/gpu_predict32_tf32_on.log) is preserved.
   No GPU predictions or CPU fallback results are represented as GPU controls;
   TF32-off and GPU-scalar comparisons were not executed after that failure.

The original inference TensorFlow/backend/precision manifest is unavailable.
**The larger historical discrepancy remains unresolved.** The investigated CPU
kernel and serializer-version effects are too small to account for it. GPU/TF32
is an untested candidate, not an established cause. The new deterministic CPU
profile was defined before comparison and was not selected to match the archive.

Compatibility wheels were downloaded from the existing pip cache and unpacked
only under `/tmp/comcom-task2a-compat*`, using process-local `PYTHONPATH`; no package
was installed/upgraded in `.venv`. Their exact filenames, hashes and variant paths
are in [compatibility_wheels.json](reports/task2a/numerics/compatibility_wheels.json).
The full historical dependency environment has not been recovered by matching
these two serializer versions alone.

## Validation-only static results

[StaticController](evaluation/static.py) emits a fixed command through the shared
actuation guard. It does not bypass initial state or cooldown. The
[validation runner](evaluation/validate_static.py) refuses training/final-test
roles before any inference and exposes no final-test execution option.

Dataset: Netflix/Main. Scored indices `[2016,2520)`, **504 slots**, 22 April 2019
00:00 through 27 April 05:45, integrating through 27 April 06:00 (**126 h**).
All three configurations use the same horizon fingerprint and workloads, ranging
from 14.43416925 to 326.98927859 Mbps. USR(2) per-instance loads range from
7.21708462 to 163.49463930 Mbps. All predictions on this specific validation
domain pass the outcome checks. YouTube is not evaluated.

| Configuration | Attributed energy (Wh) | Mean SEC (J/Mbit) | QoS violation rate | Type switches | Scaling events | USR instances launched | Blocked |
|---|---:|---:|---:|---:|---:|---:|---:|
| DPDK | 103.344020799 | 0.0102870065 | 0 | 0 | 0 | 0 | 0 |
| USR(1) | 150.060348645 | 0.0090111751 | 0 | 1 | 0 | 1 | 0 |
| USR(2) | 151.085224330 | 0.0091988299 | 0 | 1 | 0 | 2 | 0 |

These are new validation diagnostics, not replacement paper results or a ranking
of trained controllers. All three start in DPDK with elapsed cooldown=4. USR
commands execute at the first scored slot; they incur one type switch and 0.012
or 0.024 launch reward cost. DPDK→USR(2) is not also counted as a scale-up. The
next cooldown counts are 1,2,3,4,4 under unchanged commands; no command is blocked.

Power is attributed W, integrated as `sum(P * 900 / 3600)` Wh. Launch coefficients
are reward costs and do not add physical startup Wh. SEC is J/Mbit; mean slot
ratios differ from the aggregate energy/volume ratio. Summary JSON also records
the latter, rewards, QoS ranges/shortfalls, candidate identities and twin identity.
Every repeated static rollout agrees exactly in actions, outcomes and accounting.

Evidence:

- [summary.json](reports/task2a/validation/summary.json): full metrics and timeline provenance.
- [predictions.csv](reports/task2a/validation/predictions.csv): raw per-instance/total outcomes and validity checks.
- [steps.csv](reports/task2a/validation/steps.csv): requested/executed actions, counters, events, reward terms and units.
- [artifact_manifest.json](reports/task2a/validation/artifact_manifest.json): exact artifacts and inference environment.

## Tests and reproduction

[test_real_twin.py](tests/test_real_twin.py) adds real artifact/serializer identity,
loading/transform consistency, balanced-load semantics, uncached/reloaded/fresh-
process determinism, unchanged model weights, strict invalid-output failures,
static state/cooldown/event accounting, common validation horizons and energy
integration. It also verifies that the Task 2A runner rejects other partitions
before any model query and that invalid preflight cannot yield partial totals.

Together with the Task 1 suite, **94 tests pass**. The captured final command and
output are in [test_results.txt](reports/task2a/test_results.txt). The legacy full
test suite was not run. Synthetic fixtures still test final-test access guards;
no real final-test controller rollout was executed.

From repository root, using the existing environment:

```bash
.venv/bin/python -m pytest revision/tests -q -p no:cacheprovider
.venv/bin/python -m revision.evaluation.validate_static \
  --output-dir revision/reports/task2a-new-validation
.venv/bin/python -m revision.evaluation.numerical_probe \
  --profile cpu_scalar --output-dir revision/reports/task2a-new-cpu-probe
```

Outputs require new paths under `revision/`; existing results are not overwritten.
Other probe profiles are listed in `numerical_probe.py`. For version controls,
download the exact wheels named in `compatibility_wheels.json` with `pip download
--no-deps --only-binary=:all:`, verify their hashes, unpack into separate temporary
directories, and prepend only the selected directories to `PYTHONPATH` when
running the same probe. `summarize_numerics.py` checks input/reference/artifact
identity before comparing completed profile outputs. A GPU failure is a diagnostic
limitation and must remain visible.

## Readiness and remaining dependencies

**Ready for deterministic controller-interface integration on the validated
Netflix validation domain. Not yet cleared for an unrestricted training campaign.**
The adapter and static interfaces are implemented and verified, and known-invalid
outputs fail explicitly. That is narrower than validating every action over every
future or training load. The low-load USR issue must be resolved through an
explicit scientific decision before such loads can safely enter policy learning.
No load-dependent action mask or surrogate repair is hidden in this adapter.

Other remaining issues: unexplained historical numerical differences; missing
original backend/TensorFlow provenance; cross-version scaler compatibility;
unverified physical QoS labels/calibration and balanced multi-instance behavior;
and the Task 1 forecasting, observation-vector/recurrent-state and checkpoint-
normalizer dependencies. No PPO/recurrent PPO/DQN, rule rewrite, model retraining,
reward sweep, manuscript edit, final-test rollout, commit, or push was performed.

The [file/integrity manifest](reports/task2a/file_manifest.json) enumerates created
Task 2A files and their hashes, including diagnostics. All pre-existing Task 1
files and historical tracked files were verified unchanged. Task 2A stops here.

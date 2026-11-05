# Corrected evaluation foundation — COMCOM-D-26-01061, Task 1

Status: implemented and tested foundation; proposed experimental design; no corrected
controller campaign has run. All additions are under `revision/`. Base commit:
`c0734715418d3f6d10785e5174aebb4bac24d0da`, on
`revision/comcom-clean-evaluation`. Historical entry points and artifacts remain
available at their existing paths and in the preservation references.

Authority: the [submitted manuscript](../Paper%20artifacts/manuscript_energyawareupf.tex),
the [scientific audit](../docs/audits/2026-10-22/SCIENTIFIC_REPOSITORY_AUDIT.md), and its
[submitted-source addendum](../docs/audits/2026-10-22/submitted-manuscript-reconciliation/AUDIT_ADDENDUM.md).
The changes below are explicit departures from the historical implementation.
They are proposed because causality, validation-only selection, and common
timestamps cannot be obtained by retaining the historical loader unchanged.

## 1. Actual inputs and provenance

All five files under `data/processed/` have 3,360 unique, ordered quarter-hour
timestamps, starting 2019-04-01 00:00 and ending 2019-05-05 23:45. There are two
underlying workloads: Netflix/Main and YouTube. Files bearing `bestpredictor` or
`with_forecast` are alternative forecasts of those same workloads, not independent
datasets. The [timeline manifest](reports/timeline_manifest.json) records all five
file hashes and hashes of their timestamps/pre-rescaling loads.

Preparation reads `timestamp` and `Traffic_bps` from these canonical files:

- Netflix: `data/processed/processed_traffic_normalized.csv`.
- YouTube: `data/processed/processed_traffic_youtube_aggregated_scaled.csv`.

`Traffic_bps / 1e6` is the pre-rescaling city-level aggregate in Mbps. The original
spatial aggregation inputs and measurement manifest are still unavailable. This
is not a reconstruction of those missing inputs. Existing `Traffic_Mbps_scaled`,
`Traffic_Predicted_Mbps`, and uncertainty columns are not consumed by preparation.
The 96 leading forecast NaNs therefore do not dictate the new split.

The loader fails on missing/negative/nonfinite loads, duplicate/out-of-order
timestamps, and gaps. It does not sort, interpolate, backfill, resample, or silently
drop rows. Source times are retained as naive source-clock values. Their timezone
and original interval-label convention are not authenticated; see Section 9.

## 2. Proposed chronological split

Boundaries in [evaluation_v1.json](configs/evaluation_v1.json) are explicit timestamps,
with intervals half open at the right boundary. No fractions are applied after
trimming, and no controller may choose its own slice.

| Role | Source indices `[start, stop)` | Labels: first through last interval start | Label rows | Scored decisions |
|---|---|---|---:|---:|
| Training | `[0, 2016)` | Apr 1 00:00–Apr 21 23:45 | 2,016 | 1,920 |
| Validation | `[2016, 2520)` | Apr 22 00:00–Apr 27 05:45 | 504 | 504 |
| Final test | `[2520, 3360)` | Apr 27 06:00–May 5 23:45 | 840 | 840 |

End of the final integration interval: May 6 00:00. Scored validation and final
durations are 126 h and 210 h respectively; scored training spans 480 h.

Rationale: preserve the submitted study's intended final 25%/840-slot suffix and
carve validation from its former development prefix. The proposed 60/15/25 split
leaves three full weeks for learning and more than five days for validation. It
maintains substantial variation for selection and scaling comparisons. Validation
does not cover a complete week; 60/20/20 (21/7/7 days) is a defensible alternative
if full weekly validation is prioritized. It changes the final suffix to 672 rows.
Rolling-origin validation is another option, requiring a separate prespecified
tuning/compute budget. No comparison of outcomes was used to choose these dates.
The supplied config is the concrete Task 1 proposal, not evidence that its sample
size or statistical power is sufficient for every reviewer experiment.

Only training labels may fit model parameters and preprocessing statistics.
Validation may select hyperparameters, forecast candidates, observation variants,
rule thresholds, and paired best-checkpoint/normalizer bundles. Final outcomes may
only be reported after those choices are frozen. The candidate list and primary
endpoint must be specified before a campaign. The current proposal selects by
**total validation reward**, preserving the historical checkpoint-selection
criterion while changing its data source. Discounted return is also reported;
using it as the selection objective is an explicit configuration/protocol change.
The low-level selection API additionally supports negative energy for explicitly
declared future protocols; objectives cannot be mixed within one selection.

Final-test data in this repository have already appeared in historical results
and the audit. These APIs prevent reuse for automated selection in the new
campaign; they cannot make the period prospectively unseen again. A genuinely
unseen confirmatory period/dataset remains desirable. Inspection of input dates,
counts, and data integrity is not controller-performance selection. Do not adapt
the model or protocol after inspecting new final-test performance.

YouTube is transfer reporting only. It cannot fit transforms, fit forecast
parameters, select a checkpoint, or tune a rule. It uses the same final 840
timestamps as Netflix. Its earlier observations supply causal context, not a
second development dataset. This is same-city, same-period, cross-service transfer.

## 3. Scaling and warmup

Two separate transforms have explicit training provenance:

1. Workload design: `LoadScale` maps the Netflix training maximum in raw aggregate
   Mbps to 400 Mbps, without subtracting the minimum. The frozen ratio applies to
   both workloads and every partition. This preserves the paper's ratio-scaling
   intent without fitting a whole-series peak or a separate transfer scale.
2. Input normalization: `TrainingMinMax` uses the scaled Netflix training minimum
   and range. Forecast/policy vector adapters must reuse it unchanged for validation
   and final test. No normalization statistics are updated during evaluation.

In these files the Netflix training peak is also the full-series peak, so the
fitted multiplier is approximately 0.1935859071, matching the historical Main
ratio. Historical YouTube uses the rounded 0.1936 multiplier; the new proposal
uses the exact Netflix training ratio. Neither transform clips validation or test
values. The manifest reports any demand above the reference 400 Mbps. A reference
peak is a study design choice, not proof of a physically validated capacity limit.
An out-of-range future workload needs an explicit scope/validation decision.

History length is explicitly **95 completed intervals**, preserving the archived
policy setting. Forecast input length is **96 completed intervals**, preserving
the predictor setting. They are not presented as the same window. Their maximum
(96 intervals) defines one common warmup for every controller and schema. Thus
training decisions begin at source row 96 (Apr 2 00:00), even for a static controller.

Validation/test reuse preceding, already completed observations as context and
score every label in their partitions. There is no second split-local warmup,
no reward/event accounting for context rows, and no overlapping outcome labels.
Within the final test, earlier completed final-test observations may inform later
decisions, as they would during online replay. Model parameters remain frozen.
A requested window exceeding available pre-partition context fails rather than
shrinking that partition's comparison horizon.

## 4. Causal decision and forecast indexing

The operational convention is explicit: a CSV row at `s_i` represents the mean
workload over `[s_i, s_i + 15 min)`. The action for that interval is issued at
`s_i`. At that epoch, observations contain only loads through row `i-1`, whose
interval has just ended. Actual row `i` is supplied only to the twin for outcomes.

| Quantity at decision `s_i` | Source |
|---|---|
| Latest measured load | row `i-1` |
| Policy history (95) | rows `[i-95, i)` |
| Predictor history (96) | rows `[i-96, i)` |
| Forecast target | row `i`, interval starting `s_i` |
| Outcome/reward | executed configuration at actual row `i` |
| Next decision | `s_i + 15 min` |

This is a one-step forecast relative to the latest completed interval. If that
completed interval is denoted `t`, the forecast is the manuscript's `t+1`.
It replaces the historical same-slot actual-load decision with a causal delayed-
telemetry interpretation; it does not silently claim they have equivalent
information. A perfectly observed at-start demand scenario could be studied
separately, but would need evidence about telemetry availability and a new contract.

`DecisionContext` contains only past traffic. The forecaster interface receives
that context, never upcoming actual load or a full future array. It must return
issuance, observation-cutoff, and target timestamps; all are checked. Learned
forecasters must declare matching training provenance with a last training label
available before the decision. Archived forecast CSVs fail to establish that
lineage and are not imported as corrected forecasts.

`PersistenceForecast` is a parameter-free test fixture and reference interface
example, not a selected replacement for the submitted LSTM. No forecast training
or forecast selection has been performed. Forecasts are recorded before taking
the action, preventing the historical post-step logging shift and final auto-reset
contamination. The last source row is scored without clamping to a later forecast
or requiring a nonexistent next observation.

Calendar features are computed directly from the timestamps they describe:
completed-load calendar at `s_(i-1)`, decision/forecast-target calendar at `s_i`.
The context/observation retain these names separately. No untrimmed calendar array
is sliced using trimmed-traffic offsets. Daily sine/cosine uses time of day, and
weekly sine/cosine uses weekday; no calendar scaler is fitted on future timestamps.

## 5. Shared evaluation interface

`EvaluationHorizon` identifies workload, preprocessing-dependent values, role,
source indices, timestamps, and window sizes by a deterministic fingerprint.
There is no controller-type argument that changes those indices. `evaluate` is a
generic runner for future adapters; it implements no PPO, recurrent PPO, rule,
or static policy and loads no historical twin/checkpoint.
`require_comparable` rejects result collections with different workload/horizon
fingerprints, actuation/reward contracts, or missing scored slots before tabulation.

- Actions are exactly `(0, 1, 2)` = DPDK, USR(1), USR(2), for every controller.
- Each horizon starts in DPDK with cooldown eligible for change. Context rows do
  not simulate prior actions. If a controller requests USR in the first scored
  slot, the executed switch is counted and its reward cost charged. A static
  native-initialization study would be a distinct contract, not silently mixed in.
- Instantaneous observations include the most recently completed load, previous
  evaluated QoS/power outcome, deployment state, cooldown, and named calendar
  fields. Previous QoS/power are absent at reset; a future vector adapter must
  provide a missing-value mask rather than query the upcoming outcome. History
  and forecast schemas augment these signals as the submitted description states.
- Only history/hybrid schemas receive the policy history. Only forecast/hybrid
  schemas receive the causal forecast. The controller cannot reach the full
  preparation context through the observation object.
- A controller adapter resets once per horizon with an explicit seed. Each call
  uses `deterministic=True`. A recurrent adapter must retain hidden state between
  calls, set episode-start only at reset, and clear it between horizons. Those
  adapters are intentionally outside Task 1 and have not been tested here.
- A twin adapter implements deterministic `outcome(action, load_mbps) -> Outcome`.
  This task tests it with synthetic outcomes only. No fitted profiling model is
  loaded or modified. The fitted-model versions, normalization pairing, inference
  backend, and real determinism still need adapter-level verification.

## 6. Cooldown, event counts, reward, and units

The cooldown state is the elapsed-slot count **at actuation time**, also exposed
to the controller. It is checked without a hidden increment. After a change at
`t`, the next observation at `t+1` has elapsed=1. With cooldown=4, changes at `t+1`,
`t+2`, `t+3` are blocked and a fresh change may execute at `t+4`. Counts saturate
at 4 when eligible. Unchanged commands do not reset the timer. Blocked commands
are discarded, incur no change cost, and are never queued. This retains the
historical executed dwell schedule while removing the observation/guard tick
ambiguity. Cooldown=0 permits changes in consecutive slots.

Events are computed from executed configurations:

- Type switch: DPDK ↔ USR(k).
- Scale-up/down event: USR(1) ↔ USR(2), with mode unchanged.
- Total reconfigurations = type switches + scale-up events + scale-down events.
- Launched USR instances = `k` on DPDK→USR(k), or added replicas within USR.
- Blocked requests are counted separately, never as executed events.

The accounting helper implements the **submitted** cost equation: 0.030 for a
switch to DPDK; 0.012 per launched USR instance, including a mode switch; 0.003 per
within-USR scale-down event. QoS target=0.90, alpha=100, QoS weight=30, epsilon=1e-6
Mbps, gamma=0.995. These values are recorded in the new configuration only.
Reward is minus alpha×SEC, QoS shortfall penalty, and executed orchestration cost.
This corrects the historical flat 0.030 cost for both switching directions.

Energy uses interval-mean attributed power:
`energy_Wh = power_W * slot_seconds / 3600`. Offered volume is
`load_Mbps * slot_seconds` in Mbit. Per-slot SEC is `power_W / max(load_Mbps, epsilon)`
in **J/Mbit**, numerically W/Mbps. Dividing by 3600 converts it to Wh/Mbit. Both
arithmetic mean slot SEC and aggregate `sum(energy_Wh)*3600/sum(offered_Mbit)` are
reported with distinct names; the latter is undefined (`None`) for zero volume.
The offered volume does not establish successfully delivered traffic.

A QoS violation is strictly `Q < 0.90`; equality is compliant. Metrics also include
mean shortfall. `Outcome` rejects nonfinite/negative power and QoS outside [0,1]
instead of silently clipping a surrogate prediction. This is an interface
validity check, not a new calibration of the models. Rewards and events remain
heuristic accounting; startup delay, transient power, live session transfer,
interruption, and physical safety overrides are not modeled by this foundation.

## 7. Selection and final-reporting gate

`Candidate` stores named SHA-256 artifact identities. Future learned-controller
adapters must include and verify the actual checkpoint, its checkpoint-time
normalizer, forecast model/transform, controller configuration, and twin bundle.
Final-training normalization files cannot stand in for checkpoint-paired files.
The hash carrier is implemented here; actual artifact-loading adapters are not.

`ValidationScore.from_result` rejects training, final-test, transfer, partial-
horizon, and wrong-candidate results. `freeze_selection` requires the same complete
validation horizon, objective, and reward/actuation contract for the candidate
set. It resolves exact score ties by candidate identity, without test outcomes.
Separate algorithm families/seeds may each freeze their validation winner; the
API is not an instruction to discard all but one family from a comparison.

Final evaluation is rejected before controller reset unless a frozen selection
authorizes the exact candidate, final horizon, contract, and observation schema.
Optional YouTube final horizons must be declared at freezing time, match source
final timestamps/window requirements, and cannot contribute selection scores.
This gate does not enforce a one-time test read: repeat deterministic reporting
is allowed, but later tuning based on any final result invalidates its held-out
interpretation. Retuning requires a new protocol and independent final evidence.

These are checks against accidental data misuse, not a security sandbox. Python
cannot prevent a user from fabricating metadata, reading raw final files, or
passing a model that secretly accesses global arrays. Artifact provenance,
adapter checks, frozen configurations, and scientific discipline remain required.

## 8. Historical differences and execution

| Historical path | Corrected foundation |
|---|---|
| Trim forecast NaNs first, then 75/25 split | Split full timestamp timeline first |
| 720/721/840 incompatible or shifted horizons | One common 840-row final suffix |
| 94-row split-local history loss, omitted final row | Pre-partition context; final row included |
| No independent validation; test selects best checkpoint | Validation-only selection, frozen final access |
| Whole-series scalers / old forecast CSV lineage | Training-only transforms; causal forecast interface |
| Actual slot-t load used for slot-t actuation | Completed telemetry used for upcoming interval |
| Calendar lag and post-step forecast logging | Named timestamp-derived calendars; pre-action forecast |
| Forecast schema drops instantaneous signals | Every schema retains instantaneous signals |
| Recurrent evaluator discards hidden state | Adapter lifecycle contract; implementation deferred |
| Rule compiler restricted to DPDK/USR(1) | Common three-action interface; rule rewrite deferred |
| Hidden cooldown increment before guard | Exposed actuation-time counter, same t+4 dwell schedule |
| Flat mode-switch cost | Submitted direction/launch-specific cost accounting |
| SEC values labeled Wh/Mbit | Explicit J/Mbit and separately integrated Wh |

From repository root, using the existing environment:

```bash
.venv/bin/python -m pytest revision/tests -q -p no:cacheprovider
.venv/bin/python -m revision.evaluation.inspect_timeline
```

The inspection command defaults to stdout. `--output revision/reports/<new>.json`
uses exclusive file creation and refuses destinations outside `revision/`.
It emits input/preparation metadata, not policy performance. No new workload CSV
or historical result file is written. Modules use the Python standard library;
only test execution requires pytest. See [test_results.txt](reports/test_results.txt)
for the captured test run. Existing legacy tests were not used as validation of
this separate foundation; no training, new seeds, DQN, reward sweep, or paper edits
were performed.

## 9. Decisions/dependencies before controller modifications

1. **Approve/freeze the proposed split and causal timing convention.** Alternatives
   are documented above. Authenticate whether source timestamps denote interval
   starts or ends, and recover timezone provenance. Until then the code uses the
   explicit source-clock interval-start convention, not an asserted historical
   fact. A verified end-labeled source needs a documented conversion before use.
2. **Select a causal forecasting training strategy.** A predictor fitted to the
   entire training prefix is valid for subsequent validation/test inference but
   leaks future training labels into early training decision inputs. The runner
   rejects that use. Options include walk-forward/out-of-fold forecasts within
   training or a nested forecaster-fit prefix followed by RL-training data. The
   latter reduces learning data and changes the common warmup. Define the approach
   and validation-based predictor selection before building learned controllers.
   Persistence remains a fixture, not an approved scientific LSTM substitute.
3. **Finalize observation-vector adapters and normalization.** Specify ordering,
   units, first-outcome masks, recurrent episode-start/state handling, and any
   feature ablations. Decide whether 95 versus 96 policy history needs a planned
   change. Check actual loaded artifact hashes and save each selected checkpoint
   together with its own normalization state; do not reuse the leaked historical
   policies as independent evidence under this new information set.
4. **Validate the twin and physical claims.** Recover QoS label/campaign provenance,
   resolve threshold residuals and numerical inference/version differences, and
   decide how to handle genuinely out-of-range surrogate outputs. Independent
   multi-instance behavior is not established by additive surrogate power. No
   load clipping, QoS repair, or new physical actuator is silently introduced.
5. **Specify and validate the fair rule/static adapters.** All get the same action
   set, timestamps, telemetry and guard. Tune rule parameters on validation only.
   Preserve initial DPDK/eligible state or declare any alternative as a separate
   contract. Do not infer rule quality from interface tests with synthetic doubles.
6. **Prespecify the later campaign and statistics.** Freeze algorithm/seed-specific
   candidate sets, tuning budget, primary/secondary endpoints and comparisons,
   sample-size/precision plan, and hardware/library manifests. No such campaign
   is started by this task. Newly reserved final slots are conditional evidence
   from a previously examined trace; obtain new confirmatory data if needed for
   stronger claims. Keep YouTube out of tuning.

Task 1 ends here. Subsequent controller adapters, corrected-model training, and
replacement scientific results require a separate authorized task.

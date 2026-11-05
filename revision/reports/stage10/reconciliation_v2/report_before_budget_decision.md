# Stage 10 — material scientific blocker; campaign not executed

**Status: incomplete.** The 40-run replacement campaign and final test were not
started. The service-contract prerequisite cannot currently be frozen without
assigning unsupported physical meanings or inventing service-policy requirements.
This is the material-scientific-blocker exception in the Stage 10 instruction,
not a claim that a design report completes the requested campaign.

Work remains on `revision/comcom-clean-evaluation`, HEAD
`c0734715418d3f6d10785e5174aebb4bac24d0da`. All Stages 2–9 are preserved.
No manuscript edit, commit or push. No model fitting, controller training,
forecast refit, controller replay, traffic final-test target access or scoring.

## 1. Sources recovered and reconciled

The thesis source was found locally, with the matching title in
[`titleAndInfos.tex`](/home/ubuntu/UpfThesisPipeline/thesis/titleAndInfos.tex).
Its [digital-twin chapter](/home/ubuntu/UpfThesisPipeline/thesis/chapters/twin/twin.tex)
contains the non-compensatory safety and activation model. Its manifest points
to the separate `UpfProfilingCampaign`, `UPF_NDT` and controller repositories.
Two MAPPO manuscript candidates were found:

- [Networking Letters source](/home/ubuntu/UpfRLControllers/reports/paper-letters/paper.tex):
  “Cooperative Multi-Agent Reinforcement Learning for Energy-Aware UPF Realisation
  Switching at the 5G Edge.” It expresses switching energy using loaded incoming
  power times activation duration, integrated into effective SEC.
- [MASCOTS source](/home/ubuntu/UpfRLControllers/reports/paper-mascots/paper.tex):
  “A Measurement-Grounded Digital Twin for Evaluating …”. It uses separate switching
  reward costs and a minimum delay/loss score. The local title/version is a
  candidate reference; its identity as the submitted MAPPO version is unverified.

These versions do not define identical rewards. The Stage 10 instruction explicitly
chooses the thesis burst-ratio model, so that instruction can take precedence for
activation accounting. Neither MAPPO candidate supplies the requested complete
four-KPI COMCOM service contract. Source identities and hashes are recorded in
[reference_manifest.json](reports/stage10/reference_manifest.json) and
[source_identity_complete.json](reports/stage10/source_identity_complete.json).
No remote source or unverifiable publication identity was substituted.

## 2. The blocking dependency is physical target and service-policy identity

| KPI | Confirmed available evidence | Missing prerequisite for the definitive campaign |
|---|---|---|
| Throughput satisfaction | COMCOM stores DN/RAN receive/transmit Mbps and a configured per-direction offered objective. Raw receive-counter units were authenticated previously. The separate thesis pipeline also derives its throughput from a DN receive counter. | Tester flow/endpoint mapping establishing what is delivered in which direction, and whether its denominator is configured per-direction or measured aggregate offered traffic. A DN Rx proxy must not silently become delivered downlink. A minimum satisfaction policy is not specified. |
| Downlink delay | COMCOM `downlink_delay_ms` has documented millisecond units. The separate thesis models `downlink_one_way_delay_distribution__weighted_mean_delay_us`. | Measurement scope/aggregation equivalence is unestablished. The thesis explicitly distinguishes its 200 µs UPF-attributed budget from end-to-end measurements. Unit conversion alone cannot establish equivalence. Specify the budget for the actual COMCOM delay target. |
| Jitter | COMCOM `RAN_jitter` matches raw `Avg us` exports; microseconds are supported. | No jitter service budget is supplied by the two-KPI thesis/MAPPO contract. The historical 10 µs value is a score-saturation cap, not an authenticated service requirement. |
| Packet loss | Cumulative RAN loss counters and within-case nonnegative increments have substantial raw provenance; processed `RAN_PLR_interval` is a stored fraction. | Prior Task 2G could not authenticate its denominator/reset/aggregation construction. The thesis constrains **DN loss counts per 3-second sample**, whereas the old COMCOM score uses a **RAN fraction**. They are different quantities. Decide and authenticate the target, direction, sample duration and budget; do not convert counts to a ratio without a denominator. |

The [Task 2G evidence](TASK2G_REPORT.md) and
[historical score specification](specifications/qos_score_v1.json) explicitly
identify the old 40 ms / 10 µs / 5% constants as score caps. The Stage 10
authorization supersedes the old score for the future campaign; it does not
turn those caps into physical service budgets.

The upstream profiling code was also inspected:
[`features.py`](/home/ubuntu/UpfProfilingCampaign/UpfProfilingCampaign/src/features.py)
derives DN receive throughput, while
[`upf_profile.py`](/home/ubuntu/UPF_NDT/src/upf_digital_twin/twin/upf_profile.py)
uses DN loss increments and a distribution-derived delay target. This is useful
candidate lineage, but does not authenticate a row-level equivalence to the
existing COMCOM CSV fields. No alternate profiling dataset was silently merged
or substituted.

### Why this materially changes the experiment

Training-only marginal counts from the unchanged COMCOM profiling sweeps 1–3:

| Illustrative comparison, not adopted policy | DPDK / 2,760 | USR / 2,736 |
|---|---:|---:|
| Stored delay >0.2 ms, numerically equal to 200 µs | 170 | 916 |
| Stored delay >40 ms, old score cap | 42 | 461 |
| Stored RAN loss increment >5 packets | 43 | 824 |
| Stored RAN loss fraction >5% | 41 | 741 |
| Stored jitter >10 µs, old score cap | 15 | 1,022 |

These are **different marginal labelings**, not comparable complete policies,
an assertion that the thesis budget applies, or a choice of thresholds. They
demonstrate why copying numbers or confusing units/targets would change model
selection, false-safe rates, reward calibration and learned behavior. Complete
training-only field statistics, hashes and denominators are in
[training_field_evidence.json](reports/stage10/training_field_evidence.json).

An empirical quantile of training jitter is not itself a service requirement.
Nor does user approval of a proxy establish its physical meaning. A deliberately
proxy-scoped study is possible only if named as such with explicit study-policy
budgets; it cannot be presented as authenticated delivered-throughput/loss
compliance. No tolerance or physical interpretation was invented here.

## 3. Activation semantics recovered; portability remains explicit

The thesis supplies:

- DPDK activation 24 s; USR activation 3.3 s; 900 s control intervals.
- Burst ratios 1.73 / .37, **ratios rather than increments**.
- `E_activation_Wh = k × P_idle_W × activation_seconds / 3600` per launch.
- Outgoing traffic service during activation; incoming service afterwards.
- Safety must consider both configurations when both actually serve.

Its illustrative idle powers .82 / .015 W give approximately .00945733 Wh
per DPDK launch and .0000050875 Wh per USR launch. These are thesis-model
calculations, **not newly measured COMCOM activation energies or idle powers**.
The original COMCOM measurements have no authenticated true-idle experiment.
Applying those idle values to COMCOM would need a clearly declared portability
assumption; using a loaded prediction as idle would not implement this model.

For USR(1)→USR(2), a future implementation can explicitly model one added replica
while the existing replica serves until it becomes active. DPDK→USR(2) also
requires an explicit parallel/sequential launch convention. The thesis's binary
realisation model does not measure replica draining. An instantaneous
USR(2)→USR(1) drain can be a declared modeling assumption, not a measured fact.
No draining duration or outage was fabricated, and no partial transition model
was integrated into controller evaluations.

## 4. Safe implementation completed without setting unknown policy

[stage10/primitives.py](stage10/primitives.py) implements unintegrated primitives:

- Explicit budget/KPI types with declared loss units and sample duration for
  count-based loss. There are **no default service budgets**.
- Conjunctive strict compliance: throughput ≥its lower budget and all three
  other KPIs ≤their upper budgets. Equality is feasible.
- Proposed continuous severity: maximum of normalized throughput shortfall,
  delay exceedance, jitter exceedance and loss exceedance. Good KPIs cannot
  compensate for a violation. No weighted-score .90 threshold is reused.
- Time-weighted operational Wh plus separately logged burst energy once, with
  explicit input parameters. No SEC reward or duplicate activation penalty.
- Worst serving-phase compliance/severity: a safe incoming configuration cannot
  conceal an unsafe outgoing interval. No invented service interruption.

Zero-loss policies require a separately specified positive severity normalization
scale; the current primitive rejects division by a zero budget rather than
inventing epsilon. Other incompatible units/nonfinite inputs also fail explicitly.
The arithmetic is reusable but **is not the replacement environment**, a selected
reward, a validated model, or an approved service contract. The specimen budgets
in unit tests are synthetic fixtures only.

**13 tests passed in 0.02 seconds** in
[final_tests.log](reports/stage10/final_tests.log). Tests cover individual KPI
failures despite perfect other KPIs, equality, normalized severity, ambiguous
units, invalid budgets, two-launch burst accounting, Wh integration, no launch
energy without a launch, and outgoing/incoming phase safety. The earlier
11-test log is retained. No scientific fitting is performed by the tests.

## 5. What must be resolved to resume the authorized work

Two concrete inputs are needed; questions were sent during the investigation:

1. **A complete service-policy contract:** minimum delivered/received-throughput
   satisfaction, delay budget and scope, jitter budget, and loss target/direction,
   units, sample interval and tolerance. Include the intended priority of avoiding
   feasible service violations relative to energy, or an explicit instruction
   to derive a dominance-preserving reward under that policy. Do not infer this
   from old score caps or tune it to controller winners.
2. **Measurement lineage and reference identity:** identify the submitted MAPPO
   version and provide/identify tester flow mapping and loss-denominator/reset
   construction, or explicitly narrow the scientific claims to named receive/
   stored-loss proxies with study-specific budgets. Existing raw-counter evidence
   can support a count-based target, but changing direction or target must be
   documented and validated, not silently called the old physical quantity.

Once that dependency is resolved, the remaining Stage 10 work remains authorized:
freeze one grouped KPI fitting protocol, validate models, calibrate one reward
from training and stated policy, integrate launch/draining assumptions with the
common actuator, rebuild causal transformed-trace forecasting/baselines/oracles,
and execute the single 2×4×5 campaign. Final testing is authorized **only after**
all those choices and checkpoint selections are frozen. It has not occurred.

No neural or other candidate fits were launched with placeholder labels. No
40-run campaign was consumed on a policy that might need replacement, and no
fallback old score/reward was used. This is a scientific prerequisite, not a
compute-budget or implementation-difficulty stop.

## 6. Preservation and honest deliverable status

New additions are isolated to `revision/stage10/`, its primitive tests,
`revision/reports/stage10/`, the prior-file snapshot and this report. All
**5,266 pre-task scientific files** are checked unchanged; see
[integrity.json](reports/stage10/integrity.json) and
[file_manifest.json](reports/stage10/file_manifest.json). The original branch,
HEAD and tracked files remain unchanged. No commit or push.

**Not delivered yet:** replacement KPI models, finalized environment/reward,
forecast models, revised baselines/oracles, policy training, convergence/seed
statistics, controller traces, final-test results or claim-confirmation tables.
The earlier claims therefore remain exactly as qualified by Stages 2–9. This
report does not confirm, weaken or withdraw claims using nonexistent new results.
The methodology-change motivation is documented above; manuscript-ready numerical
replacement claims await completion of the actual campaign.

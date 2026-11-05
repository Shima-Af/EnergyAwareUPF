# Task 2G — final QoS definition and model-validation plan

**Decision recorded:** retain the actual four-component `QoS_w` definition and
all historical labels. The manuscript's three-component description is a reporting
discrepancy to correct transparently. No model or controller was trained, and the
manuscript was not edited.

## 1. Versioned specification

[qos_score_v1.json](specifications/qos_score_v1.json), identifier
`comcom.recovered_composite_qos.v1`, is the authoritative specification for the
recovered score. Let `clip` restrict a value to the stated interval:

| Component | Exact recorded input | Formula | Weight |
|---|---|---|---:|
| Throughput | `r=DN_Rx_Mbits/s`, `a=Objective Throughput (DL/UL)`, both Mbps | `clip(r/(a+1e−9),0,1)` | .40 |
| Delay | `d=downlink_delay_ms`, ms | `1−clip(d,0,40)/40` | .20 |
| Jitter | `j=RAN_jitter`, µs | `1−clip(j,0,10)/10` | .20 |
| Loss | `p=RAN_PLR_interval`, stored fraction | `1−clip(100p,0,5)/5` | .20 |

`Q = .40 S_th + .20 S_lat + .20 S_jit + .20 S_loss`.
The notebook normalizes weights by their sum, which is already one. The caps are
**40 ms, 10 µs and 5%**; they are score-saturation parameters, not standardized
service budgets. Equality at Q=.90 counts as meeting the composite target.
`QoS_dd` remains an exploratory PCA score and is not the surrogate training target.

The specification preserves clipping and the denominator epsilon exactly.
Missing/nonfinite values must be reported, not imputed into historical labels.
The observed load domain is 0.01–5,000 Mbps per instance; no idle-load behavior
is inferred. Independent specification tests reproduce all 9,158 existing
`QoS_w` values within 2×10⁻¹⁶, consistent with Task 2F.

## 2. Physical meanings: stronger evidence and remaining limits

We inspected the relevant CSV payloads in all **220 LoadCore ZIP exports** and
joined them to the processed rows by exact timestamp. Raw throughput headers are
`Kbits Rx/s`; raw RAN jitter headers are **`Avg us`**.

| Exact raw matches / processed rows | DPDK | USR |
|---|---:|---:|
| Stored DN receive Mbps = exported receive kbit/s ÷1,000 | 4,572 / 4,594 | 4,554 / 4,564 |
| Stored RAN jitter = exported `Avg us` | 4,576 / 4,594 | 4,549 / 4,564 |

All matched values agree exactly. The remaining timestamps have no exact match
in that metric's export; none was silently filled or nearest-neighbor repaired.
Thus the jitter unit is now strongly supported by direct export evidence, rather
than only a notebook column name. See [raw-unit summary](reports/task2g/raw_unit_counter_summary.json),
[row checks](reports/task2g/raw_row_checks.csv), and
[entry headers/hashes](reports/task2g/raw_export_entries.json).

**Offered versus received traffic:** the denominator is a configured per-direction
objective supported by the case catalog, not measured offered aggregate throughput.
The numerator is a **receive-side counter at the DN-labelled endpoint**, not its
transmit counter. It is an achieved-throughput proxy; it does not alone prove
application goodput or complete end-to-end delivery. Both DN/RAN Rx/Tx counters
appear in the exports. End-to-end flow direction requires tester/device mapping;
DN Rx must not simply be renamed “delivered downlink.” The composite combines
that counter with downlink/RAN quality fields. Do not claim a purely downlink
profiling campaign or a validated mapping from bidirectional measurements to the
single traffic-workload scalar without the missing configuration evidence.

**Packet loss:** the stored cumulative RAN loss counter matches the raw counter
on 4,593 DPDK and all 4,564 USR comparable rows; nonnegative within-case differences
with a zero first interval explain the corresponding stored interval numerators.
The raw packet exports contain cumulative Tx/Rx and loss counters, while the
processed `*_Tx_pks`/`*_Rx_pks` fields resemble exported per-second rates; mixing
them as if all were cumulative would be unsafe. Direct rate denominators and
four raw-counter-delta denominator constructions did **not** reproduce every
stored RAN loss fraction. Zero-denominator/reset/aggregation treatment is not
authenticated. [Denominator checks](reports/task2g/loss_denominator_checks.csv)
record comparable counts and failures. Zero DN loss in DPDK makes matching DN
ratios there uninformative. We retain the stored RAN fraction and its exact ×100
conversion without claiming its physical construction is fully validated.

The delay summary's original aggregation, a few unmatched timestamps, acquisition
flow mapping and the loss denominator remain limitations. None invalidates the
confirmed numerical score reconstruction or justifies changing historical labels.

## 3. What Q≥.90 does and does not imply

For an **exact measured score** with components in [0,1],
`1−Q = Σ w_j(1−S_j) ≤ .10` implies each deficit is at most `.10/w_j`.
Consequently:

- Throughput score ≥.75: receive throughput ≥`.75(a+epsilon)`.
- Delay score ≥.50: recorded delay ≤20 ms.
- Jitter score ≥.50: recorded jitter ≤5 µs.
- Loss score ≥.50: stored loss fraction ≤.025, or 2.5%.

Thus Q≥.90 **does imply these conditional bounds**, and cannot coexist with
delay/jitter/loss exceeding their historical score caps. It does not require
every component score to be ≥.90 or full delivery of offered traffic. With other
scores equal to one, 2.5% loss still gives Q=.90—25,000 times a 10⁻⁶ ratio.
These limits do not establish 5QI compliance. They refer to the recorded summaries
and depend on their unresolved physical semantics.

The 3GPP mapping specifies QoS characteristics such as packet delay budget and
packet error rate; it does not define this composite score or provide a 100 ms
jitter column in its standardized 5QI mapping table. See
[ETSI TS 123 501 V18.5.0, §5.7.3–5.7.4 and Table 5.7.4-1](https://www.etsi.org/deliver/etsi_TS/123500_123599/123501/18.05.00_60/ts_123501v180500p.pdf#page=191).
Process/tester-level averages are not automatically end-to-end delay/per-packet
error guarantees. More fundamentally, **predicted** Q≥.90 does not establish even
the measured-score bounds: Task 2E documents false-safe predictions. The reward's
soft penalty also permits tradeoffs; it does not enforce a hard constraint.

Use **“composite quality score”** or **“composite optimization objective”**. In
controller tables use **“predicted composite-score shortfall rate”**; reserve
“measured-score violation” for profiling labels and keep it separate from
individual KPI/service violations.

## 4. Proposed manuscript corrections — not applied

[task2g_qos_replacement.tex](proposals/task2g_qos_replacement.tex) provides an exact
replacement paragraph, component equations, weighted equation and interpretation.
It preserves `eq:qos_weighted` for replacement and adds a component-equation label.
It is not included by the manuscript. Its grouped-protocol paragraph remains
commented because that experiment has not run.

| Current location / statement | Required correction |
|---|---|
| Profiling description, lines 363–365: downlink-only data and 5QI-derived 300 ms / 100 ms / 10⁻⁶ thresholds | Describe recorded DN/RAN counters and the recovered study-specific score; disclose unresolved flow mapping. Keep 5QI only as separately supported background, not the source of these caps. |
| Twin construction, lines 398–403: three components and .40/.40/.20 weights | Replace with the supplied four-component equations and exact raw-column mapping. Explicitly identify the reporting discrepancy. |
| Reward wording, around line 281: QoS compliance “enforced” | State that predicted composite-score shortfalls receive a soft penalty. Do not imply a hard safety mechanism. |
| Evaluation metric, around line 525, and associated tables/captions | Specify that controller “violations” are predicted-score shortfalls; distinguish measured profiling errors, false-safe predictions and individual KPI failures. |
| Abstract, contribution claims and results discussion: no increased/same violations, near-perfect compliance, operational safety | Qualify the metric and evidence. Do not claim unchanged comparative QoS until corrected aligned experiments support it; no replacement numerical claim is proposed here. |
| Profiling “held-out” accuracy paragraph/table context | Disclose historical early-stopping reuse. Describe new results only after grouped validation is executed, without claiming a fresh physical campaign. |

All historical labels/results remain historical evidence. Correcting prose does
not make old test selection, timestamps, recurrence or statistical comparisons valid.

## 5. Prepared grouped QoS validation/retraining protocol

[qos_validation_v1.json](configs/qos_validation_v1.json) specifies a limited future
experiment; **it was not executed**.

| Mode | Train, sweeps 1–3 | Validation, sweep 4 | Internal test, sweep 5 |
|---|---:|---:|---:|
| DPDK | 2,760 rows | 916 | 918 |
| USR | 2,736 rows | 915 | 913 |

Keep each complete sweep/case together; no random row splitting. Use the unchanged
score and source hashes. Fit log1p-load input mean/std on training only, retain
targets in original [0,1] score units, and freeze selection before internal-test
scoring. All data have been inspected previously, so sweep 5 is an internal
redevelopment check, not newly untouched physical evidence.

**Both original neural hidden architectures can be retained.** Replace only the
final linear head with a sigmoid directly in score units; remove centered target
standardization. This guarantees bounded outputs mathematically, including between
measured loads, but does not guarantee calibration or threshold reliability.

The minimal proposal is two fixed variants per mode: original 64/32/16 ReLU hidden
layers with BatchNorm after the first two and .20 dropout after the first; and
the same dense widths without BatchNorm/dropout. Seeds are 42/123: **eight fits
total**, at most 600 epochs each, Adam .001, batch 64, validation stopping patience
80, LR-halving patience 30 and minimum LR 10⁻⁵. Loss is training-row MSE. Save best
validation weights with scaler/configuration/seed/code/runtime provenance.

Select by lowest validation-row RMSE separately for each mode; exact ties use
parameter count then candidate name. Report both seeds' validation behavior.
Score only each selected neural candidate on internal test, alongside fixed
training-only references. No test-based selection, search expansion, train-plus-val
refit or default-twin promotion is allowed.

Simple references are (1) the training target mean and (2) a piecewise-linear curve
through training per-load means, in raw Mbps, with no extrapolation or unsupported
monotonicity constraint. These references receive identical groups and scoring
scopes. Original saved-network comparisons remain descriptive because those
networks saw rows from all five sweeps.

Required reporting: MAE, RMSE, R², bias, residual quantiles; false-safe/unsafe
counts with explicit all-row, predicted-safe and measured-unsafe/safe denominators;
predicted-safe coverage; measured shortfall `max(0,.90−Q)` mean/p95/max and
false-safe optimism. Report by partition, complete sweep and exact load, including
≤400 Mbps and USR(2)'s ≤200 Mbps per-instance study range. Separate measured- and
prediction-centered ±.05 threshold bands. Add load-macro summaries and seed spread
without treating correlated rows as independent physical replicates. Null rates
for absent denominators must not be reported as zero risk.

Validation RMSE identifies a point-predictor candidate, **not pilot readiness**.
Before fitting, agree acceptable false-safe frequency/severity and minimum safe
coverage (or justify rejecting every action), plus the group-level evidence
required. No numerical risk tolerance was supplied, so none is invented from
historical test outcomes. Independent calibration/physical campaigns would be
needed for stronger uncertainty or deployment claims. No traffic outcomes enter
profiling-model selection.

## 6. Next implementation task and verification

**Recommend Task 2H: implement and execute the prepared bounded-QoS grouped
comparison**, after risk/reporting tolerances and the conditional physical scope
are explicitly agreed. Retain the four-component target; do not redefine it,
reopen power-model searches or train controllers. Report whether bounded neural
models improve score-threshold reliability over the simple references. Treat
unresolved counter semantics as limitations until independently resolved.

Seven focused tests verify both full label sets, exact composite implications,
the loss counterexample, the proposed grouped budget and transparent unmatched-row
handling. Execution results are in [the test log](reports/task2g_tests.log).
An initial exact-raw-join assumption failed because some raw timestamps are absent;
the diagnostic now records those gaps explicitly. An initial test compared a
floating ratio with exact integer equality; it was corrected to numerical
approximation. Initial logs are retained. No model outputs or historical data
were repaired.

All new paths/hashes are listed in [the file manifest](reports/task2g/file_manifest.json).
[Integrity evidence](reports/task2g/execution_and_integrity.json) verifies the prior
repository/revision files and original profiling artifacts. **Stopped after
Task 2G:** no model training, controller training, traffic final-test outcomes,
manuscript edits, commit or push.

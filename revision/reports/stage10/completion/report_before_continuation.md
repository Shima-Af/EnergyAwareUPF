# Stage 10 — approved QoS policy implemented; physical loss-target blocker confirmed

**Campaign status: incomplete; no controller training or final testing has started.**
The user has resolved the policy and reference-identity questions. This update
supersedes the earlier claim that service budgets or the submitted paper are missing.
The remaining blocker is now more precise: the historical stored loss fraction
has an exactly reconstructable but dimensionally inconsistent denominator.
Training a physical loss-ratio predictor against it would violate the Stage 10
requirement to authenticate the target rather than invent measurement semantics.

The earlier report is preserved verbatim in
[report_before_budget_decision.md](reports/stage10/reconciliation_v2/report_before_budget_decision.md).
Stages 2–9, original labels and all trained artifacts remain unchanged.

## 1. The selected study policy is implemented

[policy_v1.json](stage10/policy_v1.json) records the explicit author decision:

`T = clip(received_Mbps / offered_Mbps, 0, 1)`

`D = 1 − clip(downlink_delay_ms, 0, 40)/40`

`J = 1 − clip(RAN_jitter_us, 0, 10)/10`

`L = 1 − clip(100 × verified_loss_fraction, 0, 5)/5`

**`Q = min(T,D,J,L)`, with `Q ≥ .90`.** There is no QoS-formulation comparison.
The corresponding study budgets are throughput satisfaction ≥90%, delay ≤4 ms,
jitter ≤1 µs, and verified loss ratio ≤0.5%. These are author-selected study-policy
budgets, not 5QI guarantees. The loss budget is conditional on authenticating a
physical fraction; no count-based five-packet tolerance is substituted.

The positive offered-load domain is explicit. The old 1e−9 denominator epsilon
is removed to express the approved positive-load throughput ratio directly;
zero offered load fails rather than receiving invented delivery semantics.
This numerical change is documented, not silently applied to historical labels.
Normalizing component scores saturate as selected; invalid physical inputs are
rejected rather than repaired.

[selected_policy.py](stage10/selected_policy.py) supplies this policy, strict
per-KPI compliance and maximum normalized budget-exceedance severity. A good KPI
cannot compensate for a bad one. The continuous severity remains separate from
unfixed reward coefficients; it is not a claim that reward calibration has run.
Its authentication gate rejects four-KPI training-target creation while loss
or throughput provenance remains unverified. No new KPI training labels have
been published as authenticated measurements.

The earlier tested energy/serving-phase primitives remain available: time-weighted
outgoing/incoming operational Wh, activation energy once, and conjunction/worst
severity across configurations that actually serve. They are not yet integrated
as the replacement controller environment.

## 2. Authoritative Resources identified and read

The local file [1571355221 paper.pdf](reference/1571355221%20paper.pdf) has the
exact submitted title **“Evaluating Centralized Value Learning in Shared-Policy
MARL for Energy-Aware Multi-Site 5G UPF Selection.”** Neither earlier repository
`paper.tex` candidate has this title; no correspondence to those versions is
assumed. The PDF itself is the methodological reference.

[Thesis_ShimaAfshar_Final.pdf](reference/Thesis_ShimaAfshar_Final.pdf) has the
specified thesis title. Its §5.6.1–5.6.2, printed pages 74–75, confirms:
24 s / 3.3 s activation, burst ratios 1.73 / .37, the ratio-based
`k × P_idle × t_activation / 3600` Wh model, and outgoing service until incoming
activation, with both serving configurations considered for safety. Its idle
values remain portable-model inputs rather than newly observed COMCOM idle
measurements. No transient outage or replica-drain measurement is invented.

The submitted MAPPO PDF uses a minimum delay/loss score with a five-packets-per-3-s
loss count; it distinguishes score shortfall from strict budget exceedance.
That supports the non-compensatory principle, but does **not** authenticate the
COMCOM stored ratio or override the newly approved four-component normalization.
The thesis is authoritative for the activation model as requested.

PDF SHA-256 identities are in
[authoritative_pdf_manifest.json](reports/stage10/reconciliation_v2/authoritative_pdf_manifest.json).
Searchable PDF extractions are saved in the same directory. These are local
resources; no inaccessible service or draft was presented as the submitted PDF.

## 3. New loss reconstruction: exact arithmetic, invalid dimensional interpretation

Read-only inspection found this formula reproduces **every stored RAN loss value**:

`RAN_PLR_interval = RAN_pks_lost_interval / (RAN_Tx_pks + RAN_pks_lost_interval)`

The zero-denominator convention is zero. Maximum absolute discrepancies are
8.33e−17 for DPDK and 1.11e−16 for USR. This is a numerical reconstruction,
not recovery/authentication of the original missing producer script.

| Check | DPDK | USR |
|---|---:|---:|
| Stored fractions reproduced | 4,594/4,594 | 4,564/4,564 |
| Timestamp-exact raw Tx-rate comparisons | 4,593 | 4,564 |
| Stored `RAN_Tx_pks` equals raw `GTPu Packets Tx/s` | 4,593/4,593 | 4,564/4,564 |
| Comparable within-case raw loss-counter differences | 4,483 | 4,454 |
| Differences match stored interval lost counts without reset handling | 4,482 | 4,449 |
| Comparable timestamp differences equal 3 s | 4,483/4,483 | 4,454/4,454 |

**The denominator adds a packet rate (packets/s) to a three-second interval
packet count (packets).** It is not an authenticated loss fraction over a common
time base. The old name and bounded [0,1] values do not fix that dimensional error.
First rows of all 110 cases per mode also have imposed zero interval counts;
they are not evidence of measured zero loss. Six nonmatching raw counter
transitions across the modes need explicit reset/decrease handling rather than
being silently treated as measured loss-free intervals. One DPDK raw timestamp
is unavailable; no nearest-time repair was introduced.

Example, USR source row 235: lost count 1,059, stored/raw Tx rate 40,000 packets/s,
and stored fraction `1059/(40000+1059) = .025792…`. Its three-second count cannot
be added directly to that rate. Multiplying by three would remove one unit
mismatch, but does **not** establish the right packet population, direction or
loss-counter definition; no such guessed “correction” was used.

Full results, matched raw export headers/hashes and examples:
[loss_reconstruction.json](reports/stage10/reconciliation_v2/loss_reconstruction.json),
[raw_export_inventory.json](reports/stage10/reconciliation_v2/raw_export_inventory.json),
[loss_examples.csv](reports/stage10/reconciliation_v2/loss_examples.csv).
This is a physical-target limitation. Historical composite labels still
numerically reproduce their historical formula; they were not overwritten.

## 4. Tester configuration recovery and remaining direction evidence

The original LoadCore ZIP exports contain nested scenario ZIPs, including
`config-data.bin` (JSON despite its extension) and `config-metadata.json`.
They were inspected in place without extracting or modifying the raw archives.
All 220 available scenario configurations were inventoried with hashes.
The JSON contains DN-side `statelessUDP.direction=downlink` and UE-side
`statelessUDP.direction=uplink` settings. This supports the bidirectional
endpoint interpretation, but stored configurations also contain potentially
inactive protocol objects. It is not by itself a full metric/flow dictionary.

The correct provisional name for `DN_Rx_Mbits/s` remains DN-received throughput,
not automatically delivered downlink. Its units and exact raw conversions were
confirmed previously. The tester’s active application/metric mapping and
packet-accounting definition must establish which received traffic and expected
packet population the loss ratio refers to. Jointly imposing study budgets on
named directional proxies is distinct from proving end-to-end delivery.

[tester_config_inventory.json](reports/stage10/reconciliation_v2/tester_config_inventory.json)
records selected non-sensitive direction fields and nested-config hashes. Full
configuration payloads, credentials and endpoint details were not copied into
the report. The original ZIPs remain the read-only source.

## 5. Precise prerequisite to resume physical KPI fitting and the 40-run campaign

The budgets and primary references are **resolved**; they are not requested again.
What remains is a verified dimensionless loss fraction with numerator and
denominator counting the same traffic population, direction and time window,
including the tester’s definition of “Lost Packets,” resets and first intervals.
The DN-receive throughput mapping also needs its active tester metric linkage.

Useful missing evidence is the original preprocessing producer and/or the
LoadCore metric/flow definition that identifies expected/sent versus received
packets and the loss-counter semantics. The nested original configurations now
provide concrete artifacts for that reconciliation. A documented reconstruction
from compatible raw counters may be possible, but a rate-plus-count arithmetic
fit is not a scientifically valid target. No substitute denominator, ×3 rescale,
count tolerance or synthetic measurement was selected merely to proceed.

Under the user’s explicit instruction, **this is the precise material scientific
blocker**. No new comparative QoS search was conducted. Four-KPI fitting, reward
calibration, revised baseline/oracle integration, 2×4×5 controller training and
once-only final testing remain unexecuted; no replacement numerical claims are
reported. This is incomplete work pending measurement semantics, not completion
of the requested campaign or a stop due to compute cost.

## 6. Verification and preservation

**24 tests passed in 0.03 seconds**: 13 accounting primitives plus 11 selected-policy
and evidence checks. They cover all four fixed budget boundaries, independent KPI
failures with otherwise perfect KPIs, severity, invalid-input rejection, provenance
failure, raw rate matches and full stored-ratio reconstruction. No scientific
model fits or controller calls occur in tests.
[Final test log](reports/stage10/reconciliation_v2/final_tests.log).

New work: `stage10/policy_v1.json`, `stage10/selected_policy.py`,
`stage10/authenticate_measurements.py`, `tests/test_stage10_selected_policy.py`,
this updated report and the isolated `reports/stage10/reconciliation_v2/` evidence.
The earlier report and evidence are preserved; no Stage 2–9 file is modified.
See the continuation integrity and file manifests in that directory.

No profiling data/model, traffic, forecaster, trained policy, reward or historical
baseline was changed. No traffic final-test target was read or scored. No
manuscript edit, commit or push. Branch remains `revision/comcom-clean-evaluation`.

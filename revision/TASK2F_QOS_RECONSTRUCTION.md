# Task 2F — targeted QoS score reconstruction

**Yes: the recovered notebook's scoring functions reproduce every historical
`QoS_w` label using an explicit mapping to the available measurements.** All
4,594 DPDK and 4,564 USR rows match within 1.12×10⁻¹⁶. Its PCA procedure also
reproduces every `QoS_dd` label within 8.33×10⁻¹⁶. This establishes a numerical
reconstruction, although the missing pre-label files and deleted/stale notebook
steps prevent authentication of the complete original execution history.

This notebook is **different from the notebook inspected in Task 2E**. The new
reference SHA-256 is
`ef3655a87bac78b022d83d872d61bc2202a98cc7de47f0399a7c05e3e2a0851c`;
the external notebook used previously has SHA-256
`49a8af4d1d23866b51866d1035b8cd4e03d0df9d7603eb6f9c10fce6a1214a30`.
The earlier report remains unchanged. Its unresolved *numerical scoring* finding
is superseded here; its measured surrogate errors and lack of independent model
validation remain applicable.

## A. What the surviving notebook actually contains

There are eight cells: seven code cells and one Markdown cell. Code-cell execution
counts in file order are **2, 25, 26, 14, 24, 16, 22**. Cell 4 defines:

`b(x,c) = 1 − clip(x,0,c)/c`

`S_th = clip(delivered/(objective + 10⁻⁹), 0, 1)`

`S_lat = b(delay,40); S_jit = b(jitter,10); S_loss = b(loss,5)`

`QoS_w = .40 S_th + .20 S_lat + .20 S_jit + .20 S_loss`.

Weights are divided by their sum inside `weighted_qos`; the defaults already sum
to one. **Throughput is active**, not commented out. Expected default columns are
`Avg DL Throughput (Mbps)`, `Avg Objective Throughput (Mbps)`, `Avg DL Delay (ms)`,
`Avg DL Jitter (us)`, and `Total Packet Loss (%)`. Their implied caps are **40 ms,
10 µs and 5%**, respectively. The argument name `loss_frac_col` and its “lost/tx”
comment conflict with the percentage column name; the matching conversion is
established below. No packet-size conversion is applied by the score functions.

PCA uses the **same four component scores**: StandardScaler fitted to all supplied
rows → one-component PCA (`random_state=0`) → flip PC1 if its correlation with
`S_th` is negative → MinMaxScaler fitted to PC1. In equations, `Z=(S−μ)/σ`,
`u=Z v₁`, oriented toward throughput, and `QoS_dd=(u−min(u))/(max(u)−min(u))`.
The separate DPDK/USR all-row fitting procedure numerically reproduces the saved
scores; no historical serialized PCA artifacts were recovered. PC1 explains
42.09%/65.87% of standardized variation. Its throughput correlation is only
0.00823 for DPDK versus 0.61836 for USR, so the orientation rule is weakly anchored
for DPDK. These are dataset-relative scores, not calibrated compliance probabilities.

**Neither scoring function is called by any surviving top-level cell.** Execution
count 14 records definition of the functions, not execution of their bodies.
No cell assigns `QoS_w`/`QoS_dd`. Later cells plot/group those names and
`Avg Objective Throughput (Mbps)`, yet the saved loading output shows 28 raw-style
columns with none of those three names. Reload counts 25/26 postdate the plotted
states 16/22/24. The saved plots therefore cannot establish a clean top-to-bottom
execution; earlier kernel state or deleted cells were involved.

Calling `compute_component_scores` as written on the available training data
raises `KeyError: 'Avg DL Throughput (Mbps)'`. The five expected default column
names are absent: literal execution yields **zero comparable scored rows**.
We executed the extracted, inspected definition cell in an
isolated diagnostic namespace; we did not rerun or modify the notebook. Full
sources, output text and the three preserved PNG outputs are in
[notebook inspection](reports/task2f/notebook_inspection.json) and
[direct execution checks](reports/task2f/direct_execution_checks.json).

## B. Data lineage: what was and was not recovered

The exact `dpdk_df_processed.csv` and `oai_df_processed.csv` were not found in the
accessible `/home/ubuntu` and `/tmp` trees or the inspected local Git histories of
the main repository, xRL, DDT and dynamic-upf-selection projects. Some private
system directories under `/tmp` were unreadable; this is not an exhaustive claim
about every file on the machine. No remote fetch was performed.

The notebook's full saved column lists contain **28 columns without either QoS
label**. Later `dpdk_processed.csv`/`oai_processed.csv` have those same ordered
columns plus `QoS_w` and `QoS_dd`. Every visible value in the two saved five-row
heads matches the corresponding training-file prefix to its displayed precision.
That supports a related earlier stage; it does **not** prove all missing-file rows
were identical or that only columns were added.

The later training files have unique `(timestamp, test set, test case, offered
load)` keys. Copies in DDT, xRL and dynamic-upf-selection match one-to-one and are
byte-identical across all rows. Earliest locally recovered training-file history
already contains both labels: dynamic-upf-selection commit `f256620` on **28 August
2025**, then xRL `ae37813` on **30 September 2025**. These are observed occurrences,
not proven label-creation dates. The intermediate write/rename step is missing;
a full missing-file-to-training-file row join cannot be performed.

[Lineage evidence](reports/task2f/data_lineage.json) records searches, historical
blob hashes, schemas and copy joins. [Saved-head comparisons](reports/task2f/saved_head_comparisons.csv)
record the limited visible-row verification.

## C. Recovered mapping and candidate comparisons

The exact matching mapping is:

| Notebook input | Available measurement / transformation |
|---|---|
| Delivered Mbps | **`DN_Rx_Mbits/s`** |
| Objective Mbps | `Objective Throughput (DL/UL)` |
| Delay ms | `downlink_delay_ms` |
| Jitter, raw value interpreted as µs | `RAN_jitter`, **without conversion** |
| Loss percent | **`100 × RAN_PLR_interval`** |

This combines a DN receive throughput counter with downlink/RAN quality fields.
It is not a demonstrated all-downlink or all-uplink score. The counter/field names
and earlier feature mappings support testing receive-side DL/UL interpretations,
but the generic “Avg DL” function labels do not authenticate the physical
direction of the matching counter. Numeric reconstruction confirms the columns
and conversion; acquisition documentation must settle their physical meaning.

The finite diagnostic comparison held notebook weights, caps and clipping fixed:
four recorded Tx/Rx throughput fields × two directional KPI sets × two loss-unit
interpretations. Tx fields were negative controls for offered-versus-delivered
confusion. Additional controls used combined-direction means, omitted throughput,
the manuscript's stated score, and PCA. **No arbitrary weight, cap or transformation
optimization was performed.** All mappings and all 42 mode/candidate comparisons
are preserved in [candidate results](reports/task2f/candidate_results.json).

Counts below use absolute difference ≤10⁻¹⁰ as a floating-point reconstruction
tolerance, not an accuracy acceptance criterion. Comparable N is 4,594 DPDK and
4,564 USR for every executable row.

| Candidate versus historical `QoS_w` | DPDK matches | MAE / max difference | USR matches | MAE / max difference |
|---|---:|---:|---:|---:|
| **Notebook: DN receive + DL KPIs, loss ×100** | **4,594** | **4.46e−17 / 1.11e−16** | **4,564** | **2.73e−17 / 1.11e−16** |
| RAN receive + DL KPIs, loss ×100 | 2,150 | .004905 / .400000 | 1,257 | .067226 / .400000 |
| DN receive + UL KPIs, loss ×100 | 5 | .034461 / .439880 | 19 | .095789 / .519805 |
| DN receive + DL KPIs, loss left as fraction | 4,545 | .001920 / .197984 | 3,180 | .048838 / .197990 |
| Combined directions, equal directional means | 2 | .023102 / .400222 | 24 | .044650 / .400119 |
| Omit throughput, retain other .20 weights | 187 | .382375 / .400000 | 159 | .358821 / .400000 |
| Manuscript, linear DL interpretation | 93 | .044460 / .600179 | 1 | .144039 / .743736 |
| Manuscript, linear UL interpretation | 1 | .047541 / .839965 | 0 | .207272 / .998619 |
| Notebook PCA, compared to `QoS_w` | 93 | .043386 / .566699 | 19 | .044082 / .107502 |

The manuscript controls use latency/jitter/loss weights .40/.40/.20, caps 300 ms /
100 ms / 10⁻⁶, notebook-style linear bounded scores and raw-jitter µs→ms conversion.
The manuscript does not fully specify component functions, so these tests reject
those explicit interpretations, not every possible bounded function.

Bitwise floating-value equality for the matching weighted formula is 2,744/4,594
and 3,159/4,564; the remaining differences are at most 1.11e−16. We do not describe
different floating-point serialization as byte-identical labels.

PCA is a distinct historical score, not the source of `QoS_w`:

| Notebook PCA versus stored `QoS_dd` | Matches / N | MAE | Maximum difference |
|---|---:|---:|---:|
| DPDK | 4,594 / 4,594 | 6.28e−17 | 6.11e−16 |
| USR | 4,564 / 4,564 | 1.69e−16 | 8.33e−16 |

Both PCA and its normalizers are reconstructed separately on each complete mode's
data, as the surviving function does. This diagnostic PCA calculation is expressly
for label reconstruction; no saved profiling regressor was trained or refitted.
[PCA parameters](reports/task2f/pca_reproduction.json) and
[all reconstructed components](reports/task2f/reconstructed_components.csv) permit
independent checking.

**Throughput explains the previously unexplained label jumps.** After subtracting
`.40 S_th`, the maximum residual label span within all 544 DPDK and 392 USR
same-load/same-DL-KPI discrepancy groups is 3.33e−16. For example, DPDK at 2 Mbps,
zero delay, jitter 2 and zero loss has score 0.56 when DN receive throughput is
zero and 0.96 when it is 2 Mbps. These are consistent four-component labels, not
unexplained conflicts of a three-component formula. Changing the traffic-direction
mapping or failing to convert loss fraction to percentage also explains substantial
mismatch; neither the all-UL nor the all-DL mapping reproduces all labels.

## D. What is established and the least disruptive next step

**Confirmed numerically:** all existing weighted and PCA labels are reproducible;
the weighted score includes throughput with weight .40, the specified DN/RAN
column combination, 40/10/5 caps and percentage loss conversion. The old training
script consumes `QoS_w`, not `QoS_dd`. This closes the main numerical-definition
gap in Task 2E without inventing or changing any labels.

**Plausible but not authenticated:** a missing mapping/renaming and scoring cell
applied these functions to the raw-style data, with PCA fitted per mode on all
rows. Exact agreement strongly supports that computation but does not identify
the original code execution or explain the saved plots' earlier dataframe state.

**Still missing:** the exact pre-label CSVs, full intermediate lineage, original
PCA serialization/version and acquisition-level confirmation of counter direction,
jitter units and loss construction. Numerical equality alone does not authenticate
those physical interpretations or make a .90 score a standardized service guarantee.

**Recommendation:** retain the existing labels as the now-reconstructed historical
four-component score, with a versioned explicit specification. A new label definition
is **not necessary solely because provenance was previously unresolved**. Do not
silently relabel them as the submitted manuscript's three-component 5QI-related
score. Resolve the direction/unit semantics before adopting the reconstructed
score for a revised campaign. If the intended scientific claim requires the
manuscript's different three-component definition, transparently redefine/rebuild
labels and regenerate affected model/controller comparisons under that declared
contract. Do not retrofit the old numbers to that definition.

Task 2E's false-safe counts, correlated-sample split and load-only prediction
limitations remain real. Recovering labels does not validate those surrogates or
authorize RL training. A grouped reliability/refitting decision should now use the
recovered score and confirmed physical semantics, rather than assume label corruption.

## Verification and stop

**11 focused reconstruction tests passed in 0.87 seconds.** They cover literal
schema failure, clipping/weights, all weighted/PCA labels, missing calls, execution
counts, throughput explanations, lineage limits and floating-point match semantics.
The [test log](reports/task2f_tests.log) and
[file manifest](reports/task2f/file_manifest.json) list verification and every new
file. Two initial diagnostic harness failures (relative output paths and incomplete
filesystem-search handling) were corrected; their logs/partial outputs are retained
separately. They occurred before scoring and did not alter historical material.

The notebook, labels, models, prior reports and manuscript are unchanged. No QoS
surrogate or controller training, traffic final-test outcome access, commit or push
occurred. **Stopped after Task 2F.**

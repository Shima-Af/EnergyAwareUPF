# Baseline definition and historical reconciliation

The `main` objects were read at `c0734715418d3f6d10785e5174aebb4bac24d0da` without checking out another branch. Copies and SHA-256 identities are under `revision/reports/stage5/sources/`. This audit adapts decision logic to the revised twin; it does not reproduce the submitted numerical tables.

## A. Published principles and declared extension

The [published paper](https://networking.ifip.org/2025/images/Net25_papers/1571142002.pdf), Sections III-B–D/G and IV-C/D, specifies a 0–100 performance score, globally normalized and bounded 0–100 energy-efficiency score, weighted utility, forecast decisions, traffic hysteresis and minimum residence time. Its reported parameters are 0.2/0.8, 26 Mbps and four intervals; the old binary crossover is 95.6 Mbps. The user-uploaded `revision/reference/Dynamic_UPF_Cyprus_2025_SLICES.pdf` agrees on these defining details. It differs in PDF bytes/layout from the stamped conference copy; neither file is represented as byte-identical to the other.

**New, explicit assumptions:** the revised composite score supplies `P=100*Q`; the earlier paper's polynomial KPI aggregation is not recreated. Set `E=100*(Wmax-W)/(Wmax-Wmin)` bounded to [0,100] for utility only. `U=.2*P+.8*E`. Normalization covers every action on a synthetic .02–400 Mbps grid at .25-Mbps spacing, plus .02. Bounds come from the already approved study domain/training peak, not validation traffic or PPO. Physical twin outputs are never clipped. Extrema are sampled profiling extrema, not a certified continuous optimization. The exact normalization values are frozen in `baseline_frozen.json`.

All three utilities compete; ties favor DPDK, USR(1), then USR(2). Changes in the winning utility are bracketed on that grid and refined to 1e-5 Mbps. No threshold is sorted or replaced to manufacture an intermediate region. A finite scan does not certify absence of very narrow unsampled regions; an encountered winner inconsistent with the frozen map raises an error.

The extension defines hysteresis by eroding each winner region by **26 Mbps at every finite crossover boundary**. A request changes the executed configuration only in the remaining strict interior; otherwise it retains that executed configuration. Domain endpoints are not crossovers. This resolves overlapping bands in favor of retention, permits jumps across regions, and reduces to the paper's binary threshold±margin rule. It is one declared multi-configuration generalization, not a uniquely specified published three-action algorithm. A source-dependent pairwise hysteresis is another possible extension; it was not tuned or substituted after validation.

| Utility-winning region (Mbps) | Winner | Interior accepting entry from another action |
|---|---|---|
| .02–15.781867 | USR(1) | Empty after upper-bound erosion |
| 15.781867–87.370037 | USR(2) | Strictly 41.781867–61.370037 |
| 87.370037–400 | DPDK | >113.370037 through 400 |

A current winner is retained, including in its boundary band. Initialization is DPDK, so USR(1) is unreachable in this version. USR(2) remains available and occupied. This follows the fixed published margin and the declared extension, not validation tuning. Large forecast jumps can skip an entry interior, and occupancy can persist outside an entry interval. The performance consequences must be attributed to these manual design choices rather than automatically to learning. The common actuator supplies the only four-slot hold-down; there is no second internal timer or reconfiguration-cost optimization.

## B. Immutable Stage 4 rule

`revision.stage3.pipeline.RuleController` is reused directly. It ranks all three configurations at the same forecast: first Q>=.90, then minimum power; if none meets the point target, highest Q then lower power. Ties favor lower action integer. If both current and best are feasible, a saving <=2% retains current. Thus its boundaries depend on the current configuration and feasibility; there is no single traffic threshold or 26-Mbps margin. Execution uses the common four-slot guard. All Stage 4 validation metrics reproduce exactly.

**Unexecuted stronger-rule proposal:** simulate each requested action with the common actuator and maximize forecast one-step reward, including .030 switch-to-DPDK, .012 per launch and .003 scale-down. This is transparent but myopic: it does not value four future committed intervals or forecast error. No proof that it dominates B follows from reward alignment. No separate training-only development comparison/acceptance criterion was frozen in the existing study, and validation/PPO outcomes are already exposed. Therefore no improved variant was selected or scored here. A later authorized development protocol can compare fixed candidates on training-only causal forecasts, freeze its criterion and rule, then assess validation with reuse disclosed. B remains immutable.

## C. Manuscript, executable code and table provenance

| Evidence identity | What it establishes | What it does not establish |
|---|---|---|
| Submitted `main:Paper artifacts/manuscript_energyawareupf.tex`, baseline and orchestration sections | Describes threshold/dead-band selection, common action space/guard; attributes absent scaling to 90.47/87.13-Mbps boundaries and ~23-Mbps overlapping bands | Does not authenticate those rules as the code that generated each table |
| `main:scripts/baseline_rule.py`, `RuleBasedSelector.decide` | Separate region selector with margin and its own timer; constructor writes thresholds/plots | The evaluation loop never calls this decision method. Its four subsequent held calls also differ from common t+4 eligibility |
| Same file, `_compute_thresholds` | Uses .3/.7, global power normalization, a QoS-deficit transform and utilization penalty; compares **DPDK vs USR(k)**; monotonicity enforcement can merge thresholds | Its k=2 crosspoint is not directly a USR(1)-vs-USR(2) scaling crossover |
| `main:scripts/evaluation_baseline-rule.py` | Constructs selector but directly ranks `.3*Q+.7*100*(3.58-P)/3.58` in its loop; margin only supplies plot lines; multi-USR disabled by config | Does not enforce the drawn ±23-Mbps margin, the selector's QoS barrier, or its region logic |
| `main:experiments/results/paper_results_assets/compile_baselines_table.py` | Independently contains the same direct binary utility ranking and loop cooldown; writes `polishe_results` | Presence of code is not a signed execution manifest linking it to the submitted 720-row table |
| Stored threshold JSONs | `results/baseline`: Th1=90.4790005994, K_eff=1; `experiments/results/Baselines`: Th1=90.3393499759, K_eff=1; both .3/.7 and multi-USR false | Neither stores a second 87.13-Mbps threshold. 90.479 is not exactly the manuscript's 90.47 |
| Historical CSV inventory | Two baseline CSVs have 721 and 720 rows respectively, and distinct hashes | No timestamp/config/checkpoint execution manifest authenticates one complete submitted-table generation chain |

`src/utils.py` inverse-transforms the QoS output without multiplying it by 100. Consequently the executed utility mixes Q in [0,1] and E on a nominal 0–100 scale despite lookup comments suggesting 0–100 performance. **Do not repair this historical weighting.** No hard .90 feasibility filter is active in the binary ranker. The plotting threshold constructor's extra transformations are not its selection objective.

Canonical `comcom_historical` is the executable inline ranker corroborated by the compiler, with DPDK ties, only DPDK/USR(1), no active traffic margin and Wmax=3.58. The old 90.47/87.13 numbers and 23-Mbps plot band are recorded, not used to fabricate decisions. Revised twin continuous inference replaces historical nearest-lookup predictions explicitly; revised composite QoS/power and current causal forecasts are common to all comparators. Its new implicit utility crossover is **89.111956 Mbps**, not an asserted reproduction of an old threshold. The historical loop's end-of-slot decrement permits changes at t+4, matching the common guard; a synthetic exhaustive request-sequence test verifies this. Blocked commands are explicitly counted by the common actuator.

The manuscript-described algorithm retains its identity in this audit but is **not separately evaluated as an authenticated historical implementation**: active scaling regions/overlap priority are not established by that text and the executed source. The canonical choice is source-driven, not selected from comparative energy. Historical table values remain historical evidence with unresolved generator provenance.

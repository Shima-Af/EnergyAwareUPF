# Audit addendum: reconciliation with the submitted manuscript

**Manuscript:** COMCOM-D-26-01061  
**Date:** 22 October 2026  
**Authority:** The user identifies the newly attached TeX as the version submitted to the journal. It supersedes the repository TeX for interpreting submission claims and reviewer comments.

This addendum corrects manuscript-dependent statements in the [original audit](/home/ubuntu/EnergyAwareUPF/docs/audits/2026-10-22/SCIENTIFIC_REPOSITORY_AUDIT.md). The original audit remains unchanged as a record of what was inspected. Where they differ about the submitted paper, this addendum takes precedence. No experiments were rerun, no code changed, and neither manuscript was edited.

The verdict remains **B. PARTIALLY REUSABLE**, with high confidence in the repository findings. Identifying the submitted text resolves the manuscript-version ambiguity; it does not resolve the independently observed code, evaluation, or artifact discrepancies.

## 1. Authoritative source and corrections

A [byte-identical submitted TeX snapshot](/home/ubuntu/EnergyAwareUPF/docs/audits/2026-10-22/submitted-manuscript-reconciliation/submitted_reference.tex), [full comparison](/home/ubuntu/EnergyAwareUPF/docs/audits/2026-10-22/submitted-manuscript-reconciliation/repository_vs_submitted.diff), and [SHA-256 source manifest](/home/ubuntu/EnergyAwareUPF/docs/audits/2026-10-22/submitted-manuscript-reconciliation/source_manifest.json) are preserved separately under `/tmp`. Line references below refer to that snapshot.

| Original audit statement or caveat | Submitted source evidence | Corrected interpretation |
|---|---|---|
| Local manuscript lists costs 0.5/0.2/0.05, conflicting with the supplied specification | §6.2, lines 484–486: 0.03/0.012/0.003 | **Withdraw this as a criticism of the submitted parameter table.** Those older values belong only to the repository manuscript. The submitted coefficients agree with the initial task specification. |
| The submitted cost semantics required confirmation against the actual submission | §4.5, lines 288–297: explicit orchestration-cost equation and definition of launched instances | The direction-specific, per-launch specification is now directly confirmed. The code mismatch remains and is more firmly established. |
| Local abstract uses the weaker phrase “low QoS violation rate” | Abstract, line 74: “without increasing QoS violations” | The original concern applies directly to the submission. Its reported means, .00333 versus .0014, contradict an unqualified no-increase statement. |
| Reviewer says 11 results subsections but local source contains 12 | Submitted §7 has 11 subsections and no separate “Summary of Findings” | **Resolve this caveat.** The reviewer's description matches the submitted source. |
| Startup figures were supplied only as context for the audit | §4.5, line 297: DPDK approximately 24 s, mean over 100 repetitions; USR approximately 3.3 s per container | These are explicit submitted claims. Their measurement logs were not validated by this reconciliation; neither figure establishes continuous service during a mode change. |
| Baseline's lack of scaling needs investigation | §7.5, line 626: thresholds 90.47 and 87.13 Mbps, approximately 23 Mbps dead-band; overlap prevents scaling | The submission contains a specific explanation that must be tested against the original baseline implementation. Audited configs disable multi-USR, and the publication compiler ranks only DPDK/USR(1); neither demonstrates the stated overlapping-dead-band mechanism. |
| Low DPDK R² should not be dismissed as harmless | §5.3, line 434 explicitly attributes low DPDK R² values to near-zero target variance, citing .0039 W RMSE | The submitted defense uses a **power-model** error to support a statement about DPDK models collectively. It does not validate the separate QoS model, whose RMSE is .0847 and whose threshold errors were recovered in the audit. |
| Exact submitted version remains unidentified | New attachment and user confirmation | Treat submission identity as resolved for this audit. Recovering its matching code/data/checkpoint provenance remains necessary. |

## 2. Confirmed reward mismatch

The [submitted equation](/home/ubuntu/EnergyAwareUPF/docs/audits/2026-10-22/submitted-manuscript-reconciliation/submitted_reference.tex:288) charges:

`L_ORCH = 0.03 * I(switch to DPDK) + 0.012 * number_of_USR_instances_launched + 0.003 * I(scale-down within USR)`.

The text explicitly includes USR launches on a mode switch. In contrast, [the audited implementation](/home/ubuntu/EnergyAwareUPF/src/environment.py:395) charges the same `type_switch_cost` for either switching direction and applies scale costs only when the UPF type stays unchanged.

| Executed transition | Submitted equation | Audited code |
|---|---:|---:|
| USR → DPDK | .030 | .030 |
| DPDK → USR(1) | .012 | .030 |
| DPDK → USR(2) | .024 | .030 |
| USR(1) → USR(2) | .012 | .012 |
| USR(2) → USR(1) | .003 | .003 |
| Change blocked by cooldown | 0 | 0 |

Thus, matching the three YAML numbers is insufficient: the events to which they apply differ. The submitted text and code agree that penalties apply to executed changes after the guard. Scale-down is described as a fixed event in §4.5 and “per-instance” in the table; with only one instance removable per within-USR transition, these are numerically equivalent here. Generalizing beyond k_max=2 would require an explicit choice.

No reward changes or retraining were performed. The archived policies remain evidence of the code they actually used, not evidence that this submitted equation was implemented.

## 3. Additional submission-to-code inconsistencies to retain

**Baseline fairness and explanation — §§4.7, 7.2 and 7.5.** The submission expressly states that controllers share the same configuration space and that the rule baseline uses a dead-band and the same cooldown guard. Its overlapping-threshold explanation does not resolve the audit finding: `allow_multi_oai=False`, a two-option table compiler, and an instantiated selector whose `decide` method is unused in the evaluation loop are distinct mechanisms. Recover the generator responsible for the original 720-row rule trace before accepting the dead-band explanation. The quoted 90.47/87.13 Mbps thresholds were **not independently reproduced** in this reconciliation.

**Observation contents — §4.3, lines 253–262.** The submission describes instantaneous QoS and energy indicators in the observation, and forecast augmentation as an addition to the instantaneous schema. In the audited `_get_observation`, current QoS and power are not explicitly included. The archived settings disable capacity and power-gap features. The forecast-only branch supplies forecast[t+H], configuration/cooldown and optional calendar features; it does not retain actual current load, and its dynamics are disabled. History/hybrid contain actual load history, with hybrid adding the forecast. This is a concrete information-set discrepancy that should be included in the observation-schema recovery work. It was visible in the earlier source but was not separately highlighted in the original audit.

**Event counts — §7.2.** The submitted table and prose call 47 offline-optimal events “type switches.” The recovered planner summary separates **12 DPDK→USR + 12 USR→DPDK = 24 type switches**, and **12 scale-ups + 11 scale-downs = 23 scaling events**. The total is 47 reconfigurations. The source field named `switching_count` includes both categories; it must not be presented as type switches while also listing the 23 scaling events separately. These counts come from previously audited artifacts, not a new rollout.

**Single-run versus multi-seed descriptions — §7.5.** Statements about FF-PPO making 17 type switches/7 scaling events and recurrent PPO making 16/0 cannot describe the five-seed averages shown in §7.2. They require explicit run/figure identifiers. In particular, the same submission reports recurrent scaling of 5.8±3.6 across seeds, so a general statement that recurrent PPO does not exploit scaling is too broad. Explanations based on recurrent entropy collapse also need to distinguish the failed observation variants from history-augmented runs and account for the evaluator's discarded LSTM state.

**QoS claims — abstract, §§7.2, 7.4 and 7.11.** The stronger no-QoS-cost wording appears beyond the abstract. A non-significant difference at p=.5 is not evidence of equivalence or non-inferiority. The reported higher violation mean must be acknowledged; a no-increase claim would need a separate justified inferential design and supporting data.

**YouTube — §7.10.** Credit the submission for already stating that PPO energy is **0.73% higher** than rule-based control. This is not an omitted fact in the submitted results discussion. The phrase “margin ... narrows” should still distinguish loss of an energy advantage from continued lower SEC. Transfer claims remain limited to another service trace from the same dataset, city and period.

**Statistical interpretation — §7.11 and conclusion.** The submission already reports p=.125 and p=.0625 and acknowledges the .05 limitation with five observations. It overstates what follows when attributing non-significance to sample size “rather than” effect magnitude or lack of systematic effect. Observed direction consistency can be described; the small experiment cannot establish those stronger conclusions. The audited mismatch between the main-table baseline and the statistical-analysis baseline is unchanged.

**Physical scope — §5.1 and conclusion.** The submission clearly says evaluation is offline and that runtime reconfiguration/control-plane integration for service continuity remains future work. This agrees with the implementation's limitations and should be preserved. Neither the audit nor a reviewer response should imply the paper claimed a completed live actuator. Responsiveness within a slot remains unvalidated.

## 4. Reviewer mapping after correction

All 13 reviewer items remain relevant. The original matrix's §7.1–§7.11 numbering maps directly to the submitted version:

1. Evaluation Metrics
2. Overall Comparison Across Controllers
3. Energy Efficiency
4. QoS Compliance
5. Orchestration Behavior and Stability
6. Ablation on Observation Schemas and Policy Parameterization
7. Training Convergence Diagnostics
8. Sensitivity Analysis
9. Seed Robustness
10. Cross-Trace Generalization
11. Statistical Significance Analysis

Specific changes to the response plan:

- **R1.2:** The submission adds related-work discussion and citation keys absent from the repository TeX, including `leyva2023comnet` and `bellin2024nfvsdn`. Start from the submitted literature review. This comparison verifies textual presence, not the underlying cited claims or bibliography completeness.
- **R1.3:** Address observation timing and proposed online integration while acknowledging the submission's existing offline limitations and explicit startup measurements.
- **R1.4/R3.3/R3.8:** Use the explicit submitted launch-cost equation as the reference. The issue is implementation semantics and reward interpretation, not the obsolete repository table's 0.5/.2/.05 values.
- **R3.2:** Assess DPDK QoS on its own residuals; the power RMSE cannot establish QoS reliability.
- **R3.6:** Reorganize the actual 11 subsections; no dispute about the reviewer's count remains.
- **R3.7:** The cooldown equation itself is unchanged. The discarded-request behavior, execution-only costs and pre-increment indexing analysis remain applicable.

The original audit's remaining reviewer recommendations continue to apply, including first-use acronym checks. The same source-identification correction applies to its recovery plan: **do not ask again for the submitted manuscript**; it is now available. Exact-code and artifact provenance remain unresolved.

## 5. What this clarification changes operationally

The supplied TeX is now the authoritative manuscript reference. The repository TeX is retained only as a different local version. Existing numerical findings—720/721/840 mismatches, recurrent-state handling, selection leakage, checkpoint-normalizer pairing, forecast artifact mismatch, calendar/forecast alignment, SEC units and unresolved small power differences—remain supported independently of which manuscript version is read.

**The immediate technical next step remains the main-table provenance manifest**, now anchored to the submitted table at line 544 and its explicit reward equation. No fixes, new training or replacement results are authorized or implemented by this reconciliation.

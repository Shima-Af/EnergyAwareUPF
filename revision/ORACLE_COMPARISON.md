# Exact reward and energy oracles — validation only

**Complete.** The three-action energy minima are **87.626649 Wh (Netflix)** and **85.059553 Wh (YouTube)**. The point-QoS-constrained minima coincide with them. All Stage 7 learned policies and rules satisfy their matched oracle bounds. No policy, reward, twin, forecast or baseline was changed.

## Scope and contract

Work date: 23 October 2026. Branch `revision/comcom-clean-evaluation`. Both workloads use their existing Stage 7 **504 validation intervals**, 22 April 2019 00:00–27 April 05:45 (126 hours), source rows 2016–2519. The prefix-only reader parses at most rows 0–2519. No final-test targets or outcomes are read or evaluated by this task. Previous reports and the Stage 7 data-access addendum remain unchanged.

Twin identity: `80866f0adf81c3f144190fe49d8cf74e2d5eefe95c90b63a2924fb8da67c465e`. Outcomes come from the hash-verified Stage 7 cache of real frozen-twin inference at identical actual loads. Frozen model artifacts are hash-checked. No new surrogate fitting, forecast fitting, controller execution, checkpoint selection or training occurs. All 174 saved selected/endpoint controller trajectories were independently checked against authorized loads, timestamps, cached outcomes and shared actuation/reward accounting; this is accounting verification, not policy replay.

Initial state is eligible DPDK, `(action=0, elapsed=4)`. A change at t blocks further changes at t+1…t+3 and permits one at t+4. Blocked requests are discarded. The oracles emit admissible executed actions; a blocked request is redundant with holding and cannot improve either objective. All 504 intervals, including the last, are integrated; terminal cost is zero, with no requirement to return to DPDK.

**Physical energy accounting:** `E = Σ P_attributed_W × 0.25 h`, in Wh. No reward-only switch, launch or scale penalty is added to energy; no startup/transient energy is invented. “Physical” here distinguishes power integration from reward: it remains predicted process-attributed energy, not measured whole-server electricity.

The unchanged reward is `−100×SEC −30×max(0,.90−Q)` minus .030 per switch to DPDK, .012 per launched USR instance and .003 per within-USR scale-down. SEC is J/Mbit. Discounted return is `G = Σ .995^t × reward[t]`; undiscounted total reward is also retained in the machine-readable results.

## Exact dynamic programs

At each time t and actuator state s, enumerate all admissible actions and the shared `actuate` successor f(s,a). Energy uses `V[t,s] = min_a { E[t,a] + V[t+1,f(s,a)] }`. Reward uses the same recurrence with cost `−.995^t×reward[t,s,a]`, hence maximizes discounted return. The constrained energy recurrence removes any executed action with predicted Q<.90; an empty continuation is infeasible. No penalty relaxation, fabricated trajectory or relaxed objective is returned.

Each binary64 stage cost is converted exactly to an integer with a common power-of-two denominator. DP sums and compares integers, avoiding accumulation error and arbitrary “equal enough” tolerances. Equal primary objectives prefer fewer executed reconfigurations, then the lexicographically smallest action sequence. Reported totals are rounded once to float. Exactness is conditional on the represented twin/stage costs, not on uncertain physical measurements. Complexity is O(T×states×actions), with 15 states for the three-action/four-slot contract.

Binary means **{DPDK, USR(1)}**; ternary adds USR(2). All share the same initial state and transitions. No causal forecast or observation-schema restriction is imposed: each oracle knows the complete future validation traffic, equally across objectives.

## Oracle results

R = reward-optimal; E = energy-optimal; EQ = Q≥.90-constrained energy. Switches count type changes; up/down count within-USR scale events. The initial departure from DPDK counts as a reconfiguration.

| Workload | Actions | Oracle | Wh | Discounted return | Mean / min Q | Q<.90 | Switches | Scale up/down |
|---|---|---|---:|---:|---:|---:|---:|---:|
| netflix | binary | R | 88.376009 | -122.443718 | 0.968276 / 0.917614 | 0% | 11 | 0/0 |
| netflix | binary | E | 88.352944 | -122.463084 | 0.968693 / 0.917614 | 0% | 13 | 0/0 |
| netflix | binary | EQ | 88.352944 | -122.463084 | 0.968693 / 0.917614 | 0% | 13 | 0/0 |
| netflix | ternary | R | 87.630190 | -119.535813 | 0.964897 / 0.907887 | 0% | 11 | 0/0 |
| netflix | ternary | E | 87.626649 | -119.608691 | 0.965311 / 0.907887 | 0% | 13 | 0/0 |
| netflix | ternary | EQ | 87.626649 | -119.608691 | 0.965311 / 0.907887 | 0% | 13 | 0/0 |
| youtube | binary | R | 85.721153 | -122.932182 | 0.968819 / 0.916833 | 0% | 11 | 0/0 |
| youtube | binary | E | 85.721153 | -122.932182 | 0.968819 / 0.916833 | 0% | 11 | 0/0 |
| youtube | binary | EQ | 85.721153 | -122.932182 | 0.968819 / 0.916833 | 0% | 11 | 0/0 |
| youtube | ternary | R | 85.059553 | -119.583538 | 0.965207 / 0.906826 | 0% | 11 | 0/0 |
| youtube | ternary | E | 85.059553 | -119.583538 | 0.965207 / 0.906826 | 0% | 11 | 0/0 |
| youtube | ternary | EQ | 85.059553 | -119.583538 | 0.965207 / 0.906826 | 0% | 11 | 0/0 |

Both constraints are feasible on both action sets. Every available action at every actual validation load already has predicted Q≥.90 (1,512 action/load pairs per workload). Thus EQ and E have identical paths and Wh; this is a non-discriminating point-model constraint, not evidence of measured service safety. Synthetic tests also cover genuine infeasibility caused by cooldown despite a safe action existing at each slot.

Netflix reward-optimal energy exceeds its energy minimum by **0.003541 Wh** with three actions and **0.023065 Wh** with two. It makes fewer type switches (11 versus 13). YouTube happens to have the same reward- and energy-optimal path in each action set; that equality is data-specific, not an equivalence of objectives.

| Value of USR(2) availability | Binary E − ternary E (Wh) | Gain in optimal discounted return |
|---|---:|---:|
| netflix | 0.726294 | 2.907905 |
| youtube | 0.661600 | 3.348644 |

These quantify the best attainable value of an enlarged action set inside the twin. The three-action oracles use DPDK↔USR(2), with **zero within-USR scaling events**. USR(2) availability and within-USR scaling behavior are distinct; these bounds do not establish a causal benefit of learned scaling.

## Controller gaps

`ΔE = controller Wh − minimum Wh` and `ΔG = maximum discounted return − controller discounted return`. Smaller is closer to the appropriate bound. Energy gaps use E, reward gaps use R; no reward-oracle Wh is used as an energy lower bound. The same energy gaps apply to EQ for all current controllers because they meet the point constraint. In general an unsafe controller cannot be compared to a constrained energy bound as though it were feasible (its constrained gap is null).

All three-action learned runs, A rules and strong rule use ternary bounds. Historical COMCOM rule C uses its **binary** bounds. Static rows show gaps to the full-action ternary bound, not an oracle restricted to constant commands. Binary-only RL policies were not trained; an oracle action-set comparison is not a controlled learned-policy action ablation.

| Workload | Rule / static | Bound | Wh | ΔE Wh | ΔE % | ΔG |
|---|---|---|---:|---:|---:|---:|
| netflix | slices_extended_manual_26 | ternary | 89.341027 | 1.714377 | 1.956 | 2.968431 |
| netflix | slices_recalibrated_v2 | ternary | 88.199815 | 0.573166 | 0.654 | 1.509999 |
| netflix | strong_rule_stage4 | ternary | 88.163527 | 0.536878 | 0.613 | 1.359604 |
| netflix | comcom_historical | binary | 88.986442 | 0.633499 | 0.717 | 1.393417 |
| netflix | static_0 | ternary | 103.264594 | 15.637945 | 17.846 | 79.736853 |
| netflix | static_1 | ternary | 145.257591 | 57.630942 | 65.769 | 42.706758 |
| netflix | static_2 | ternary | 149.924831 | 62.298182 | 71.095 | 43.341129 |
| youtube | slices_extended_manual_26 | ternary | 86.035795 | 0.976241 | 1.148 | 2.244972 |
| youtube | slices_recalibrated_v2 | ternary | 85.233755 | 0.174202 | 0.205 | 0.307527 |
| youtube | strong_rule_stage4 | ternary | 85.155469 | 0.095916 | 0.113 | 0.220034 |
| youtube | comcom_historical | binary | 85.812688 | 0.091535 | 0.107 | 0.187693 |
| youtube | static_0 | ternary | 103.266538 | 18.206985 | 21.405 | 125.711214 |
| youtube | static_1 | ternary | 135.511312 | 50.451758 | 59.313 | 41.986051 |
| youtube | static_2 | ternary | 140.337860 | 55.278306 | 64.988 | 42.365612 |

Learned rows below average the five fixed seeds in each observation-schema cell. Policy and checkpoint choices remain frozen from Netflix validation. YouTube is transfer with target-specific forecast analytics, not a newly untouched end-to-end transfer experiment. Raw 300k endpoints remain separate from validation-selected checkpoints.

| Workload | Algorithm / schema | Selected Wh mean ± sample SD | Selected ΔE Wh | Selected ΔE % | Selected ΔG | Endpoint ΔE Wh |
|---|---|---:|---:|---:|---:|---:|
| netflix | ppo / instant | 88.877855 ± 0.371655 | 1.251206 | 1.428 | 2.686924 | 1.991594 |
| netflix | ppo / history | 88.193191 ± 0.177649 | 0.566542 | 0.647 | 0.963427 | 1.200541 |
| netflix | ppo / forecast | 89.005739 ± 0.578091 | 1.379089 | 1.574 | 2.924872 | 1.930740 |
| netflix | ppo / hybrid | 88.303945 ± 0.191569 | 0.677295 | 0.773 | 1.449632 | 1.256955 |
| netflix | recurrent_ppo / instant | 88.171001 ± 0.217464 | 0.544351 | 0.621 | 0.914129 | 0.887646 |
| netflix | recurrent_ppo / history | 88.019672 ± 0.045100 | 0.393023 | 0.449 | 0.662741 | 1.334134 |
| netflix | recurrent_ppo / forecast | 88.242386 ± 0.323306 | 0.615736 | 0.703 | 1.308816 | 0.969231 |
| netflix | recurrent_ppo / hybrid | 88.086207 ± 0.116988 | 0.459557 | 0.524 | 0.786741 | 1.175012 |
| youtube | ppo / instant | 86.563479 ± 0.697659 | 1.503926 | 1.768 | 2.428581 | 1.764227 |
| youtube | ppo / history | 87.576235 ± 1.245820 | 2.516682 | 2.959 | 2.867520 | 3.016230 |
| youtube | ppo / forecast | 86.339879 ± 0.614642 | 1.280326 | 1.505 | 2.596024 | 1.576760 |
| youtube | ppo / hybrid | 86.681175 ± 0.289108 | 1.621622 | 1.906 | 3.125142 | 2.059832 |
| youtube | recurrent_ppo / instant | 86.545713 ± 0.737488 | 1.486160 | 1.747 | 1.769046 | 1.262355 |
| youtube | recurrent_ppo / history | 86.327425 ± 0.428599 | 1.267872 | 1.491 | 1.828274 | 1.568240 |
| youtube | recurrent_ppo / forecast | 86.151247 ± 0.717986 | 1.091693 | 1.283 | 1.578524 | 0.864060 |
| youtube | recurrent_ppo / hybrid | 87.052841 ± 0.962422 | 1.993287 | 2.343 | 3.384243 | 1.244557 |

Per-seed selected and endpoint values, including all rule/static rows, are in [controller_gaps.csv](reports/oracles/controller_gaps.csv). These validation gaps are descriptive, not significance tests, convergence evidence or held-out generalization estimates. A small oracle gap also does not show that learning causes the improvement over a rule.

## Bounds, verification and preservation

**The reward-optimal oracle is an upper bound on discounted reward. The energy-optimal oracle is a lower bound on attributed Wh.** EQ is the energy lower bound only within its feasible point-QoS-constrained set. Each statement is conditional on the matched workload, initial state, action set, cooldown, horizon and frozen twin. Neither oracle is a deployable causal controller. Profiling uncertainty, false-safe QoS errors, per-direction load semantics, balanced USR(2), missing idle/transient measurements and process power attribution remain limitations; extra decimal places do not remove them.

**Tests: 45 passed in 6.85 seconds.** Twenty-four short-sequence cases enumerate every requested sequence (including blocked commands), covering both action sets, all three objectives and four fixed synthetic seeds. They compare exact rational objectives, tie-break counts and executed paths. Other checks cover reward-only costs versus Wh, t+4 cooldown, true infeasibility, strict near-equal energy ordering, deterministic reload of all 12 real-data solutions, every saved controller trajectory and bound, and final-target-poisoned prefix reading. No scientific fit or controller replay is performed by tests. The broad legacy suite was not invoked.

New files: `revision/oracles/{__init__,solver,run,report}.py`; `revision/tests/test_oracles.py` and `test_oracle_artifacts.py`; this report; the prior-file SHA snapshot; and `revision/reports/oracles/` (12 step CSVs, comparison JSON, 174-row gaps CSV, provenance, execution and test logs, integrity record and file manifest). [Complete file manifest](reports/oracles/file_manifest.json).

Evidence: [comparison](reports/oracles/comparison.json), [provenance](reports/oracles/provenance.json), [tests](reports/oracles/tests.log), [integrity](reports/oracles/integrity.json). All **5,203 pre-task scientific files** are hash-checked unchanged. Git tracked files, branch and HEAD remain unchanged; no commit or push. No manuscript edit, default-model promotion, reward change, baseline change or training. **Stopped after this oracle extension.**

Reproduction: in a fresh output workspace with the pinned Stage 7 artifacts, run `.venv/bin/python -m revision.oracles.run`, then `.venv/bin/python -m revision.oracles.report`. Writers refuse to overwrite existing outputs. Tests run with `.venv/bin/python -m pytest -q revision/tests/test_oracles.py revision/tests/test_oracle_artifacts.py revision/tests/test_forecast_service.py::test_parser_never_converts_final_test_targets revision/tests/test_forecast_service.py::test_prefix_reader_prohibits_final_test_and_uses_raw_fields -p no:cacheprovider`.

"""Refresh the required Stage 10 report from recorded implementation evidence."""
import json
from pathlib import Path
from revision.evaluation.twin import ROOT
from .independent_fits import OUT,MODELS


def main():
    read=lambda name:json.loads((OUT/name).read_text())
    counters=read('counter_feasibility_summary.json')
    quality=read('partial_kpi_threshold_details.json')
    forecasts=read('rebuilt_forecast_metrics.json')
    reward=read('reward_calibration_v2.json')
    lines=['# Corrected COMCOM campaign — Stage 10 implementation and remaining loss-target blocker', '',
      '**Status: substantial implementation and independent fitting completed; the full '
      'controller campaign is not executed.** The approved four-KPI policy is unchanged. '
      'Active tester flows and DN-received-throughput direction have been reconciled. '
      'The original exports still do not authenticate the packet population and '
      'sequence/time attribution of `Lost Packets`; therefore no physical loss labels, '
      'loss predictor, real four-KPI controller replay or PPO training has been fabricated.', '',
      'This continuation implements the parts independent of that blocker: transition '
      'kernel, causal Gym environment, balanced KPI adapter, seven baseline adapters, '
      'three exact transition-aware oracles, a gated definitive 40-run harness, one '
      'training-calibrated reward, **six KPI model fits and twenty forecasting fits**. '
      'No additional QoS formulation or model-family search was conducted.', '',
      'All earlier experiments remain intact. The preceding report is archived at '
      '[report_before_continuation.md](reports/stage10/completion/report_before_continuation.md). '
      'Work remains on `revision/comcom-clean-evaluation`, HEAD '
      '`c0734715418d3f6d10785e5174aebb4bac24d0da`. No commit or push.', '',
      '## 1. Fixed policy and authoritative references', '',
      '`Q=min(T,D,J,L)`, target .90: received/offered satisfaction ≥90%, delay '
      '≤4 ms, jitter ≤1 µs, verified loss fraction ≤.005. The component normalizations '
      'remain those recorded in [policy_v1.json](stage10/policy_v1.json). These are '
      'study-policy budgets, not external 5QI guarantees. No five-packets-per-three-seconds '
      'substitution, weighted average or old composite-QoS surrogate is used.', '',
      'The submitted [MAPPO PDF](reference/1571355221%20paper.pdf) and '
      '[final thesis PDF](reference/Thesis_ShimaAfshar_Final.pdf), especially §5.6, '
      'remain the primary methodological references. Their identities are pinned in '
      '[the reference manifest](reports/stage10/reconciliation_v2/authoritative_pdf_manifest.json). '
      'The earlier draft `paper.tex` files are not substituted for the submitted PDF.', '',
      '## 2. Active flows and throughput mapping — resolved within the tester scope', '',
      'All **220 original trial configurations** were checked for enabled range, enabled '
      'objective and selected objective type, rather than reading inactive JSON branches. '
      'Every trial has one enabled StatelessUDP flow in each direction, one UE/range, '
      'flow ID 1 and 1,250-byte payloads: DN transmits downlink; UE/NG-RAN transmits '
      'uplink. For all **440 directional checks**, configured packet rate ×payload '
      'bytes×8/10⁶ equals the processed per-direction offered Mbps.', '',
      'Consequently `DN_Rx_Mbits/s` is **uplink received throughput at the DN endpoint** '
      'for this configuration, not delivered downlink. Its raw `Kbits Rx/s` conversion '
      'was already matched; no repeated unit investigation was needed. Only timestamp-exact '
      'raw matches enter the new throughput fits: 4,572/4,594 DPDK and 4,554/4,564 USR '
      'rows are available across the five sweeps; unmatched receive measurements are '
      'excluded with source indices recorded, never filled. Configured demand is the '
      'denominator, not measured injected aggregate bytes or application goodput.', '',
      '[Active flows](reports/stage10/completion/active_flows.json) and '
      '[offer mapping](reports/stage10/completion/active_offer_mapping.csv) pin the '
      'evidence. This direction is also consistent with the primary '
      '[6G-SANDBOX LoadCore measurement report](https://6g-sandbox.eu/wp-content/uploads/2025/04/6G-SANDBOX_D5.3_V1.pdf), '
      '§2.3. That reference supports endpoint orientation; it does not define this '
      'export’s `Lost Packets` semantics.', '',
      'The study thus jointly evaluates the approved DN-uplink receive satisfaction '
      'and the recorded downlink/RAN delay/jitter fields. No unidirectional end-to-end '
      'delivery or aggregate DL+UL capacity claim is introduced. The workload scalar '
      'retains the historical mapping to per-direction offered study load.', '',
      '## 3. Same-interval reconstruction — numerical candidates, no authenticated loss label', '',
      'The already-established rate-plus-count error was not investigated again. '
      'The new work differences the **original cumulative counters**, within each '
      'trial, on exact common timestamps. It considers only the two evidence-supported '
      'count denominators: receiver `ΔRAN_Rx+ΔLost`, and downlink sender `ΔDN_Tx`. '
      '`RAN_Tx` is uplink transmission and is not silently reused as a downlink '
      'denominator. Rates are not added to interval counts.', '',
      '| Processed-row feasibility | DPDK | USR |',
      '|---|---:|---:|',
      f"| Rows matched to a raw snapshot | {counters['dpdk']['matched_processed_rows']} | {counters['usr']['matched_processed_rows']} |",
      f"| Numerically valid receiver-based candidates | {counters['dpdk']['valid_numeric_receiver_candidates']} | {counters['usr']['valid_numeric_receiver_candidates']} |",
      f"| Numerically valid sender-based candidates | {counters['dpdk']['valid_numeric_sender_candidates']} | {counters['usr']['valid_numeric_sender_candidates']} |",
      f"| `ΔDN_Tx != ΔRAN_Rx+ΔLost`, among comparable rows | {counters['dpdk']['nonzero_conservation_residuals']}/{counters['dpdk']['conservation_comparable']} | {counters['usr']['nonzero_conservation_residuals']}/{counters['usr']['conservation_comparable']} |",
      f"| Maximum absolute count disagreement | {counters['dpdk']['max_abs_conservation_residual']:,.0f} | {counters['usr']['max_abs_conservation_residual']:,.0f} |",
      '| Physical loss labels released | **0** | **0** |', '',
      'These are not competing QoS formulations and neither candidate is selected '
      'on fit quality. They expose the missing measurement definition. Common '
      'timestamps do not establish common packet cohorts: queued/in-flight traffic, '
      'sequence-gap detection, late arrivals, metric update timing and resets can '
      'differ. The available columns do not distinguish these mechanisms. Dividing '
      'by receiver-plus-lost mechanically produces a bounded fraction but does '
      'not establish that `Lost Packets` counts the same expected population.', '',
      'Explicit handling in [intervals.py](stage10/completion/intervals.py):', '',
      '- First raw sample: unknown preceding interval. A first *retained processed* '
      'sample may use its actual preceding raw snapshot if present; no forced zero.',
      '- Missing/nonfinite counters or absent/non-three-second predecessor: unknown; '
      'no forward-fill, nearest timestamp, or rate×3 repair.',
      '- Any counter decrease/reset: unknown for that interval, never clipped to zero. '
      'Subsequent numerical differences do not authenticate a hidden sequence epoch.',
      '- Zero expected packet population: fraction undefined, including 0/0. '
      'Lost greater than expected: inconsistent, not clipped to one.',
      '- A known zero counter increment remains zero only as an observed numeric '
      'increment. It does not turn an unverified physical fraction into zero.', '',
      '[interval_feasibility.csv](reports/stage10/completion/interval_feasibility.csv) '
      'retains all raw timestamps, predecessor times, deltas, candidate statuses '
      'and processed-row mappings. **Every `physical_loss_fraction` is null** and '
      'every population-authentication flag is false. No corrected-label dataset '
      'has been misrepresented as authenticated.', '',
      '**Exact missing evidence:** the metric definition for the original LoadCore '
      'version’s NG-RAN `Lost Packets`, including whether it is sequence-gap versus '
      'finalized loss, which data/bearer/flow population it covers, late/out-of-order '
      'arrival adjustments, timestamp attribution and counter-reset epochs; plus '
      'the corresponding same-population expected/received counter definition. '
      'Alternatively, synchronized same-flow sender/receiver packet traces with '
      'sequence IDs and a stated delivery/lateness rule could establish the cohorts. '
      'Configuration direction alone cannot supply those facts. Inspection of all 220 original export ZIPs and 660 nested ZIPs found no .pcap/.pcapng/.cap packet traces; see [capture inventory](reports/stage10/completion/packet_trace_inventory.json). This is the remaining '
      'material blocker, not another QoS-policy choice or a missing arithmetic factor.', '',
      '## 4. Independent KPI fits completed and retained', '',
      'One prespecified nonnegative family, **64-tree RandomForestRegressor**, depth '
      '6, minimum leaf 10, seed 42, was fitted once per available mode/target. '
      'There is no architecture, seed or hyperparameter selection. Positive target '
      'leaf averages guarantee nonnegative predictions without output repair. Input '
      'is log1p(per-instance load); all fitted parameters use sweeps 1–3 only. '
      'Sweep 4 is validation and sweep 5 internal evaluation, with no result-driven '
      'refitting. These remain exposed same-campaign groups, not fresh physical tests.', '',
      'Four delay/jitter fits were followed by exactly two throughput fits after '
      'active-offer provenance was resolved. The latter use the same frozen family, '
      'not a performance-driven search expansion. Six models are saved under '
      '`revision/models/stage10/kpi/`. Loss has **no model** and no fallback.', '',
      '| Mode / KPI | Role | n | MAE | RMSE | R² | False-safe / predicted-safe | False-unsafe |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for q in quality:
        if q['role']=='train':continue
        lines.append(f"| {q['mode']} / {q['target']} | {q['role']} | {q['n']} | {q['mae']:.6g} | {q['rmse']:.6g} | {q['r2']:.6g} | {q['false_safe']}/{q['predicted_safe']} | {q['false_unsafe']} |")
    lines += ['', 'Units are milliseconds, microseconds and throughput fraction respectively. '
      'The poor DPDK delay fit (validation R²≈−151), substantial throughput false-safety '
      'and USR jitter’s zero predicted-safe validation coverage are preserved, '
      '**not rescued by another search**. A zero numerator with zero predicted-safe '
      'denominator is undefined risk, not a successful safety model. Independently '
      'predicted means remove weighted compensation but do not remove within-load '
      'variation or guarantee per-observation budgets.', '',
      '[Threshold details](reports/stage10/completion/partial_kpi_threshold_details.json) '
      'include bias, false-safe denominators/severity and a descriptive ±10%-of-budget '
      'measured band (not a new compliance threshold). '
      '[Per-load/sweep residuals](reports/stage10/completion/kpi_errors_by_load_sweep.csv), '
      '[all predictions](reports/stage10/completion/all_partial_kpi_predictions.csv) '
      'and [domain checks](reports/stage10/completion/partial_kpi_domain_checks.json) '
      'retain every available evaluation. No model is promoted to the default twin.', '',
      '## 5. Transition-aware environment, reward and oracle implementation', '',
      '[engine.py](stage10/completion/engine.py) is the common outcome kernel. '
      'It applies eligible DPDK initialization, three configurations, the original '
      'four-slot cooldown, discarded blocked requests, time-weighted outgoing/incoming '
      'operational Wh and the thesis burst model exactly once per launch. '
      'Strict per-KPI compliance and maximum severity consider **every serving '
      'phase**, not just the incoming configuration. Unsafe serving seconds, '
      'startup exposure, type switches, within-USR scale events, launches and '
      'blocked commands remain separate fields.', '',
      'Declared replica model: serial USR launches, 3.3 s each, with barrier '
      'cutover after all requested replicas are ready. USR(1)→USR(2) keeps the '
      'existing replica serving for 3.3 s; DPDK→USR(2) retains DPDK for 6.6 s. '
      'First-ready replica idle overlap during the second launch is included '
      'using the explicit portable idle parameter. Scale-down is instantaneous '
      'removal plus balanced redistribution at the boundary, **an unmeasured '
      'modeling assumption**, not a measured drain or outage. No ongoing prewarm '
      'or session-transfer mechanism is invented.', '',
      'Thesis idle parameters .82/.015 W and burst ratios 1.73/.37 are explicitly '
      'portable assumptions, not authenticated COMCOM idle/transient measurements. '
      'The interval-mean load is held constant for modeled sub-interval phases; '
      'there is no new sub-slot telemetry or physical service-continuity claim.', '',
      'One prepared reward is fixed:', '',
      '`r = −[ E_total_Wh/Emax_train + 3×strict_violation '
      '+ severity/(1+severity) + .01×executed_change ]`.', '',
      f"`Emax_train={reward['energy_scale_wh']:.15g} Wh` comes from **69,120** "
      'eligible transition/power combinations using only the four Netflix training '
      'scenarios (1,920 slots×4×9). It uses the unchanged frozen power models, '
      'including activation/ready-idle energy, and no controller ranking or '
      'validation data. SEC is reported but is **not** the energy reward denominator. '
      'The strict weight 3 is derived from the normalized component bounds, '
      'ceil(1+1+.01), not searched. Within the training energy bound, safe immediate '
      'cost ≤1.01 and strict-unsafe cost ≥3; low offered load cannot overwhelm this '
      'preference through SEC. This is not a whole-horizon safety guarantee or an '
      'assurance for unseen energies. The .01 stability preference is dimensionless, '
      'not an additional physical activation-energy charge.', '',
      '[Reward calibration v2](reports/stage10/completion/reward_calibration_v2.json) '
      'includes the serial-launch ready-idle overlap. The initial calculation is '
      'retained; coefficients and the maximum energy scale are unchanged. No '
      'scientific fit was repeated for that accounting correction. Full four-KPI '
      'reward-magnitude diagnostics remain gated by the missing physical loss target.', '',
      '[environment.py](stage10/completion/environment.py) implements causal '
      'start-of-interval observations, missing-previous-outcome masks, W96 history '
      'and all four schemas (13/109/14/110 features). Only completed telemetry and '
      'the new issued forecast enter observations. [provider.py](stage10/completion/provider.py) '
      'implements balanced USR load division, summed power and unchanged per-instance '
      'KPI assumptions, with artifact checks and no missing-loss fallback.', '',
      '[oracles.py](stage10/completion/oracles.py) implements exact discounted-reward, '
      'physical-energy and strict-QoS-constrained energy bounds using the **same '
      'transition kernel**, binary or ternary actions, integerized cost comparisons '
      'and fewer-change tie-breaks. Infeasibility is explicit. Brute-force synthetic '
      'enumeration verifies them; **no real-data oracle result is fabricated** '
      'without an authenticated full KPI provider.', '',
      '## 6. Rebuilt causal forecasts and baseline adapters', '',
      'The exact frozen Stage 9 transforms are used: original ×1, low ×.1, '
      'moderate overload ×2, and mixed daily factors [.1,1,4], preserving source '
      'timestamps and the authorized 60/15/25 split. Derived scenarios are not '
      'independent measured traffic. Training scores 96–2015; validation 2016–2519.', '',
      'Twenty **newly fitted** Poisson boosted models: four scenarios×five Netflix '
      'past prefixes 672/1008/1344/1680/2016. The family/settings were fixed before '
      'these fits using the already completed forecasting work; no candidate '
      'comparison or validation-driven family selection occurs. This is a fixed-family '
      'protocol, explicitly different from Stage 6’s adaptive prequential family '
      'choice. Each fit uses only its available prefix and prefix-fitted scale. '
      'No later weights are backfilled into earlier observations. The first '
      '576/1,920 training decisions use declared persistence cold start; the next '
      '1,344 use earlier-prefix models. All 504 validation forecasts use the '
      '2016-prefix model. No old forecast outputs are reused.', '',
      'YouTube uses the Netflix-trained scenario providers on its completed '
      'history: **zero YouTube parameter fits**, unlike earlier target-specific '
      'analytics. No final-test forecast targets were parsed. There are '
      '11,712 issued training/validation forecast rows across both workloads.', '',
      '| Scenario | Netflix validation MAE / RMSE Mbps | YouTube transfer MAE / RMSE Mbps |',
      '|---|---:|---:|']
    for scenario in ['original','low','moderate_overload','mixed']:
        a=next(r for r in forecasts if (r['workload'],r['scenario'],r['role'])==('netflix',scenario,'validation'))
        b=next(r for r in forecasts if (r['workload'],r['scenario'],r['role'])==('youtube',scenario,'validation'))
        lines.append(f"| {scenario} | {a['mae']:.6f} / {a['rmse']:.6f} | {b['mae']:.6f} / {b['rmse']:.6f} |")
    lines += ['', '[Ledgers](reports/stage10/completion/rebuilt_forecasts.csv), '
      '[forecast metrics](reports/stage10/completion/rebuilt_forecast_metrics.json) '
      'and `models/stage10/forecast/` contain provenance and hashes. '
      '[data.py](stage10/completion/data.py) binds these ledgers to common horizons, '
      'rejects final targets and does not retain actual targets in the provider.', '',
      '[baselines.py](stage10/completion/baselines.py) provides three static adapters, '
      'the binary historical COMCOM ranker, extended SLICES manual, independently '
      'training-recalibrated SLICES, and the strong diagnostic rule. A/C utilities '
      'reuse their preserved design principles with **new min-score outputs**; '
      'strong-rule feasibility and energy account for transitions, retaining its '
      '2% deadband. All requests execute through the common Engine. Actual '
      'SLICES regions must be rebuilt from authenticated KPI curves; old crossover '
      'numbers are not substituted. The manual 26-Mbps margin remains separate '
      'from fixed-k=1 training-calibration MAE margins: original 14.349328, '
      'low 1.434933, moderate 28.698655, mixed 29.362579 Mbps. These are not '
      'confidence bounds or controller-tuned parameters.', '',
      '## 7. Gated definitive campaign harness and verification', '',
      '[train.py](stage10/completion/train.py) prescribes exactly 2 algorithms×4 '
      'schemas×5 seeds [42,123,456,789,1024], 300,000 interactions each. Four '
      'collectors receive one scenario each, 75,000 interactions/scenario/run; '
      'they are not independent datasets. Existing PPO settings and shared '
      '128-unit recurrent LSTM are retained. It reuses the previously verified '
      'exact-budget rollout boundary, saves policy/normalizer pairs only after '
      'updates, preserves all three RNG streams during checkpoint validation, '
      'keeps recurrent state through each horizon, and never updates normalization '
      'from validation. Optimizer and component-level episode logs are retained.', '',
      'Selection is total Netflix validation reward summed across the four equal '
      '504-slot scenarios, with earlier step as tie-breaker; scenario metrics '
      'remain separate. YouTube cannot select checkpoints. No retry or unfavorable '
      'seed removal is automatic. Artifact/implementation hashes and **all eight '
      'validated KPI manifests** are mandatory before any run directory or '
      'scientific PPO object is created. This gate currently fails as intended.', '',
      '**53 tests passed in 4.43 seconds**, including exhaustive transition-oracle '
      'enumeration, true constraint infeasibility, unknown/zero/reset interval '
      'handling, balanced-model/domain checks, train-only energy scaling, causal '
      'observations and forecast targets, paired checkpoint/normalizer behavior '
      'with synthetic models, recurrent state persistence, exception-safe RNG '
      'isolation, and final-target-poisoned prefix parsing. No policy gradient '
      'updates occur in tests. These are integration-kernel tests, **not a complete '
      'real-twin campaign preflight**. [Sealed log](reports/stage10/completion/sealed_tests.log).', '',
      '## 8. What remains and the exact release condition', '',
      'The throughput direction/offer issue is resolved in the stated tester scope. '
      'No new budget or QoS-formulation decision is requested. A documented loss '
      'counter/cohort definition or sequence-resolved trace is still needed to '
      'authenticate a corrected three-second fraction. Once available, release '
      'versioned corrected labels with explicit unknown/reset masks, fit the two '
      'remaining fixed-family loss models, evaluate joint/per-KPI reliability, '
      'finish real-provider reward diagnostics, rebuild actual rule regions and '
      'run the complete real integration preflight. The six existing KPI models '
      'and twenty forecasters are retained, not searched again to rescue results.', '',
      'The already authorized 40-run campaign, all baseline/oracle comparisons '
      'and the **once-only final-test evaluation** can then proceed after the '
      'full methodology/selection freeze. No final-test reader is enabled merely '
      'by the partial-kernel tests. No full-controller numerical replacements, '
      'convergence/seed conclusions, comparative claim confirmations or manuscript '
      'edits exist yet. Earlier claims remain as qualified by Stages 2–9.', '',
      'Preservation: all **5,298 pre-continuation scientific files** are checked; '
      'only this intentionally updated report changes and its previous content '
      'is archived verbatim. Stages 2–9, original measurements/models/results, '
      'and the submitted manuscript remain unchanged. '
      '[Integrity](reports/stage10/completion/integrity.json) and '
      '[file manifest](reports/stage10/completion/file_manifest.json) list every '
      'new implementation, model, forecast, metric and log. Scientific new fits '
      'are **6 KPI +20 forecast; 0 PPO/0 recurrent PPO**. No loss value was '
      'invented or silently replaced by zero; no commit or push.', '',
      'Reproduction in a fresh output namespace: run modules '
      '`revision.stage10.completion.counter_audit`, `independent_fits`, '
      '`throughput_fit`, `verify_models`, `calibrate_energy`, then this '
      '`write_report` module. Limit CPU threads as in the saved execution logs. '
      'Scientific artifact writers refuse overwrite. Use the three '
      '`revision/tests/test_stage10_*.py` files plus the two prefix-reader '
      'tests recorded in the test log. External measurement ZIPs still need '
      'their own backup; manifest hashes are not copies.', '']
    (ROOT/'revision/FINAL_COMCOM_CORRECTED_CAMPAIGN.md').write_text('\n'.join(lines))


if __name__=='__main__':main()

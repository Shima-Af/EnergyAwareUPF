"""Consolidate frozen full-factorial results without selecting on transfer outcomes."""
import collections,json,statistics
from .common import CFG,ROOT,OUT,dump


def main():
    comparison=json.loads((OUT/'comparison.json').read_text())['workloads'];families=json.loads((OUT/'family_summary.json').read_text())
    effects=json.loads((OUT/'paired_effects.json').read_text());convergence=json.loads((OUT/'convergence.json').read_text())
    costs=json.loads((OUT/'computational_cost.json').read_text());optimizer=json.loads((OUT/'optimizer_summary.json').read_text())
    old=json.loads((ROOT/'revision/reports/stage5/comparison.json').read_text())['controllers']
    text='''# Final observation-schema ablation — COMCOM-D-26-01061

**Stage7 complete, with a disclosed preflight data-access exception (§9).**
Improved external forecasting **did not lower selected mean Netflix energy when
added to history for either PPO family**. Recurrent PPO+history has the lowest
Netflix selected mean; PPO+history provides the cheaper primary comparator.
No primary paired contrast is significant after the prespecified adjustment.
On YouTube, every selected learned policy uses more energy than the strong rule.
These are reused-validation, frozen-twin results, not final generalization or
physical QoS evidence. No model/configuration is automatically promoted.

Work date:23 October2026. Branch `revision/comcom-clean-evaluation`; HEAD
`c0734715418d3f6d10785e5174aebb4bac24d0da`. All40 scientific fits completed without
failure/retry: **12000000 interactions, exactly300000/run**, across2 algorithms,
4 schemas and5 seeds. All560 checkpoint/normalizer pairs, optimizer/episode logs,
selected models and exact endpoints are preserved. Forecasts, twin, reward and
historical artifacts were not changed. No manuscript edit, commit or push occurred.
No final-test performance was evaluated or used for selection. The literal
no-target-access restriction was breached by a legacy preflight reader; that
exception is described explicitly below rather than hidden by the prefix-only
scientific pipeline.

## 1. Observation and timing audit

The submitted §§4.3/6/7.6 describe instantaneous, history-augmented,
forecast-augmented and hybrid schemas. Historical `src/environment.py` replaces
rather than consistently augments some base fields and uses current-interval
load/next-index forecasts. The approved corrected start-of-interval convention
requires **completed** telemetry: at t, use load[t−1], outcome[t−1] and a forecast
for t, then choose/guard/execute the action before scoring actual load[t].
No schema receives realized target-interval load or QoS before choosing its action.

| Schema | Common features | Traffic context added | Dimension |
|---|---:|---|---:|
|instant|13|None|13|
|history|13|96 completed loads, t−96…t−1|109|
|forecast|13|One forecast targeting t, H=1|14|
|hybrid|13|Both the96 completed loads and forecast|110|

Common features, **in this exact order**, are `completed_load`, `previous_qos`,
`previous_power`, `previous_sec`, `previous_valid`, `configuration_dpdk`,
`configuration_usr1`, `configuration_usr2`, `cooldown`, `time_sin`, `time_cos`,
`weekday_sin`, `weekday_cos`. Load/history/forecast divide by400; previous power
is attributed W, SEC is J/Mbit, cooldown is elapsed_slots/4. At reset previous
outcome fields are0 with valid mask0, configuration is eligible DPDK. The four
calendar coordinates use the known decision timestamp and are common even to
instantaneous control.

History appends `completed_history_0`…`completed_history_95`, oldest first;
forecast appends `causal_target_forecast` last. The most recent history value
intentionally duplicates common completed load. [Exact definition](stage7/OBSERVATION_SPEC.md)
and [machine-readable order](reports/stage7/prepared.json) specify every field.
Forecast-only means **common telemetry plus forecast**, not replacement of load.

The manuscript specifies a96-sample forecast input and symbolic policy W. This
stage explicitly implements the user's W=96 policy history. Stage4 actually used
95 history values and109 hybrid inputs; its109-dimensional hybrid cannot be
silently loaded as Stage7's109-dimensional history. Protocol/schema/feature
identities are checked when loading every bundle. Stage4 comparisons also change
forecast selection/cold starts, so they cannot isolate forecast-quality effects.
Within Stage7, all four schemas preserve the same common information and horizons;
parameter counts change with input dimension and are reported.

## 2. Frozen experiment and forecasting provenance

[Protocol](stage7/protocol.json) and [pre-fit freeze](reports/stage7/protocol_frozen.json)
prespecified all40 jobs and five primary energy contrasts. Each algorithm/schema
uses seeds42,123,456,789,1024, four collectors, n_steps1024, batch64, ten optimizer
epochs, lr.0001, gamma.995, GAE.90, entropy.15, clip.20 and target-KL.03.
Recurrent PPO uses shared LSTM128 without a separate critic LSTM. Default PPO
value/gradient settings remain unchanged. Interactions are matched; parameter
counts, recurrent sequence work and early-KL stopping need not match CPU work.

The unchanged exact-budget implementation uses a final248-step rollout/collector
(992 interactions), trains the tail normally and stops at300000. Fourteen paired
post-update checkpoints include every24576 interactions,32768 and300000.
Checkpoint selection maximizes **Netflix validation total reward**, with earlier
step breaking ties. Observation normalization is trained only on Netflix and saved
with each checkpoint; validation/transfer moments remain frozen and rewards
unnormalized. Recurrence persists through504 decisions and resets only per episode.
Exception-safe checkpoint capture restores Python, NumPy and CPU Torch RNG states.

Each training episode scores1920 intervals after the common96-slot context. All
controllers score the identical504 Netflix validation timestamps,22 April2019
00:00–27 April05:45 (126 hours), and the corresponding504 YouTube timestamps.
Both source readers stop at2520 and validate Stage6 authorized-prefix fingerprints.
YouTube policy scoring is gated until **all40 Netflix selections are frozen**;
none of its outcomes chooses a checkpoint. Its existing validation role is now
explicitly authorized for policy transfer in this new adapter; the older generic
final-report-only transfer gate remains unchanged. Shared `observe`, `actuate`
and `account` semantics are reused, with Netflix equivalence tested.

No forecasting model was fitted. Netflix training consumes the exact frozen causal
prequential ledger: **912 persistence rows (47.5%) and1008 boosting rows**. Of
these,576 are minimum-history cold start and336 the first comparative block;
there is no retrospective winner backfilling. Each validation workload uses its
own frozen full-prefix Poisson boosting service. YouTube-specific external
analytics is not required to transfer from Netflix. Policy transfer therefore
means **Netflix policy/normalizer with target-workload forecasting**, not an
unseen end-to-end zero-training pipeline.

The twin identity remains
`80866f0adf81c3f144190fe49d8cf74e2d5eefe95c90b63a2924fb8da67c465e`.
The exact scalar-inference cache is an optimization, checked against the frozen
twin and pinned before training, not a new fitted outcome model. Actions remain
DPDK/USR1/USR2, initial eligible DPDK, changes at t permit another at t+4 and
blocked commands are discarded. Reward remains −100×SEC −30×score shortfall
−.030 switch-to-DPDK −.012 per USR launch −.003 within-USR scale-down.

## 3. Primary Netflix and transfer comparisons

Energy is process-attributed Wh = sum(P_W×.25h); mean slot SEC is J/Mbit.
Aggregate SEC is separately retained in every seed's full metrics. Dispersion is
sample SD across five training seeds, conditional on one fixed workload/twin.
All eight cells and unfavorable results are reported.

| Workload | Algorithm/schema | Selected energy mean±SD Wh | 300k endpoint mean±SD Wh | Mean SEC | Selected reward mean±SD | Energy wins vs strong rule |
|---|---|---:|---:|---:|---:|---:|
'''
    for workload in ['netflix','youtube']:
        for key,m in families[workload].items():
            text+=f"| {workload} | {key} | {m['energy_wh_mean']:.6f} ± {m['energy_wh_sample_sd']:.6f} | {m['endpoint_energy_mean_wh']:.6f} ± {m['endpoint_energy_sample_sd_wh']:.6f} | {m['mean_sec_j_per_mbit_mean']:.8f} | {m['total_reward_mean']:.6f} ± {m['total_reward_sample_sd']:.6f} | {m['energy_wins_vs_strong']}/5 |\n"
    text+='''
Adding forecasts to history raises Netflix selected mean energy by **.110753 Wh
for PPO** and **.066535 Wh for Recurrent PPO**; only2/5 seed blocks improve in
each case. At the raw300k endpoints the corresponding mean effects are+.056415
and−.159123 Wh, respectively: endpoint behavior is mixed, not an unqualified
forecast benefit. Forecast-only versus instantaneous also raises selected means
by.127883/.071385 Wh. Better forecasting accuracy does not automatically yield
better policy optimization or decisions under this budget.

With improved forecasting, recurrence's selected mean energy advantage over PPO
is.217738 Wh in hybrid, versus.173519 Wh with history. The interaction is
−.044219 Wh, with wide uncertainty. **There is no evidence here that improved
forecasting removes the value of recurrence**, nor a noninferiority/equivalence
result proving it can be dropped. Common clocks/last telemetry already convey
predictive information; all forecast-containing schemas also have slightly
different input dimensions/parameter initializations. No mechanism is assigned
without a further controlled test.

On YouTube, adding forecasts improves PPO relative to its weaker history transfer
mean by.895060 Wh, while worsening recurrent history transfer by.725415 Wh.
Recurrent forecast-only has the lowest transfer mean among learned cells, but
**it is not selected/promoted on that basis**. Every selected learned seed uses
more energy than the strong rule. Accurate target-specific analytics therefore
does not establish policy-transfer superiority. Exact endpoints sometimes improve
transfer despite worse Netflix validation; both are retained without reselection.

### Prespecified paired analysis

Δ means left minus right; negative energy favors the left. Five independent seed
labels form prespecified blocks across conditions. Input/algorithm changes do
not produce identical RNG draws; the test assumes symmetric/exchangeable paired
differences. Intervals are descriptive95% Student-t intervals with n5, conditional
on a normal-difference approximation and this workload, not on surrogate accuracy,
traffic periods or physical campaigns. No practical equivalence margin exists.

Five primary selected-Netflix energy tests (four pairs plus interaction) use the
exact two-sided mean sign-flip procedure with Holm correction. With five blocks,
the theoretical minimum unadjusted two-sided p is.0625; lack of significance is
not evidence of equality. SEC/reward, endpoints, secondary pairs and transfer
contrasts are descriptive and are not additional confirmatory tests.

| Netflix contrast | Δ Wh | Descriptive95% interval | Left lower-energy seeds | Exact p | Holm p |
|---|---:|---|---:|---:|---:|
'''
    for key,m in effects['netflix'].items():
        if m['primary_test']:
            text+=f"| {key} | {m['mean_difference']:+.6f} | [{m['mean_difference_t95_lower']:+.6f}, {m['mean_difference_t95_upper']:+.6f}] | {m['energy_wins_negative']}/5 | {m['exact_two_sided_sign_flip_p']:.4f} | {m['holm_adjusted_p']:.4f} |\n"
    text+='''
All primary intervals include0; adjusted p≥.625. Negative/mixed results have the
same reporting status as favorable means. [All seed-block differences](reports/stage7/paired_seed_differences.csv)
and [SEC/reward/endpoint/transfer effects](reports/stage7/paired_effects.json)
retain every planned comparison. No slots or deterministic evaluation repeats
are treated as independent replicates.

## 4. Frozen baseline families

Four dynamic versions represent three families: original SLICES26, separately
versioned `slices_recalibrated_v2`, immutable Stage4 strong rule and historical
COMCOM binary ranker. The three static actions are additional references.
SLICES utilities/regions and overlap-resolution are unchanged; v2 changes only
its margin to the exact Stage6 training-calibration MAE:14.268830938789403 Mbps
Netflix and8.353153828365365 Mbps YouTube (k=1). No margin was optimized on PPO,
validation energy or transfer outcomes. The original26-Mbps files/results remain
unchanged. B retains feasibility preference and2% power deadband; C retains mixed
.3/.7 utility scales, binary availability and no active plotted23-Mbps margin.
All execution/reward/event accounting is common; launch costs are not physical Wh.

| Workload | Baseline | Wh | Mean SEC | Min Q | Switches | Scale up/down | Blocked | D/USR1/USR2 slots | Reward | Reconfiguration penalty |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|
'''
    for workload in ['netflix','youtube']:
        rows=comparison[workload]['controllers_selected']
        for name,m in rows.items():
            if m.get('family')!='baseline':continue
            text+=f"| {workload} | {name} | {m['energy_wh']:.6f} | {m['mean_sec_j_per_mbit']:.8f} | {m['min_predicted_qos']:.6f} | {m['type_switches']} | {m['scale_up_events']}/{m['scale_down_events']} | {m['blocked_requests']} | {'/'.join(map(str,m['action_counts']))} | {m['total_reward']:.6f} | {m['orchestration_reward_magnitude']:.3f} |\n"
    text+='''
New forecasts reduce unchanged SLICES26 Netflix energy from90.382301 to89.341027
Wh and strong-rule energy from88.277182 to88.163527 Wh; the historical binary
rule instead rises88.957901→88.986442 Wh. These deterministic rules ignore the
changed history dimension, so their request/trajectory differences reflect the
new forecast stream. Better RMSE is not uniformly lower controller energy.
Recalibration further reduces A to88.199815 Wh; it does not establish learning's
contribution. C still excludes USR2, so comparisons with it confound action access.

Only recurrent history/hybrid **selected family means** beat B on Netflix, by
.143855/.077320 Wh (about.163%/.088%). All eight raw300k mean energies exceed B;
only2/40 individual endpoints beat it. On YouTube, neither any selected seed nor
any endpoint beats B. The non-learning rule remains a required strong comparator,
not an obstacle to be tuned away.

## 5. All individual seed results

Selected step is chosen on Netflix once and applies to both workloads. SEC is
mean slot J/Mbit; Q is the minimum predicted composite score. SW is type changes,
SC is within-USR up/down, BK is blocked requests. All cases have zero predicted
shortfall slots, mean/max shortfall and QoS penalty; conditional severity is null,
not demonstrated zero physical risk. Full selected and endpoint metrics, including
mean Q, aggregate SEC, launches, penalties, occupancy and endpoint reward/events,
are in [comparison JSON](reports/stage7/comparison.json) and per-workload CSVs.

'''
    for workload in ['netflix','youtube']:
        text+=f"### {workload}\n\n| Algorithm/schema | Seed | Selected step | Selected Wh | Endpoint Wh | SEC | Min Q | SW | SC up/down | BK | Selected reward |\n|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|\n"
        for algorithm in CFG['algorithms']:
            for schema in CFG['schemas']:
                for seed in CFG['seeds']:
                    name=f'{algorithm}_{schema}_{seed}';m=comparison[workload]['controllers_selected'][name];e=comparison[workload]['controllers_endpoint'][name]
                    step=convergence[name]['selected_step']
                    text+=f"| {algorithm}/{schema} | {seed} | {step} | {m['energy_wh']:.6f} | {e['energy_wh']:.6f} | {m['mean_sec_j_per_mbit']:.8f} | {m['min_predicted_qos']:.6f} | {m['type_switches']} | {m['scale_up_events']}/{m['scale_down_events']} | {m['blocked_requests']} | {m['total_reward']:.6f} |\n"
        text+='\n'
    text+='''
## 6. Switching, adaptation and QoS limits

| Workload | Cell | Mean D/USR1/USR2 slots | Mean switches | Mean scale up/down | Mean blocked |
|---|---|---|---:|---|---:|
'''
    for workload in ['netflix','youtube']:
        for key,m in families[workload].items():
            text+=f"| {workload} | {key} | {'/'.join(f'{n:.1f}' for n in m['occupancy_mean'])} | {m['type_switches_mean']:.1f} | {m['scale_up_events_mean']:.1f}/{m['scale_down_events_mean']:.1f} | {m['blocked_requests_mean']:.1f} |\n"
    text+='''
Both families can scale. Hybrid feedforward PPO averages10.2 within-USR changes
and13 type changes on Netflix, versus2.2/11.8 for its history variant; forecasting
is not demonstrated to suppress reconfiguration. Recurrent history averages1.8
within-USR changes and11 type switches. Occupying USR2 can result from a direct
DPDK→USR2 launch; it is not itself a USR1→USR2 scaling event. Scaling's causal
energy contribution is not isolated by these counts or a schema comparison.

The manuscript's FF-hybrid-exclusive scaling and recurrent-instant/forecast static
collapse narratives do not hold for these selected corrected runs. Every selected
learned policy improves over static DPDK on both workloads, but that does not imply
superiority over dynamic rules. Configuration timelines, blocked requests and
[load/rise/fall strata](reports/stage7/adaptation_by_load_and_direction.csv)
provide descriptive adaptation evidence, not real-time alarm-response guarantees.
Realized load direction is used only after decisions for analysis.

All1512 static action/load combinations per validation workload meet point Q≥.90.
Thus every learned/baseline/endpoint rollout has zero predicted shortfalls; this
horizon cannot discriminate QoS-control quality. Profiling false-safe errors,
composite-score semantics, process-attributed power, workload-direction/loss-label
uncertainties, balanced independent USR2 approximation and absent startup/transient
or service-continuity modeling remain binding. Shared surrogate error need not
cancel between different action occupancies. Tiny Wh differences are not measured
whole-server or operational-QoS gains.

## 7. Learning stability and computational trade-offs

All40 selected checkpoints occur after32768 interactions, but every cell's final
Netflix mean energy is worse than its selected mean. The last three validation
points (270336,294912,300000) are mixed in26 runs, decreasing in9 and increasing
in5. Neither fluctuations nor an increasing last window demonstrate stationarity.
Best-checkpoint selection remains useful but adds selection optimism; final
endpoints and every unfavorable checkpoint are retained. Entropy/loss diagnostics
are descriptive and do not prove an entropy-collapse cause.

| Cell | Selected step min/max | Late increasing/mixed/decreasing | Mean endpoint−selected Wh | Mean endpoint−32k reward |
|---|---|---|---:|---:|
'''
    for algorithm in CFG['algorithms']:
        for schema in CFG['schemas']:
            group=[convergence[f'{algorithm}_{schema}_{s}'] for s in CFG['seeds']];cnt=collections.Counter(g['late_direction'] for g in group)
            text+=f"| {algorithm}/{schema} | {min(g['selected_step'] for g in group)}/{max(g['selected_step'] for g in group)} | {cnt['increasing']}/{cnt['mixed']}/{cnt['decreasing']} | {statistics.mean(g['endpoint_minus_selected_energy_wh'] for g in group):+.6f} | {statistics.mean(g['endpoint_minus_32k_reward'] for g in group):+.6f} |\n"
    text+='''
| Cell | Mean training CPU s | Mean training wall s | Mean sequential inference median/p95 ms | Whole-worker peak RSS MiB | Policy parameters KiB |
|---|---:|---:|---|---:|---:|
'''
    for algorithm in CFG['algorithms']:
        for schema in CFG['schemas']:
            g=[costs[f'{algorithm}_{schema}_{s}'] for s in CFG['seeds']]
            avg=lambda k:statistics.mean(v[k] for v in g)
            text+=f"| {algorithm}/{schema} | {avg('training_cpu_seconds_including_checkpoint_validation'):.3f} | {avg('training_wall_seconds_including_checkpoint_validation'):.3f} | {avg('inference_median_ms'):.4f}/{avg('inference_p95_ms'):.4f} | {avg('process_peak_rss_mib'):.2f} | {avg('policy_parameter_bytes')/1024:.2f} |\n"
    text+='''
Total scientific training CPU time is6872.48s (1.909 CPU hours), including checkpoint
serialization/validation, under12 independent one-thread workers. Inference was
remeasured sequentially after training on the same trajectories, with normalization
and recurrent-state audit included. Worker timings are also preserved. Recurrent
history costs about1.87× training CPU and2.16× policy inference versus PPO history;
its selected mean energy gain is.173519 Wh, conditional on the twin.

RSS includes imported TensorFlow/Torch, caches and shared libraries; it is not
intrinsic policy-only memory. Parameter bytes exclude optimizer/activations.
Forecast execution and profiling-outcome simulation are excluded from policy
latency. Stage6's separate provider median was about.63/.72ms Netflix/YouTube;
that cost may be shared by analytics consumers but is not zero. Adding separately
measured components is not an end-to-end NWDAF benchmark. Policy inference is
still far below15 minutes; no physical actuation latency or control/training
energy was measured or added to attributed-UPF Wh.

![Selected versus endpoint energy](reports/stage7/figures/energy_observation_comparison.png)

![All learning curves](reports/stage7/figures/validation_learning_curves.png)

![Training diagnostics](reports/stage7/figures/training_entropy_and_episode_rewards.png)

[Netflix timelines](reports/stage7/figures/netflix_configuration_timelines.png)
and [YouTube timelines](reports/stage7/figures/youtube_configuration_timelines.png)
include all primary seed configurations and dynamic baseline versions, with PDF
exports. [Convergence](reports/stage7/convergence.json),
[optimizer summaries](reports/stage7/optimizer_summary.json) and
[per-seed costs](reports/stage7/computational_cost.json) retain complete evidence.

## 8. Configuration recommendations and scientific boundary

**The question is not answered by a universal hybrid or feedforward winner.**
On Netflix, adding the improved external forecast does not improve selected mean
energy for either family. Recurrence remains descriptively competitive even with
forecasts, but adjusted paired evidence does not establish superiority or prove
that recurrence is necessary. On transfer, forecast benefits depend on family and
schema and the strong rule beats all selected learned policies.

For later, separately authorized reviewer experiments, retain **Recurrent PPO+
history** as the Netflix energy-focused candidate and **PPO+history** as the
lower-cost primary comparator. Keep both hybrid cells as the declared forecast
controls, not as automatically preferred configurations. Instantaneous and
forecast-only cells complete the negative/mixed ablation and must remain in the
record. Do not promote recurrent forecast-only because its YouTube mean happens
to be best among learned cells: that would select on transfer outcomes.

Keep the strong rule and independently calibrated SLICESv2 in every comparison;
include C with its binary-action limitation clearly labeled. More seeds should
follow a precision/endpoints plan, not stopping at significance. Any stability
intervention, algorithm alternative, action-space attribution or reward sweep
requires a new protocol; none begins here. The final test remains unscored, and
no deployment/QoS conclusion follows from these model-conditional results.

The Stage4 RNG-isolated LSTM campaign remains a separate reference: old hybrid
energy means88.250821 FF/88.170231 recurrent versus old rule88.277182 Wh. This
stage changes W95→96 and the complete training forecast/provider/cold-start
process; neither cross-stage means nor pooled repeated seed labels isolate
forecast quality. No old policies are reselected or replaced.

## 9. Verification, data-access exception and preservation

**Final restricted verification:106 passed.** Tests cover four schema dimensions/
ordering, common telemetry, future-target poisoning, ignored forecast privacy,
reset/last-slot indexing, actual→outcome timing, shared cooldown/accounting,
paired checkpoint identity and normalization, RNG-state restoration, recurrent
state/reset, all40 budgets/74 updates/156 completed collector episodes, four
120-slot tails/run, all560 paired artifacts, transfer selection guards and
identical504-slot timestamps. A regression check forbids the legacy full-file
reader in Stage7 inputs. [Final prefix-guard log](reports/stage7/final_prefix_guard_tests.log).

Original preflight:4 synthetic assertion failures/92 passes from strict float
equality (3.9699999999999998 versus3.97). Only the decimal assertion changed;
its log was retained. Two later preflight invocations passed96 tests before any
scientific run. Two existing20-interaction synthetic budget fixtures ran in
those legacy checks, separate from the40 scientific fits. The final suite does
not train additional policies or refit forecasts.

**Actual restriction breach:** the reused `test_stage4.py` preflight checks
called `stage3.common.inputs`, whose legacy reader parsed the complete Netflix
CSV, including its final-test suffix, to reconstruct old timeline identities.
It also transforms the full array while constructing partitions. No final-test
values were printed, deliberately inspected, scored, supplied to gradient fitting
or used to select a model/checkpoint. However, literal **no target access was not
satisfied**; this report does not claim otherwise. The intended declaration in
the pre-fit protocol is retained, with this explicit qualification.

The scientific Stage7 preparation, training and evaluation independently read
only authorized prefixes and verify Stage6 fingerprints; no demonstrated dataflow
from that incidental legacy parsing affects learned parameters or selections.
Remaining tests exclude the broad legacy real-data identity checks. The safe
[README commands](stage7/README.md) now do likewise. Preserve the
[data-access addendum](reports/stage7/DATA_ACCESS_ADDENDUM.json) and all original
logs. This is an access-boundary failure, **not final-test performance exposure**;
future reports should say unscored final test, not claim targets were never parsed.

All **2210 pre-task scientific files** and original tracked files are verified
unchanged by [integrity](reports/stage7/integrity.json). New isolated code/config/
observation audit is in `revision/stage7/`; model bundles and per-run logs are in
`revision/models/stage7/{algorithm}/{schema}/seed_{seed}/`; full comparison tables,
trajectories, statistics, compute records and plots are in `revision/reports/stage7/`.
[The final freeze manifest](reports/stage7/observation_freeze_manifest.json) lists
every new path/hash and the incident qualification. Prior Stage4–6 results,
forecast providers and default twin remain intact. No manuscript, Git reference,
commit or remote was changed. **Stop after Stage7.**
'''
    # Readability only; scientific numbers are generated above from sealed results.
    for a,b in [('Stage7','Stage 7'),('Stage6','Stage 6'),('Stage5','Stage 5'),('Stage4','Stage 4'),('October2026','October 2026'),('date:23','date: 23'),('All40','All 40'),('exactly300000','exactly 300000'),('across2','across 2'),('and5','and 5'),('All560','All 560'),('both96','both 96'),('the96','the 96'),('by400','by 400'),('fields are0','fields are 0'),('mask0','mask 0'),('uses95','uses 95'),('uses seeds42','uses seeds 42'),('n_steps1024','n_steps=1024'),('batch64','batch=64'),('lr.0001','lr=.0001'),('gamma.995','gamma=.995'),('GAE.90','GAE=.90'),('entropy.15','entropy=.15'),('clip.20','clip=.20'),('target-KL.03','target-KL=.03'),('LSTM128','LSTM128'),('final248','final 248'),('at300000','at 300000'),('every24576','every 24576'),('through504','through 504'),('scores1920','scores 1920'),('common96','common 96'),('identical504','identical 504'),('timestamps,22','timestamps, 22'),('April2019','April 2019'),('corresponding504','corresponding 504'),('at2520','at 2520'),('all40','all 40'),('and1008','and 1008'),('these,576','these, 576'),('and336','and 336'),('by.127883','by .127883'),('five blocks,','five blocks,'),('p is.0625','p is .0625'),('intervals include0','intervals include 0'),('point Q≥.90','point Q≥.90'),('original26','original 26'),('and2%','and 2%'),('plotted23','plotted 23'),('margin to88','margin to 88'),('from90','from 90'),('to89','to 89'),('from88','from 88'),('rises88','rises 88'),('selected26','selected 26'),('to88','to 88'),('costs about1.87','costs about 1.87'),('and2.16','and 2.16'),('about.63','about .63'),('below15','below 15'),('in26','in 26'),('in9','in 9'),('in5','in 5'),('are6872','are 6872'),('is6872','is 6872'),('under12','under 12'),('only2/40','only 2/40'),('eight raw300k','eight raw 300k'),('all1512','all 1512'),('averages10.2','averages 10.2'),('and13','and 13'),('versus2.2/11.8','versus 2.2/11.8'),('averages1.8','averages 1.8'),('and11','and 11'),('All40','All 40'),('after32768','after 32768'),('passed96','passed 96'),('existing20','existing 20'),('equality (3.969','equality (3.969'),('versus3.97','versus 3.97'),('verification:106','verification: 106'),('preflight:4','preflight: 4'),('passed96','passed 96'),('means88','means 88'),('old rule88','old rule 88')]:text=text.replace(a,b)
    path=ROOT/'revision/FINAL_OBSERVATION_ABLATION.md'
    with path.open('x') as f:f.write(text)
    print(path,flush=True)

if __name__=='__main__':main()

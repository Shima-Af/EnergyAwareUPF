"""Compose the single consolidated Stage 4 report from completed corrected evidence."""
import json,re
from pathlib import Path
import numpy as np
from revision.evaluation.twin import ROOT
from .common import CFG
O=ROOT/'revision/reports/stage4_rng_isolated'
M=ROOT/'revision/models/stage4_rng_isolated'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
    controllers=json.loads((O/'controller_comparison.json').read_text())['controllers']
    convergence=json.loads((O/'convergence.json').read_text());sensitivity=json.loads((O/'forecast_sensitivity.json').read_text())
    scaling=json.loads((O/'scaling_accounting.json').read_text());summary=json.loads((O/'comparison_summary.json').read_text())['algorithms']
    accuracy=json.loads((O/'forecast_accuracy.json').read_text());rule=controllers['rule_three_action']
    initial=json.loads((ROOT/'revision/reports/stage4/controller_comparison.json').read_text())['controllers']
    pilot=json.loads((ROOT/'revision/reports/stage3/controller_comparison.json').read_text())['controllers']
    testlog=(ROOT/'revision/reports/stage4_final_tests.log').read_text().splitlines()[-1]
    text='''# Full training and convergence evaluation — COMCOM-D-26-01061

**Stage 4 complete.** The primary evidence below uses the corrected, RNG-isolated
campaign: five fresh seeds per PPO family, exactly 300,000 interactions each,
the unchanged frozen digital twin/reward/rule/contract, and the common 504-slot
Netflix validation horizon. Results are exploratory and validation-selected;
the final 840-slot test remains untouched. No manuscript edit, DQN, broad reward
sweep, remaining reviewer campaign, commit or push occurred.

Work date: **23 October 2026**. Branch `revision/comcom-clean-evaluation`, HEAD
`c0734715418d3f6d10785e5174aebb4bac24d0da`.

## 1. Findings at a glance

'''
    for alg in CFG['algorithms']:
        d=summary[alg];e=d['metrics']['energy_wh'];r=d['metrics']['total_reward']
        text+=f"- **{alg}: {e['mean']:.6f} ± {e['sample_sd']:.6f} Wh** (sample SD, five seeds), {d['mean_energy_percent_over_rule']:+.3f}% relative to rule; {d['lower_energy_than_rule']}/5 seeds use less energy and {d['higher_reward_than_rule']}/5 have higher total reward. Mean total reward {r['mean']:.6f}.\n"
    text+=f"- **Unchanged rule: {rule['energy_wh']:.6f} Wh**, mean SEC {rule['mean_sec_j_per_mbit']:.8f} J/Mbit, total reward {rule['total_reward']:.6f}.\n"
    text+='''- Best-selected and final-endpoint performance must be distinguished. More
  checkpoint opportunities are not themselves evidence of stable convergence.
- All 1,512 action/validation-load pairs meet the frozen point-score target, so
  zero predicted shortfalls is non-discriminating and establishes no safety claim.
- Calendar-LSTM validation accuracy improves, but persistence remains better;
  causal prefix training forecasts become worse. Forecast quality remains a limit.

## 2. Frozen design and necessary implementation correction

[The Stage 4 protocol](stage4/protocol.json) was saved before fitting. Seeds are
42, 123, 456, 789 and 1024 for each algorithm. Stage 3 hyperparameters are retained:
four collectors, n_steps=1024, batch64, ten PPO epochs, lr=.0001, gamma=.995,
GAE=.90, entropy=.15, clip=.20, target-KL=.03; recurrent shared LSTM128, no separate
critic LSTM. Both algorithms use the same 109-element hybrid observation.

All controllers retain DPDK initialization, eligible four-slot cooldown, three
actions, causal completed-interval telemetry, the 95-slot history, 96-slot forecast
input, and the original corrected contract. Rejected commands are discarded;
changes at t permit another at t+4. The reward remains −100×SEC −30×score shortfall
minus .030 switch-to-DPDK, .012 per launched USR instance and .003 within-USR scale
down. No reward coefficients, score formula, twin outputs or rule thresholds are
tuned. The rule still uses all three forecast-conditioned curves, the .90
point-score preference and fixed 2% power deadband. Only its forecast inputs change.

Frozen twin identity:
`80866f0adf81c3f144190fe49d8cf74e2d5eefe95c90b63a2924fb8da67c465e`.
See [final twin validation](FINAL_DIGITAL_TWIN_VALIDATION.md) for the three neural
models plus USR power regression, exact artifacts/scalers, measurement lineage
and unresolved physical assumptions. The [initial frozen protocol](reports/stage4/training_protocol_frozen.json)
and [corrected execution amendment](reports/stage4_rng_isolated/training_protocol_frozen.json)
pin code, config, forecasts, twin identity and the preflight results.

**A confirmed RNG side effect required a complete implementation-correction rerun.**
Ten initial full-budget runs completed. Post-training verification demonstrated
that SB3 checkpoint loading resets Python, NumPy and Torch RNGs: `_setup_model`
sets the saved seed and initializes layers before restoring weights. Because
checkpoint validation ran during training, it altered subsequent exploration and
minibatch draws. This also affects interpreting Stage 3's inherited callback.

[The diagnostic probe](reports/stage4/evaluation_rng_probe.json) and
[RNG-isolation amendment](stage4/RNG_ISOLATION_AMENDMENT.md) record the cause and
correction. The entire checkpoint capture/evaluation now saves/restores all three
CPU RNG states, including on exceptions. No algorithm hyperparameters, forecast
weights, baseline, objective or seed list changed. All ten corrected jobs were
prescribed before their outcomes were known; no seed is discarded or selected
for a favorable rerun. Initial files remain at `models/stage4` / `reports/stage4`;
primary files use **`models/stage4_rng_isolated` / `reports/stage4_rng_isolated`**.
Both campaigns are retained and must not be pooled as ten independent seeds.

The first 24,576-interaction checkpoint metrics match between the two campaigns
for every seed, before validation could perturb the next random stream. Later
trajectories diverge as expected. This supplies a direct implementation-regression
check rather than an unexplained change of scientific protocol. In total there
are 20 full controller fits / six million interactions, but only ten corrected
fits / three million interactions form the primary five-seed-per-algorithm study.

**Exact budget and checkpoint timing:** regular rollouts remain1024 steps per
collector. After299008 interactions, the last rollout is248 steps per collector,
giving992 further interactions and exactly300000 total. Its returns/advantages
and optimizer updates are computed normally; no tail samples are discarded and
there is no unreported303104-step overshoot. The final minibatch may contain32
samples. This budget-boundary behavior is tested for both algorithms.

Each run preserves14 post-update policy/normalizer pairs: every24576 interactions
(approximately25000), an additional32768-step matched pilot-budget checkpoint,
and300000 after the final optimizer update. Selection remains maximum validation
**total reward**, with earlier checkpoint as tie-breaker, never energy superiority.
VecNormalize moments update only during training; rewards remain unnormalized.
Recurrent hidden/cell states persist during evaluation and reset only per episode.

Training data score1920 slots per episode after common warmup. Four collectors
repeat this deterministic workload; they are not independent datasets. Each run
logs74 optimizer updates,156 completed collector episodes and four incomplete
120-slot episode tails. CPU operation is deterministic under the pinned runtime:
Python3.12.3, TensorFlow2.20.0/Keras3.13.2, NumPy2.4.2, Torch2.10.0+cu128,
SB3/sb3-contrib2.7.1, Gymnasium1.2.3; one Torch thread per independent worker.
No GPU training is used. Up to four independent processes run concurrently.

## 3. Forecast training, selection and accuracy

Exactly two LSTM candidates retain Stage 3's 128/64 stacked hidden widths,
96 completed input slots, dropout and nonnegative softplus head: load-only and
six known decision-calendar features. Seed42, batch128, Adam.001, maximum100
epochs, patience15, minimum scaled-MSE improvement1e-6 and plateau reduction
(factor.5 after6 epochs, minimum lr1e-5) were fixed beforehand. No search was
expanded after seeing forecast or controller results.

Full-prefix gradient targets are96..2015; the504-slot validation suffix selects
checkpoint/candidate. The load-only candidate ran46 epochs, selected31; calendar
ran41, selected26 and won by validation RMSE. This is validation-selected accuracy,
not an untouched final test. No controller result selected the forecast.

For training forecasts, selected architecture fits prefixes ending192,608,1024,
1440,1856. Each uses an internal chronological holdout: the last20% of available
sequence targets supplies stopping information, entirely before its subsequent
forecast block. Scalers and gradient labels use only the earlier training subset.
Both gradient and stopping cutoffs are recorded. The initial96 scored training
slots retain explicit persistence cold start (5% of scored training), not an
undisclosed replacement of the LSTM. All validation inputs use the full-training-
prefix selected LSTM. Calendar information is known at decision time; actual
interval-t load never enters its own forecast input.

Seven scientific fits ran227 epochs total of a700-epoch cap. The corrected
campaign copies these artifacts byte for byte; **no forecasts are refitted for
the RNG correction**. Model/scaler manifests, epoch histories and weights are
retained under [forecast artifacts](models/stage4/forecast/), and selection is
[hash-pinned](reports/stage4/forecast_selection.json). The Stage 3 forecaster and
all datasets remain unchanged.

'''
    fs=json.loads((O/'forecast_selection.json').read_text())['validation_metrics'];rows=[]
    for name,d in fs.items():rows.append([name+' validation',d['n'],f"{d['mae_mbps']:.6f}",f"{d['rmse_mbps']:.6f}",f"{d['bias_mbps']:+.6f}",f"{d['r2']:.6f}"])
    for name,d in accuracy.items():
        if name=='validation_lstm':continue
        rows.append([name,d['n'],f"{d['mae_mbps']:.6f}",f"{d['rmse_mbps']:.6f}",f"{d['bias_mbps']:+.6f}",f"{d['r2']:.6f}"])
    text+=table(['Forecast / scope','N','MAE Mbps','RMSE Mbps','Bias Mbps','R²'],rows)
    text+='''

The selected validation RMSE improves from Stage 3's28.508475 to25.690384 Mbps,
but still trails persistence20.810623. The causal training-ledger RMSE worsens
from51.358767 to66.333705 Mbps: small earlier prefixes plus internal stopping
produce poorly calibrated early forecasts. More training budget is not proof
that every prefix predictor improves. This weakness is preserved and limits
attribution of controller findings; persistence is saved as a labelled alternative,
not substituted into primary training. No final-test forecast/outcome is evaluated.

[Forecast ledger](reports/stage4_rng_isolated/forecasts.csv) records2424 issued
training/validation predictions and cutoffs. [Accuracy](reports/stage4_rng_isolated/forecast_accuracy.json)
and [persistence ledger](reports/stage4_rng_isolated/persistence_forecasts.csv)
are complete. The selected saved network reproduces its ordered validation
forecasts exactly under the fixed inference profile.

![Forecast comparison](reports/stage4_rng_isolated/figures/forecast_comparison.png)

![Forecast learning curves](reports/stage4_rng_isolated/figures/forecast_training.png)

## 4. Complete validation controller comparison

All rows use the same504 quarter-hour timestamps,22 April2019 00:00 through
27 April05:45,126 hours. Horizon hash:
`1d0955ec0c78322db213461b94c8bc4a17d38cfb10679a98994ea3200a0ce0c4`.
Energy is attributed W×.25h inWh; SEC is J/Mbit. Arithmetic mean slot SEC differs
from aggregate energy/offered-traffic SEC. QoS is a predicted composite score.

'''
    rows=[]
    for n,r in controllers.items():rows.append([n,f"{r['energy_wh']:.6f}",f"{r['mean_sec_j_per_mbit']:.8f}",f"{r['aggregate_sec_j_per_mbit']:.8f}",f"{r['total_reward']:.6f}",f"{r['mean_predicted_qos']:.6f}",f"{r['min_predicted_qos']:.6f}",f"{r['qos_violation_rate']:.6f}"])
    text+=table(['Controller','Wh','Mean SEC','Aggregate SEC','Total reward','Mean Q','Min Q','Shortfall fraction'],rows)
    text+='\n\n'+table(['Algorithm','Energy mean±sample SD Wh','Mean SEC±SD','Reward mean±SD','Energy below rule','Reward above rule'],[[alg,f"{d['metrics']['energy_wh']['mean']:.6f} ± {d['metrics']['energy_wh']['sample_sd']:.6f}",f"{d['metrics']['mean_sec_j_per_mbit']['mean']:.8f} ± {d['metrics']['mean_sec_j_per_mbit']['sample_sd']:.8f}",f"{d['metrics']['total_reward']['mean']:.6f} ± {d['metrics']['total_reward']['sample_sd']:.6f}",f"{d['lower_energy_than_rule']}/5",f"{d['higher_reward_than_rule']}/5"] for alg,d in summary.items()])
    text+='''

Dispersion is descriptive across five independently seeded training runs on one
fixed workload, not uncertainty across cities, periods or physical campaigns.
This is not completion of reviewer replication/statistical requirements. No
seed-pooling across stages, per-slot pseudo-replication or superiority p-value is
used. Validation selected forecasts and checkpoints; these are not final-test
claims. Tiny selected energy differences need interpretation relative to model
error and selection optimism, not merely the number of printed decimal places.

![Controller energy and occupancy](reports/stage4_rng_isolated/figures/controller_comparison.png)

'''
    text+=table(['Controller','Sum100×SEC','Sum QoS penalty','Sum reconfiguration penalty','Mean shortfall','Max shortfall'],[[n,f"{r['energy_reward_magnitude']:.6f}",f"{r['qos_reward_magnitude']:.6f}",f"{r['orchestration_reward_magnitude']:.6f}",f"{r['mean_qos_shortfall']:.8f}",f"{r['violation_shortfall_max']:.8f}"] for n,r in controllers.items()])
    text+='''

All three static actions have minimum predicted Q>.90 on this horizon. Therefore
all1512 action/load combinations satisfy the point target; zero shortfalls do
not distinguish policy QoS behavior. Conditional shortfall severity is undefined
(null) when no shortfall occurs. This cannot establish actual service safety,
formal5QI compliance, or an acceptable false-safe risk. Prior profiling results
already show severe false-safe outcomes, which more controller training does not
repair. No new risk tolerance or conservative replacement score is invented.

## 5. Convergence, pilot comparison and endpoints

The32768-step checkpoint permits a within-run, same-forecast budget comparison.
The best checkpoint available by that budget is the better of24576 and32768.
The full selected result has more selection opportunities, so its improvement
alone is not an unbiased learning-effect estimate. Raw300000-step endpoints and
late trajectories are reported separately.

'''
    rows=[]
    for n,c in convergence.items():
        chosen=controllers[n];selected_step=int(chosen['selected_checkpoint'].split('_')[-1]);early=max([s for s in c['all_scores'] if s['timesteps']<=32768],key=lambda s:s['total_reward'])
        rows.append([n,selected_step,f"{early['metrics']['energy_wh']:.6f}",f"{c['matched_32768']['energy_wh']:.6f}",f"{chosen['energy_wh']:.6f}",f"{c['endpoint_300000']['energy_wh']:.6f}",f"{c['endpoint_minus_32768_reward']:+.6f}",c['late_observed_direction'],f"{c['late_reward_slope_per_100k']:+.3f}"])
    text+=table(['Run','Selected step','Best≤32k Wh','32k endpoint Wh','Selected Wh','300k endpoint Wh','Reward300k−32k','Late direction','Late slope reward/100k'],rows)
    text+='\n\n'
    for alg in CFG['algorithms']:
        cc=[convergence[f'{alg}_{s}'] for s in CFG['seeds']]
        selected_later=sum(int(controllers[f'{alg}_{s}']['selected_checkpoint'].split('_')[-1])>32768 for s in CFG['seeds'])
        text+=f"- **{alg}:** raw final reward improves over32k in {sum(c['endpoint_minus_32768_reward']>0 for c in cc)}/5; raw final energy decreases in {sum(c['endpoint_minus_32768_wh']<0 for c in cc)}/5. Mean32k/final/selected energies are {np.mean([c['matched_32768']['energy_wh'] for c in cc]):.6f} / {np.mean([c['endpoint_300000']['energy_wh'] for c in cc]):.6f} / {summary[alg]['metrics']['energy_wh']['mean']:.6f} Wh. {selected_later}/5 selected checkpoints occur after32k. Late directions: "+', '.join(f"{k}={sum(c['late_observed_direction']==k for c in cc)}" for k in ['increasing','decreasing','mixed'])+'.\n'
    text+='''

The last three validation points are270336,294912,300000. Their directions and
OLS slopes are descriptive, not stationarity/equivalence tests; uneven checkpoint
spacing is retained. A fluctuating or deteriorating final window cannot be
called settled convergence. Falling training losses or improving stochastic
training returns do not override poorer deterministic validation performance.
[Every checkpoint](reports/stage4_rng_isolated/convergence.json) is retained.

![Validation convergence](reports/stage4_rng_isolated/figures/validation_convergence.png)

![Training episode rewards](reports/stage4_rng_isolated/figures/training_episode_rewards.png)

![Optimizer diagnostics](reports/stage4_rng_isolated/figures/optimizer_diagnostics.png)

The optimizer JSONL logs loss, policy/value loss, entropy loss, approximateKL,
clip fraction, explained variance and update count after each update, with exact
interaction timestamp. Standard SB3 CSV logs can lag an update because its loop
dumps before training; use the explicit JSONL for timing. Monitor CSVs and full-
episode logs preserve stochastic training rewards, energy, score and events.
Negative entropy_loss is expected; it is not itself an optimization failure.
Final entropy estimates remain about .56–.79 for FF and .73–.83 for recurrent
PPO; these do not establish a near-zero entropy-collapse mechanism. Maximum
logged mean approximate KL is .00950 for FF and .05398 for recurrent. The traces
show fluctuations but do not identify a single causal explanation for them.

Stage 3 means (two seeds) were89.900771 FF /88.502157 recurrent Wh versus rule
88.255351. Stage 4 uses five seeds, changed forecasts and RNG-isolated validation;
comparing those means alone cannot isolate training length. The within-Stage 4
matched-budget/endpoints above provide the cleaner budget diagnostic. Neither
stage proves performance on the untouched final840 intervals.

## 6. Switching, scaling and contribution limits

'''
    text+=table(['Controller','DPDK/USR1/USR2 slots','Type switches','Scale up','Scale down','Blocked','USR launches'],[[n,'/'.join(map(str,r['action_counts'])),r['type_switches'],r['scale_up_events'],r['scale_down_events'],r['blocked_requests'],r['usr_instances_launched']] for n,r in controllers.items()])
    text+='\n\n'+table(['Learned run','Within-USR changes','Changes with lower immediate power','Sum local energy difference Wh'],[[n,d['events'],d['events_lower_executed_power'],f"{d['local_energy_difference_sum_wh']:+.6f}"] for n,d in scaling.items()])
    text+='''

Local energy difference compares the executed USR level with holding its previous
level at the same realized slot load. Positive values mean lower executed slot
power. This diagnostic does not propagate cooldown, later state or policy
responses, and is **not a causal no-scaling policy ablation**. It can identify
locally unfavorable scale changes but cannot assign a trajectory energy advantage
to scaling. A no-scaling retraining/ablation is not performed here. Event type,
USR(2) occupancy and launches must remain distinct: a DPDK→USR(2) change launches
two instances but is a type switch, not a within-USR scale-up event.

## 7. Does forecast quality materially change behavior?

Selected Stage 4 policies and normalizers are replayed with unchanged Stage 3 LSTM
or persistence forecasts, without retraining or checkpoint reselection. The rule
uses the same alternate forecast. Earlier outcomes then change subsequent
telemetry and recurrent state. These are inference-sensitivity experiments,
not fully trained forecast-factorial causal comparisons.

'''
    rows=[]
    for label,records in [('primary',controllers),*sensitivity['fixed_stage4_policies'].items()]:
        rows.append([label,f"{records['rule_three_action']['energy_wh']:.6f}",* [f"{np.mean([records[f'{alg}_{s}']['energy_wh'] for s in CFG['seeds']]):.6f}" for alg in CFG['algorithms']]])
    text+=table(['Forecast at inference','Rule Wh','FF mean Wh','Recurrent mean Wh'],rows)
    text+='\n\n'
    for label,records in sensitivity['fixed_stage4_policies'].items():
        for alg in CFG['algorithms']:
            rr=[records[f'{alg}_{s}'] for s in CFG['seeds']];delta=[r['energy_wh']-controllers[f'{alg}_{s}']['energy_wh'] for s,r in zip(CFG['seeds'],rr)]
            text+=f"- {label}, {alg}: changed requests range {min(r['changed_requests'] for r in rr)}–{max(r['changed_requests'] for r in rr)} of504; energy change range {min(delta):+.6f}–{max(delta):+.6f} Wh.\n"
    text+='\n'+table(['Fixed Stage 3 policy','Original Stage 3 forecast Wh','Stage 4 forecast Wh'],[[n,f"{pilot[n]['energy_wh']:.6f}",f"{r['energy_wh']:.6f}"] for n,r in sensitivity['fixed_stage3_policies_new_forecasts'].items()])
    text+='''

Forecast accuracy and controller energy are different selection objectives.
These replays quantify effects rather than declaring a post-hoc forecast winner;
the primary LSTM remains validation-RMSE-selected. Training forecasts also changed
and worsened, so this study cannot fully separate training-distribution effects
from inference forecast quality. The short-prefix weakness remains unresolved
within the fixed search budget. [All replay metrics](reports/stage4_rng_isolated/forecast_sensitivity.json)
are retained, with no final-test outcomes.

## 8. Initial campaign, tests, provenance and preservation

The implementation-affected initial campaign is retained, not silently replaced:

'''
    text+=table(['Run','Initial affected selected Wh','RNG-isolated selected Wh'],[[n,f"{initial[n]['energy_wh']:.6f}",f"{r['energy_wh']:.6f}"] for n,r in controllers.items() if n.startswith(('ppo_','recurrent_ppo_'))])
    text+='''

These are repeated seed labels under two execution protocols, not20 independent
replicates. Primary status was decided by the implementation fix, not favorable
metrics. The [initial campaign snapshot](reports/stage4_initial_campaign_sha256.json)
preserves all initial results/code. The [pre-Stage 4 snapshot](reports/stage4_prior_sha256.json)
checks490 prior scientific files unchanged. Its original self-reference was a
bookkeeping error: it recorded the snapshot's own empty file at creation. That
entry alone was excluded and the original snapshot retained; no prior artifact
was changed by this correction.

Before initial training:247 tests passed, one post-training check skipped,
28 warnings. Before RNG-isolated training:254 passed, one then-pending artifact
check skipped,28 warnings. New exception-safe RNG-state and actual checkpoint
loading regression tests passed before rerunning. Additional post-training checks
verify both campaigns, exact budgets, all paired checkpoints/scalers, chronology,
forecast cutoffs/reload, final-test rejection, reward/cooldown arithmetic, common
horizons, normalization immutability, recurrent state and fixed-policy replay.

'''
    text+=f"**Final full revision suite: {testlog}.**\n\n"
    text+='''Logs: [initial preflight](reports/stage4_preflight_tests.log),
[corrected preflight](reports/stage4_isolated_preflight_tests.log),
[final tests](reports/stage4_final_tests.log),
[forecast training](reports/stage4_forecasting.log),
[initial workers](reports/stage4_training.log),
[corrected workers](reports/stage4_isolated_training.log).
The28 warnings are the preexisting Keras/NumPy array-copy deprecations. Tiny
synthetic profiling/PPO fits in unit tests are fixtures, not scientific candidates.
All prescribed scientific runs and failures/diagnostic history are preserved.

Primary [model bundles](models/stage4_rng_isolated/controllers/),
[per-seed metrics](reports/stage4_rng_isolated/controller_comparison.json),
[comparison CSV](reports/stage4_rng_isolated/controller_comparison.csv),
[family summaries](reports/stage4_rng_isolated/comparison_summary.json),
[scaling accounting](reports/stage4_rng_isolated/scaling_accounting.json),
[optimizer summary](reports/stage4_rng_isolated/optimizer_diagnostics.json), and
PNG/PDF plots form the evidence package. Each run directory retains optimizer
JSONL, episode JSONL, four Monitor files, partial-episode tails, selection ledger
and14 paired policy/normalizer bundles. Corrected and initial campaigns preserve
280 checkpoint pairs in total. Forecast/scaler artifacts copied into the corrected
namespace have identical hashes; they are not additional fits.

[The final integrity record](reports/stage4_rng_isolated/integrity.json) checks
both preservation snapshots, frozen source/twin/config identities, all model
hashes and unchanged Git branch/HEAD/tracked files. The
[new-file manifest](reports/stage4_rng_isolated/file_manifest.json) lists every
scientific addition and hash, excluding bytecode and the manifest itself. External
profiling exports still need separate backup. Reproducible command order and
runtime correction are documented in [Stage 4 instructions](stage4/README.md) and
the [RNG amendment](stage4/RNG_ISOLATION_AMENDMENT.md). Use the corrected command
sequence in the amendment; the original README/launcher describe the preserved
initial affected execution. The plot adapter supplies a
union CSV schema for static and learned rows; it changes no numeric values.

## 9. Answers to the six scientific questions

1. **Does longer training help?** Selectively. Only 2/5 FF runs select a checkpoint
   after 32k; raw final reward improves over 32k in 2/5 and energy in 1/5.
   Recurrent PPO selects after 32k in 4/5, with final reward improving in 5/5
   and energy in 3/5. Mean endpoint energy barely changes for FF (88.766904 to
   88.757776 Wh) and improves modestly for recurrent (89.810869 to 89.613369 Wh).
   Best-selected means are better, but selection opportunities contribute.
   The Stage 3-to-4 comparison also changes forecasts and RNG handling; it cannot
   isolate training length on its own.
2. **Does either algorithm beat the fair rule?** Yes, narrowly in selected mean
   validation energy: FF by 0.026361 Wh (0.030%) and recurrent by 0.106951 Wh
   (0.121%). Individual energy wins are 3/5 and 4/5; reward wins are 4/5 each.
   This supports near parity and limited conditional plausibility, not robust
   or statistically established superiority. It does not reproduce the historical
   advantage as final held-out evidence.
3. **Are the curves converged at 300k?** No settled convergence is demonstrated.
   FF late reward declines monotonically in 3/5 runs, increases in 1/5 and is
   mixed in 1/5. All recurrent late windows are mixed, with negative fitted
   slopes. Recurrent seed 42 ends at 92.475392 Wh after selecting 88.222109 Wh
   at 294912; seed 1024 spikes to 96.719172 Wh at 294912 and ends at 90.245646 Wh.
   Earlier best-checkpoint selection limits harm but does not establish stability.
4. **Does forecast improvement change behavior?** Yes, but the observed effects
   are usually modest and depend on the policy. Stage 4 policies replayed with
   Stage 3 forecasts change 0–8 requests out of 504 and roughly up to 0.025 Wh;
   persistence changes 0–6 requests and up to +0.209507 Wh for FF. The rule's
   energy also changes. Selected LSTM validation RMSE improves, while its causal
   training forecasts worsen and persistence remains more accurate. These
   inference replays do not isolate the full effect of training with a different
   forecast distribution.
5. **Does scaling cause the energy benefit?** Not established. The lowest-energy
   selected FF seed (123, 87.940790 Wh) makes no within-USR scale changes.
   Some scaling events lower immediate power; others raise it. Local sums span
   −0.016843 to +0.020249 Wh, excluding future responses. Two-instance occupancy
   can still arise through DPDK→USR(2) launches. Within-USR scaling is neither
   demonstrated necessary nor proven causally beneficial by this experiment;
   a controlled policy ablation would be needed for the latter claim.
6. **Which original claims remain?** Energy savings versus static provisioning
   are supported conditionally on the twin. PPO superiority over a fair rule is
   uncertain beyond the tiny, selected validation means above. Predicted-QoS
   compliance is non-discriminating here and provides no measured safety evidence.
   A feedforward-only or scaling-over-switching narrative is unsupported: both
   families can scale, several strong policies do not, and no learned selected
   policy has fewer type switches than the rule's 11.

This closes Stage 4. Five seeds are for convergence investigation, not completion
of reviewer statistical replication. Any later reviewer campaign must retain all
results, prespecify seeds/endpoints/statistics, freeze remaining model choices,
and preserve the final840-slot test until separately authorized. The fixed twin's
process-attribution, workload-direction/loss semantics, correlated campaigns,
conditional QoS false-safety, absent idle/transient measurements and balanced
USR(2) assumptions remain limits that additional policy training cannot remove.
No remaining reviewer experiment or final-test evaluation starts automatically.
'''
    path=ROOT/'revision/FULL_TRAINING_EVALUATION.md'
    with path.open('x') as f:f.write(text)
    print('Consolidated Stage 4 report written')

if __name__=='__main__':main()

# Scientific repository audit — COMCOM-D-26-01061

**Manuscript:** AI-Driven Energy-Efficient Orchestration of User Plane Functions in 5G Core Networks  
**Audit date:** 22 October 2026 · **Revision deadline:** 18 November 2026  
**Repository:** `/home/ubuntu/EnergyAwareUPF`  
**Audited commit:** `a23e7f7de6c6e07fddad8411dde542b38123e785` (`main`, also `origin/main`)

## 1. Executive verdict

**B. PARTIALLY REUSABLE — high confidence.**

The deterministic twin, four fitted profiling models and their scalers, both processed traffic traces, all ten selected PPO checkpoints, normalization files, run configurations, and substantial evaluation evidence are recoverable. The existing virtual environment works. CPU replay of all five feed-forward and five recurrent checkpoints on both traces reproduced **all 20 archived executed-action sequences exactly**, without training.

This does **not** establish exact reproduction of the submitted methodology or scientific validity of the comparisons. The manuscript combines incompatible experimental slices and artifact generations. Several implementation defects materially affect interpretation:

1. Main-trace rule/static reference results use **720 slots**, PPO and the offline planner use **721**, and their starting/ending timestamps differ. Neither is the stated 840-slot test evaluation.
2. The recurrent evaluator fails to recognize the recurrent policy and discards its hidden state on every prediction. The archived recurrent action sequences match this defective evaluation path.
3. Best-checkpoint selection repeatedly evaluates the same traffic suffix later presented as held out. Only final-training observation-normalization statistics are saved, not statistics paired with each best checkpoint.
4. The reported forecast RMSE belongs to a later calendar/residual predictor; the principal multi-seed checkpoints use the older forecast CSV. The later predictor was selected by test RMSE.
5. The rule baseline used in publication-table generation only chooses DPDK or USR(1). Its comparison does not isolate learned preference for horizontal scaling when PPO also has USR(2).
6. Fresh inference shows small but systematic power differences from archived outputs. Action agreement is established; numerical reproduction of all energy results is **not**. A scikit-learn serialization-version mismatch is observed, but the cause of the power differences is unresolved.
7. The current B2 sweep specifies QoS target **0.95**, whereas all ten archived run configurations specify **0.90**. Running the supplied sweep unchanged would not reproduce those runs.

The repository is useful for historical reconstruction and a corrected revision campaign. It is not ready for simply adding seeds or alternatives and combining them with the published tables.

### Scope and actions performed

No existing repository files were modified, no dependencies installed/upgraded, no models trained, no manuscript edited, no Git branches changed, and no commits created. Git status was clean before and after inspection. All new audit scripts, logs, replay CSVs, and this report are under `/tmp/upf-scientific-audit/`.

Executed checks: dependency imports; saved-checkpoint inference; deterministic static/rule/offline computations; profiling-model inference on recovered external CSVs; arithmetic checks on archived predictions and tables; five inspected offline-planner unit tests, all passing. The full test suite, model retraining, live deployment, and reviewer experiments were **NOT TESTED**.

A successful replay can reproduce a flawed procedure. Throughout this report, “archive verified,” “freshly executed,” and “methodologically validated” are distinct claims.

## 2. Repository inventory and methodology reconstruction

Paths below are relative to the repository unless explicitly absolute. Availability of external material does not make the Git repository self-contained.

| Component and principal functions/classes | Evidence/path | Availability and compatibility | Dependencies and reuse decision |
|---|---|---|---|
| Experiment configuration | `config.yaml`; `experiments/sweep_config*.yaml`; per-run `config.yaml` | Original numeric PPO settings recoverable; root config is recurrent/hybrid, not either selected winning pairing; B2 sweep has drifted to tau=0.95 | YAML available. Use archived run snapshots for reconstruction; sweep cannot be reused unchanged |
| PPO/RecurrentPPO construction, `create_agent`, `_common_kwargs` | `src/agent.py` | SB3 PPO and sb3-contrib RecurrentPPO, supplied hyperparameters present | Installed and importable. Reusable core; improve seed provenance and checkpoint pairing for revision |
| Training, `train`, `run_single_experiment`, `apply_overrides` | `src/train.py`; `scripts/run_experiments.py` | 300,000 requested steps; actual final checkpoints 303,104; eval callback selects best test return | Reusable orchestration after validation split and checkpoint-stat fixes; not unchanged scientific protocol |
| Deterministic twin, `ManualCooldownEnv` | `src/environment.py` | Discrete(3), equal per-instance load division, summed power, per-instance QoS, no stochastic model error | Gymnasium/NumPy available. Historical semantics recoverable; timing, reward description and observation alignment need resolution |
| Profiling inference, `UPFApproximatorBundle.predict_batch` | `src/utils.py:145`; `saved_models/prediction_models/` | Four Keras models and eight profiling scalers present; log1p(load), input scaling, inverse target scaling | TensorFlow/Keras/joblib available. Can reuse fitted artifacts for historical replay; scaler compatibility warning remains |
| Profiling measurements | Repository `data/processed/dpdk_processed.csv` and `oai_processed.csv` are absent | Recovered at `/home/ubuntu/DDT_UPF_Selection_DRL/data/processed/`; identical SHA-256 copies also in `xRL_UpfSelection` and `dynamic-upf-selection-rl` | External dependency. Recovered CSVs support original rounded accuracy results; package provenance before portability claim |
| Profiling training, `train_and_save_model` | `/home/ubuntu/xRL_UpfSelection/scripts/PowerPerformance/train_power_models.py` and `train_performance_models.py` | Missing from audited Git tree/history; external scripts recover log transform, train-fitted scalers and 80/20 seed-42 split | Read only. Candidate historical sources, not an authenticated training manifest. Both use their evaluation split for early stopping |
| Profiling evaluation, `evaluate_mode` | `scripts/compute_profiling_model_accuracy.py`; root accuracy/prediction CSVs | Script explicitly searches external fallback directories; fresh inference completed | Reusable for auditing fitted models. Validation-set reuse and threshold decisions need new analysis |
| Traffic loader, `load_and_preprocess_data`, calendar/dynamics helpers | `src/utils.py:287` | Trimming precedes 75/25 split; separate episode burn-in; calendar arrays are not trimmed with traffic | Executable. Historical behavior recoverable; cannot reuse unchanged for corrected held-out claims |
| Main/YouTube traffic | Five CSVs under `data/processed/` | Each has 3,360 rows; two underlying workloads, not five independent traces | Replay-ready. Original spatial NetMob data and a complete original aggregation chain are not in this repository |
| Traffic preprocessing and original LSTM | `scripts/traffic/{preprocess_traffic,scale_traffic_files,train_traffic_predictor,predict_traffic}.py`; `saved_models/prediction_models/traffic_predictor*` | Original predictor and scalers present; raw configured input absent; scaler fitted before temporal split | Historical inference reusable; original preprocessing/training is not leakage-safe |
| Later forecast pipeline | `traffic_predictor/run_improved_traffic_predictor.py`, `models/`, `results/`; `processed_traffic_*_bestpredictor.csv` | Separate 65/15/20 workflow and later forecasts; reported 19.12 RMSE comes from this family | Do not substitute silently for checkpoint-training forecasts; selection by test RMSE needs correction |
| Selected policy artifacts | `experiments/results/B1_top2_seeds/{mlp_hybrid,lstm_history}/seed_{42,123,456,789,1024}/` | All ten best/final checkpoints, normalization files, evaluations.npz and configs exist | All ten load and replay. Preserve each run as a bundle; final checkpoint is not the selected policy |
| Other selected runs | `finalModels/{mlp_hybrid,lstm_history}/seed_456/` | A1-derived selections, not a replacement for the whole B1/B2 seed family | Available; keep run identity explicit |
| Policy evaluator, `evaluate` | `src/evaluate.py` | Deterministic predictions; defaults to best_model.zip; broken recurrent detection; post-step forecast logging | Dependencies available. Must correct before a new recurrent comparison; default invocation writes inside run directories |
| Rule/static baselines | `scripts/baseline_rule.py`; `scripts/evaluation_baseline-rule.py`; `experiments/results/paper_results_assets/compile_baselines_table.py` | Multiple different algorithms/slices; threshold selector exists but publication compiler uses direct utility ranking over two modes | Historical functions executable in isolation. Full scripts write results; class constructor writes figures/threshold JSON; not safe as read-only imports of runners |
| Offline planner, `compute_offline_optimal`, `solve_offline_optimal_actions` | `src/offline_optimal.py`; `scripts/run_offline_optimal_baseline.py` | Finite-horizon discounted-reward dynamic program using same guarded transitions | Freshly executed; five tiny solver tests pass. Reward optimum is not an energy lower bound |
| Statistical analysis and figures | `polished_results/Wilcoxon/wilcoxon_analysis.py`; `experiments/results/paper_results_assets/wilcoxon_analysis.py`; `notebooks/*.ipynb` | Multiple table families and conventions; some notebook loaders force 720 rows | Reusable plotting/statistical scaffolding after source-of-truth and timestamp alignment are fixed |
| Manuscript and documentation | `Paper artifacts/manuscript_energyawareupf.tex`, PDF, `references.bib`; root reports and `docs/` | Local manuscript exists but conflicts with supplied submission specification on cost values | Editorial source available; exact submitted-version identity remains unverified. Existing reports are evidence to check, not proof of correctness |

### Git provenance

No tags were found. Local branches are `main` and `ActualSwitchCost`; remote-tracking branches also include the documentation PR. No remote fetch was performed; history findings apply to locally available objects.

- `ade54f6` (17 February): initial framework and saved profiling/forecast artifacts.
- `1d5c949` (18 February): uniform trimming and schema alignment changes in `src/utils.py`.
- `0f4fc8a` (4 March): experiment outputs and environment reset alignment.
- `9f1a004`, `f43a781` (18 March): offline planner, then discounted objective/update.
- `b9ed603` (20 March): separate improved traffic predictor.
- `7608fd1` (22 March): baseline scripts and publication material.
- `a23e7f7` (22 October): pre-revision snapshot, manuscript updates and B2-to-B1 artifact-directory renaming.

The core environment reward logic already exists in the initial framework; the inspected change to its reset makes all schemas start at W−1. The fitted profiling artifacts are unchanged between the initial framework and HEAD. Comparing `ActualSwitchCost` against `main` does not reveal an alternative direction-specific reward implementation in `src/environment.py`; the relevant core addition is the offline planner. The branch name alone is not evidence of the requested cost semantics.

The polished PPO runner still points to `experiments/results/B2_top2_seeds`, now renamed `B1_top2_seeds`. The baseline compiler writes to misspelled `polishe_results`, whereas stored publication artifacts are in `polished_results`. These are concrete rerun hazards.

### Available environment

The existing `.venv` imports successfully: Python 3.12.3; NumPy 2.4.2; pandas 2.3.3; SciPy 1.17.0; scikit-learn 1.8.0; TensorFlow 2.20.0; Keras 3.13.2; PyTorch 2.10.0+cu128; stable-baselines3 2.7.1; sb3-contrib 2.7.1; Gymnasium 1.2.3; pytest 9.0.2; PyYAML 6.0.3.

`requirements.txt` has no version pins. System Python lacks TensorFlow, Keras, SB3, sb3-contrib, Gymnasium and pytest; use of system Python is not equivalent. Checkpoint metadata records Python 3.12.3, SB3 2.7.1, PyTorch 2.10.0+cu128, NumPy 2.4.2 and Gymnasium 1.2.3, with GPU enabled. Profiling scalers were serialized under **scikit-learn 1.7.2**, causing a warning under 1.8.0. No dependency changes were made. CPU replay cannot establish GPU training reproducibility.

### Exact environment and reward semantics

[Environment implementation](/home/ubuntu/EnergyAwareUPF/src/environment.py:351) and [prediction implementation](/home/ubuntu/EnergyAwareUPF/src/utils.py:196):

- Actions are 0=DPDK, 1=USR(1), 2=USR(2). Each slot evaluates the chosen configuration at the observed load for that same slot.
- USR with k instances evaluates one surrogate at load/k, returns its QoS unchanged, and multiplies its power by k. There is no interference, load-imbalance model, uncertainty propagation, or validated physical multi-instance interaction in this computation.
- SEC is total predicted attributed power divided by `max(max(traffic,0),1e-6)`. An idle-subtracted quantity is logged but is **not** used in reward.
- Reward is `−100*SEC −30*max(0,0.90−Q) −type_penalty −scale_penalty`.
- Predictions are not explicitly clipped to a valid power/QoS range. DPDK lookup misses return `(0,0)`; canonical precomputation avoids misses by using exact trace keys. Generic lookup generation uses a different 500-point 0–1000 Mbps grid and is not interchangeable with exact trace precomputation.

| Executed transition | Actual penalty | Relation to supplied reference specification |
|---|---:|---|
| USR(k) → DPDK | 0.030 | Matches c_DPDK numerically |
| DPDK → USR(1) | 0.030 | Does not implement 0.012 per launched USR instance |
| DPDK → USR(2) | 0.030 | Does not implement 2×0.012 for launches |
| USR(1) → USR(2) | 0.012 | Scale-up per added instance |
| USR(2) → USR(1) | 0.003 | Scale-down per removed instance; equals one event here, but differs for larger jumps when k>2 |
| Requested change blocked by cooldown | 0 | Costs apply only to executed changes |

Cooldown increments **before** checking `<4`. After a change at t, requests at t+1, t+2 and t+3 are blocked; another change can execute at t+4. Blocked commands are discarded, not queued. No pending-action variable stores them. A fresh command can be issued next slot. Both mode changes and scale changes reset the counter; unchanged configurations do not. The policy sees the pre-increment counter, so an observation of 3 can precede an accepted change. Equation (3) is consistent only if its u_t means the post-tick counter at actuation time; otherwise its observation/guard indexing differs by one.

The `prewarm` YAML section is not active: the environment sets activation delay to zero and the warm pool to zero. The capacity override is commented out. There is no mid-slot alarm handler, safety override, session transfer, make-before-break implementation, or real orchestrator actuator in this pipeline. Fifteen minutes is the trace/energy integration time base, not a real-time scheduler. The supplied 24 s/3.3 s startup figures do not demonstrate service-continuous transition latency and are not injected into environment transitions or energy integration.

The local manuscript parameter table instead lists **0.5 / 0.2 / 0.05**. This is an additional three-way discrepancy among supplied submission specification, current manuscript source, and actual run configs. The calibration notebook proposes a reward-scaled spike/delay model, but it is not wired into `ManualCooldownEnv`.

### Traffic, forecasts, observations and splits

The files cover 1 April 2019 00:00 through 5 May 2019 23:45: **3,360** quarter-hour rows, consistent with 35×96. The stated 3,359 is not the file length.

Actual PPO preparation is:

`3360 → trim first 96 forecast-invalid rows → 3264 → chronological split 2448/816 → skip 94 rows at env reset and omit final row → 721 evaluated slots`.

The archived RL history window is **95**; the predictor input window is **96**. They must not be described as the same setting.

| Slice | Original CSV indices, zero-based | Actual interval |
|---|---|---|
| RL training array | 96…2543 | 2 April 00:00…27 April 11:45 |
| RL test array | 2544…3359 | 27 April 12:00…5 May 23:45 |
| PPO/offline evaluated slots | 2638…3358 | 28 April 11:30…5 May 23:30, 721 slots |
| Polished rule/static slices | 2639…3359 | 28 April 11:45…5 May 23:45, 721 slots |
| Original manuscript main rule/static CSVs | 2640…3359 | 28 April 12:00…5 May 23:45, 720 slots |

The nominal 75/25 split of the complete file would be 2520/840, not 2519/840. The loader applies a different split after trimming. At 15 minutes per slot, 720, 721 and 840 observations represent 180, 180.25 and 210 hours of integrated power respectively.

Main traffic is ratio-scaled by approximately 0.1935859071, with minimum 4.21107 and peak 400 Mbps. YouTube uses factor 0.1936 and peaks at 295.41692 Mbps. Scaling is performed before the split using the full-series range; there is no runtime hard cap. This is an offline workload-design choice that accesses the complete trace, not a demonstrated deployment capacity. The stored preprocessing input is already aggregated traffic; original NetMob spatial records and an authenticated original aggregation manifest are not supplied in this Git tree.

The old predictor fits MinMaxScaler on the complete series, creates 96-step sequences, and uses an 80/20 split; its nominal test split is also early-stopping validation. Its training targets extend into the suffix used for RL evaluation. The later predictor fits scalers only on its training prefix and uses 65/15/20 splits, but `run_improved_traffic_predictor.py:571` chooses among candidates using **test RMSE**. A separate untouched final test remains necessary.

Hybrid observations contain actual load history through t and forecast[t+1]. With correctly indexed one-step forecasts, the latter can be computed from history through t; however, using observed slot-t aggregate load to choose the action whose reward is evaluated at t requires an explicit online timing convention. It is not evidence that completed slot telemetry is available at the start of that slot.

Additional alignment defects:

- [Calendar construction](/home/ubuntu/EnergyAwareUPF/src/utils.py:472) splits the original untrimmed timestamp arrays using the trimmed traffic split size. Calendar timestamps lag the associated traffic by 96 samples (one day); weekday features are consequently wrong.
- [Forecast logging](/home/ubuntu/EnergyAwareUPF/src/evaluate.py:481) reads the forecast after stepping, so logged forecast values are shifted relative to the decision; auto-reset also contaminates the final logged forecast. This affects diagnostic plots and interpretation, even when policy observations are unchanged.
- Forward/backward filling exists for remaining forecast NaNs. This run has its leading missing block trimmed; backward filling must not be generalized as causal preprocessing without checking gaps.

### PPO and checkpoint selection

All listed hyperparameters match the archived run configs: n_steps=1024, batch=64, epochs=10, lr=1e−4, gamma=.995, GAE=.90, entropy=.15, clip=.20. There are four training environments, target_kl=.03, vf_coef=.5 and max_grad_norm=.5. Four environments ×1024 rollout steps explain why a requested 300,000 budget produces **303,104** final timesteps.

Feed-forward best checkpoints occur at 12k, 8k, 30k, 20k and 32k steps for seeds 42,123,456,789,1024. All corresponding final evaluation returns collapse to approximately −975.4. Best-model selection is therefore implemented and essential. `metrics.json` is produced from the final in-memory model, explaining why seed-42 final metrics report roughly 147.79 Wh/static DPDK rather than the selected-policy result.

[EvalCallback](/home/ubuntu/EnergyAwareUPF/scripts/run_experiments.py:197) evaluates the `test` payload every approximately 2,000 environment timesteps. Its five deterministic episodes repeat the same trajectory; they are not five independent experiments. Selection uses evaluation return, not minimum energy. [Normalization is saved only at training end](/home/ubuntu/EnergyAwareUPF/scripts/run_experiments.py:228), potentially changing how an earlier checkpoint is evaluated.

The batch runner seeds Python, NumPy and Torch explicitly and can enumerate additional seeds. `create_agent` does not pass an algorithm seed; the standalone `src/train.py` does not supply the same explicit global seeding. Do not infer that either entry point is an equivalent reproducibility protocol.

## 3. Reproduction readiness

### Reference results, archives and fresh execution

All energies below are UPF-attributed **Wh**, not measured whole-server electricity. “Archive” means recomputation from existing stored CSVs, not a new experiment. Fresh results are audit evidence, not replacements for manuscript numbers.

| Quantity | Submitted reference | Recovered archive | Fresh execution/status |
|---|---:|---:|---|
| Main offline-optimal energy | 116.21 | 116.206383 | 116.184382 on 721 slots; numerical mismatch remains |
| Main FF-PPO energy, mean±sample SD | 122.58±1.50 | 122.5786±1.5002 (polished table) | 122.555792±1.498715; identical actions, differing power |
| Main FF-PPO SEC | .00748±.00015 | .007480±.000152 | .007477440±.000151797 |
| Main FF-PPO violation rate | .00333±.00362 | Rounded table .00334; raw-slot counts agree with reference rounding | .003328710±.003616756 |
| Main recurrent energy | 125.29±5.11 | 125.2936±5.1059 | 125.276896±5.111298 using existing defective evaluator semantics |
| Main rule energy | 123.92 | **123.923325**, original 720-row CSV; polished 721-row version 124.320879 | Current compiler: 124.309506/721; slice-reconstructed 124.104299/720. Original value not reproduced by current rule calculation |
| Main rule SEC / violation | .00775 / .0014 | .007750842 / .001388889, original 720 rows | Current 721-row compiler .007778325 / .001386963 |
| Main static DPDK energy | 147.59 | **147.586982**, original 720 rows | **147.586983** using same 720 timestamps; canonical PPO slice gives 147.792460 |
| Main static USR(1) energy | 178.36 | **178.357874**, original 720 rows | 178.318018 on reconstructed 720 slice; canonical PPO slice gives 178.888887 |
| YouTube FF-PPO energy | 124.47±1.60 | 124.4696±1.6040 | 124.447907±1.606708; identical actions |
| YouTube rule energy | 123.57 | 123.570, polished table | Current compiler 123.482963/721; numerical/implementation reconciliation remains |
| YouTube static DPDK energy | 147.78 | 147.779, polished table | 147.779274/compiler slice; 147.779574/PPO slice |

The two static-USR audit rollouts using the environment begin with its standard DPDK reset and request USR immediately; power for every evaluated slot is USR power. Their logged initial transition is not a physical startup-energy measurement. Compiler static baselines have zero reconfiguration counts.

| FF seed | Reference Wh | Archived step-energy sum Wh | Fresh replay Wh | Executed-action differences |
|---|---:|---:|---:|---:|
| 42 | 122.94 | 122.934401 | 122.915281 | 0/721 |
| 123 | 122.33 | 122.320753 | 122.302036 | 0/721 |
| 456 | 120.79 | 120.790018 | 120.768054 | 0/721 |
| 789 | 121.98 | 121.978495 | 121.951042 | 0/721 |
| 1024 | 124.88 | 124.869608 | 124.842544 | 0/721 |

The reference values for seeds 42,123,1024 are not the ordinary two-decimal rounding of these archived step sums. Their origin should be reconciled rather than silently rewritten. All five seeds are recoverable; independent retraining was **NOT TESTED**.

### Numerical acceptance rule

No arbitrary “close enough” tolerance was used. The polished compiler rounds each per-slot energy to six decimal places. Summing N such values can differ from an unrounded sum by at most `N×0.5×10^-6 Wh`, plus negligible summation error: 0.000360 Wh for 720 rows and 0.0003605 Wh for 721. The reconstructed static-DPDK result differs from its full-precision archive by about 0.00000051 Wh and passes this **serialization-only** check on the same timestamps.

By contrast, the FF-PPO replay differences are approximately 0.019–0.027 Wh on Main, far beyond that bound. Archived power is rounded to six decimals, allowing at most 0.0000005 W of serialization error; observed maxima reach about 0.000665 W. These are unresolved inference/provenance differences, not justified rounding. Hardware execution, Keras/TF behavior, and the scaler version mismatch are candidates, not established causes. Repeated-training variation is a separate issue and was not tested.

### Profiling-model accuracy and threshold residuals

The externally recovered CSVs contain 4,594 DPDK and 4,564 USR rows, with offered loads from 0.01 to 5,000 Mbps. Applying the documented 80/20 shuffled seed-42 split gives 919 and 913 evaluation rows.

| Model | Reference RMSE / R² | Fresh fitted-model inference RMSE / R² | Finding |
|---|---|---|---|
| DPDK QoS | .0847 / .153 | .08465653 / .15285176 | Matches reported precision |
| DPDK power | .0039 / .239 | .00386169 / .23929525 | Matches reported precision |
| USR QoS | .0785 / .907 | .07847765 / .90681015 | Matches reported precision |
| USR power | .3038 / .960 | .30383960 / .96046663 | Matches reported precision |

This recovers rounded metrics, not an independently untouched model test: the external training scripts use the same split for early stopping. Row-level splitting also does not establish independence between temporally adjacent samples from a profiling run.

DPDK QoS: 103 measured observations are below .90; **41 are predicted safe**. Another 57 measured-safe observations are predicted unsafe. For the descriptive band `|Q_measured−.90|≤.05`, there are 91 observations and RMSE .036784. A training-mean constant predictor gives RMSE .091988, versus .084657 for the MLP: some predictive value exists, but this does not settle threshold reliability. Archived residuals restricted to offered load≤400 Mbps still contain **27 false-safe observations among 629 DPDK evaluation samples**; these are not solely errors beyond the paper's workload range. Counterfactual controller-ranking changes under model uncertainty were **NOT TESTED**.

Raw grouped measurement files exist externally, as do candidate training scripts. The exact original label-generation/measurement provenance is incomplete: the audited repository consumes `QoS_w` rather than constructing it. Its unused configuration weights differ from the manuscript's .4/.4/.2 latency/jitter/loss weights. An external exploratory QoS notebook uses .6/.2/.2 and different thresholds, but does not establish how the saved CSV labels were generated. Do not substitute that notebook or a newer scoring function as the original pipeline.

### Forecast accuracy

The archived later-model prediction file recomputes to **RMSE 19.11999963, R² .89795267, N=672**, matching 19.12/.898. Its test targets start at original row 2688 (29 April 00:00). Fresh inference of this saved later predictor was **NOT TESTED**. The older predictor's stored evaluation report gives RMSE **36.145669**, R² **.637611**, N=653, and no calendar features. Its model/CSV identity must not be conflated with the later reported model.

### Check outcomes and evidence

- `tests/test_offline_optimal.py`: **5 passed**, caches disabled, temporary output isolated.
- Twenty saved-policy rollouts: completed; action equality verified against all corresponding polished traces.
- Static, rule and offline planning calculations: executed separately from training. Existing pure baseline function definitions were extracted without running their file-writing top-level code; the 720-row calculation is explicitly a slice reconstruction, not proof that the original baseline generator was recovered.
- Profiling inference and archived forecast arithmetic: completed.
- Full training, new seeds, corrected recurrent-state evaluation, new traffic, QoS-label regeneration, physical switching and uncertainty propagation: **NOT TESTED**.

Machine-readable evidence: [replays](/home/ubuntu/EnergyAwareUPF/docs/audits/2026-10-22/replay_results.json), [baselines](/home/ubuntu/EnergyAwareUPF/docs/audits/2026-10-22/baseline_results.json), [profiling residuals](/home/ubuntu/EnergyAwareUPF/docs/audits/2026-10-22/residual_results.json), [87-file SHA-256 manifest](/home/ubuntu/EnergyAwareUPF/docs/audits/2026-10-22/artifact_manifest.json). The directory also contains the read-only inference harnesses and logs.

## 4. Reviewer action matrix — all 13 points

Priorities: **P0** resolve before comparative revision experiments; **P1** essential reviewer response; **P2** editorial/optional work that can proceed independently. Section numbers refer to the local manuscript; exact submitted numbering should be checked against the submission PDF.

| ID / manuscript section | Required fix, including code need | Relevant components | New experiment? | Reusable infrastructure / recovery dependency | Difficulty | Suggested validation | Priority |
|---|---|---|---|---|---|---|---|
| **R1.1** Title | Replace “AI-driven” with digital-twin/DRL terminology; no code | Manuscript title, abstract | No | Manuscript source usable; no recovery dependency | Low | Title must state offline digital-twin scope without implying deployed control | P2 |
| **R1.2** §2 Related work | Broaden verified survey and add comparison table; no experiment code | `references.bib`, §2 | No ML experiment | Bibliography usable after citation verification; does not depend on numerical reproduction | Medium | Read primary papers from requested venues; fill comparison dimensions only from supported evidence | P1 |
| **R1.3** §§3.3–3.5, 4.1, 5.1, 6.1 | State actual observation/actuation timing and limitations. Code only if claiming new runtime mechanisms | Environment reset/step, forecast indexing, inactive guard/prewarm settings | No for an accurate offline explanation; yes for live responsiveness claims | Offline environment reusable; online actuator/safety system absent | Medium for response; high for deployment | Timeline of what is known at decision time; distinguish 15-min slot, cooldown, startup delay and service continuity | P1 |
| **R1.4** §4.5, §§7.2–7.8 | Reward decomposition and appropriate alternative algorithm; diagnostic code and Dueling DQN implementation needed later | `environment.step`, logged penalties; `agent.py`; sweeps | Yes: analysis and fair alternative training | Logs/twin reusable after P0 fixes; discrete action space compatible with DQN | Medium diagnostics; high comparator | Per-load component magnitudes, low-load epsilon behavior, identical data/actions/reward/budget and independent validation | P1 |
| **R1.5** Abstract, §§1–6 | Define acronyms at first use; proofread units and terminology; no code | Manuscript `.tex`, `.bib` | No | Available manuscript; exact submitted version must be identified | Low | Abstract and main-text acronym pass separately; remove claims stronger than evidence | P2 |
| **R3.1** §§7.9, 7.11 | More independent seeds, defensible CIs/effects and testing; repair protocol before extension | `run_experiments.py`, B2 sweep, both Wilcoxon scripts | Yes, after recovery | Seed enumeration/seeding exists; current sweep and test selection unsafe unchanged | Medium plus compute | Prespecified sample size/endpoints; seed-level differences; no pseudo-replication or stopping when significant | P1, P0 prerequisites |
| **R3.2** §5.3, §7.4 | DPDK residuals near tau, simple predictors, downstream selection impact; analysis code needed | Profiling CSVs external; fitted models; `compute_profiling_model_accuracy.py` | Residual verification partly done; independent calibration/selection-impact experiments needed | Data/model/split recoverable; label and campaign provenance still incomplete | Medium–high | Train-mean/simple-model comparison, threshold confusion, load/run-held-out residuals and decision robustness | P1 |
| **R3.3** §§4.5, 6.2, 7.8 | Sensitivity to all five weights; extend sweep mapping for `sec_scale`; resolve mode/launch semantics | Reward YAML, `environment.step`, `apply_overrides` | Yes: retrain for each reward setting | All five numeric weights configurable in YAML; alpha absent from sweep mapping; costs do not implement requested launch semantics | Medium plus compute | Energy, SEC, violations, utilization, mode/scale changes, blocked requests and component magnitudes; prespecified ratios/seeds | P1 |
| **R3.4** §§6.1, 7.10 | Limit generalization claims and add genuinely different traffic if available | Five processed CSVs, traffic predictor, YouTube evaluations | Optional new data/evaluation; no experiment needed to correct wording | Only two underlying local traces; no verified independent period/city/dataset found in audited tree | Medium; high if data acquisition needed | Distinguish same-source cross-service, cross-period, cross-location, cross-dataset; freeze scaling and predictor training source | P1 wording; P2 expansion |
| **R3.5** §§5.1–5.3, 6.1, 7.8.3 | Explain 400 Mbps and k_max=2 as study choices; no parameter increase during audit | Scaling scripts, CSV ranges, `num_oai_instances`, twin | No for honest scope; yes before expanded operational claims | Measurements extend above 400, but policy/domain/multi-instance validity is not proven there | Low for explanation; medium–high validation | Range/units and sampling coverage, saturation residuals, load balance and k-dependent physical behavior | P1 |
| **R3.6** §7 | Combine related result subsections; no code/model changes | Results source, plots/table generators | No | Editorial organization independent; final numeric claims depend on P0 work | Low | One coherent story with robustness/ablations grouped and limitations explicit | P2 |
| **R3.7** §3.4 Eq.(3), §§4.1/4.5 | Explain discarded blocked requests and tick convention; add focused guard checks later if formalizing contract | `environment.py:351–441`; planner transition tables | No training needed | Exact behavior recovered; rule selector has a different internal counter convention | Low–medium | Enumerate requests at t…t+4; verify no queue, reset only on executed change, zero blocked-action cost | P0 |
| **R3.8** §§4.5, 6.2, 7.1 | Clarify reward penalties, SEC units and unmodeled startup energy; reconcile numeric costs | Reward config, environment, calibration notebook, manuscript table | Diagnostics only; cost calibration experiments if claiming physical mapping | Logged terms reusable; no implemented measured startup-to-reward conversion | Low–medium | Unit derivation and component table; distinguish heuristic penalties from Wh and transition measurements | P0 |

### R1.1 — suggested titles

1. **Digital-Twin-Based Deep Reinforcement Learning for Energy-Aware UPF Orchestration in 5G Core Networks**
2. **Energy-Aware UPF Selection and Scaling in 5G Core Networks Using Deep Reinforcement Learning**
3. **Offline Digital-Twin Evaluation of PPO for Energy-Aware UPF Orchestration**

The first balances specificity and scope; the third is most explicit about what has been validated.

### R1.2 — comparison-table dimensions

Use one row per verified primary study and columns for: venue/year; network function and decision scope; learning/control method; energy quantity and measurement basis; QoS objective versus hard constraint; physical/digital-twin validation; heterogeneous UPF selection; horizontal scaling; reconfiguration-cost model; cooldown/actuation constraints; forecasting/observation timing; online versus offline evaluation; traffic diversity; and reproducibility/statistical evidence.

Search TWC, TCOM, TMC, Computer Communications, Computer Networks and relevant related venues. Separate radio-resource studies from UPF/VNF orchestration rather than presenting them as identical tasks. This audit does not fabricate an expanded bibliography or claim the literature review is complete. Existing citation keys require verification against the underlying papers before new comparison claims.

### R1.4 / R3.8 — reward balance and units

For freshly replayed main seed 42, over 721 slots:

| Reward magnitude | Mean per slot | Sum |
|---|---:|---:|
| `100*SEC` | .760088906 | 548.024101 |
| QoS shortfall penalty | .000117561 | .084761 |
| Type-switch penalty | .000748960 | .540000 |
| Scale penalty | .000016644 | .012000 |

The SEC term dominates the realized magnitude for this policy, but a small realized QoS penalty can reflect avoidance of bad actions; it does not prove the QoS weight is unimportant. Compare counterfactual candidate actions at identical loads/states, particularly where their QoS crosses .90.

Observed evaluation load never reaches epsilon (minimum 4.21107 Mbps). Epsilon=1e−6 therefore does not change this trace's denominator, but a zero/near-zero-load diagnostic is necessary before broader low-load claims. At fixed positive power, SEC increases inversely with load; this can make low-load slots dominate the objective. The denominator is **offered**, not successfully delivered, traffic, so service shortfall is represented only through the QoS penalty.

The code produces `W / (Mbit/s) = J/Mbit`. Values such as .00748 are consequently **J/Mbit**, numerically W/Mbps. Converting to Wh/Mbit requires dividing by 3600. The manuscript repeatedly labels these values Wh/Mbit without this conversion. Energy totals, computed as `sum(P_W*0.25 h)`, are correctly in Wh. The mean of per-slot SEC ratios also differs from total energy divided by total transferred traffic; specify which metric is optimized/reported.

Proposed explanatory wording: “Reconfiguration coefficients are additive penalties on the reward scale. They are applied only to executed changes. They represent a chosen preference against reconfiguration and are not measured startup energy in joules or watt-hours. Startup delay and transient power are not integrated into the present twin.” Alpha carries the numerical scaling needed to combine SEC with these penalties.

For the alternative algorithm, a Dueling DQN with the same three actions is appropriate. Its defining value/advantage architecture must actually be implemented; using a generic DQN class does not automatically constitute Dueling DQN. See the [original dueling-network paper](https://proceedings.mlr.press/v48/wangf16.html). Standard TD3 is designed for continuous actions; mapping a continuous output to three discrete configurations changes the method and adds an arbitrary mapping, so it is not a straightforward comparison here. The [TD3 documentation](https://spinningup.openai.com/en/latest/algorithms/td3.html) explicitly restricts it to continuous action spaces.

### R3.1 — significance and independent replicates

Archived polished Wilcoxon outputs recover energy p=.125 (4/5 lower-energy seeds) and SEC p=.0625 (5/5 lower-SEC seeds). However, those outputs compare against **124.321 Wh/.00778**, while the manuscript overall table uses **123.923325 Wh/.007750842**. The reported energy effect size −1.7424 Wh and SEC percentage −3.86% refer to the former baseline. Matching p-values does not reconcile mixed baseline references.

With five nonzero untied differences, the minimum exact two-sided signed-rank p is `2/2^5=.0625`; it is not significant at .05. Increasing seeds can improve estimation, but cannot repair leakage or a mismatched evaluation horizon.

The experimental unit is an independently trained policy evaluated on a fixed, prespecified workload. The deterministic rule value can be subtracted from each seed's outcome for a one-sample signed-rank analysis of differences; it is not a set of independent baseline training replicates. Identical seed numbers across different algorithms do not by themselves justify a special paired design. State the pairing mechanism and evaluate all methods on identical workloads. Signed-rank inference also assumes an appropriate symmetric difference distribution; see the [SciPy Wilcoxon documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).

Do not treat individual correlated traffic slots or five repeated deterministic evaluation episodes as independent samples. Report absolute/relative effects and uncertainty across seeds, describe conditional-on-workload limitations, and use a prespecified sign/permutation procedure if signed-rank assumptions are not credible. Include uncertainty for bounded QoS violation rates without presenting impossible negative rates as physically meaningful. Correct for planned multiple comparisons or designate primary/secondary endpoints. Choose 10, 15 or 20 seeds in advance from an effect-size/precision goal; do not keep adding seeds until p<.05.

### R1.5 / R3.6 — manuscript inspection and organization

The local abstract uses DPDK and PPO without expansion. PPO is expanded in the introduction; NWDAF and MDAF occur in the introduction before definitions in §3.3. No “Data Plane Development Kit” expansion was found in the inspected source. LSTM first appears in §6.1 without expansion. Check 5QI, NFV, SBA, IDAF, MLP, RAPL and API similarly. UPF and QoS are expanded in the abstract. Correct the README's claim that SEC is maximized/throughput-per-watt: the implementation minimizes power-per-offered-throughput through a negative reward.

The local source has 12 result subsections including “Summary of Findings,” whereas the reviewer describes 11. This supports verifying the submitted document version before mapping every editorial instruction literally.

Proposed Section 7 structure:

1. Evaluation protocol and metrics, including exact slices and checkpoint selection.
2. Main comparison: energy, SEC and QoS together.
3. Orchestration behavior and stability.
4. Learning behavior and ablations: policy type, observations, convergence.
5. Robustness and sensitivity: seeds, statistical analysis, reward weights, cooldown, tau and k_max.
6. Cross-trace evaluation and scope of generalization.
7. Limitations and implications of the deterministic twin.

This is an outline only; no paper text was rewritten.

## 5. Scientific risks and internal consistency

**Critical — validity of recurrent comparison.** [src/evaluate.py:430](/home/ubuntu/EnergyAwareUPF/src/evaluate.py:430) uses `getattr(model.policy, "is_recurrent", False)`. Loaded RecurrentPPO policies return false. The fallback calls `predict(obs)` without recurrent state, resetting memory each decision. Training uses RecurrentPPO, but published evaluation does not exercise its intended temporal state. History features still contain past traffic; this does not restore recurrent-state semantics. Re-evaluation with proper recurrent state is a necessary later correction and was deliberately **NOT TESTED** as a replacement result in this audit.

**Critical — selection leakage.** The same RL suffix selects checkpoints, observation schemes and some favorable configurations, then supplies reported performance. The later forecast model is also selected by test RMSE. Chronological splitting alone does not make these estimates untouched tests. Correcting this requires a distinct validation/final-test protocol and probably a newly trained campaign; replay alone cannot undo selection.

**Critical — incompatible horizons and unfair scaling access.** The main 720-slot baselines and 721-slot PPO are not aligned by timestamp. The polished 721-slot baselines remain shifted one slot. Several notebook functions remove rows by count rather than establish a common timestamp set. In addition, archived config disables multi-USR for the baseline and the publication compiler hardcodes DPDK versus USR(1). This cannot support the claim that PPO chooses scaling over switching more intelligently than a baseline with equivalent scaling opportunities.

**High — baseline identity.** The manuscript describes threshold regions with a dead-band. `RuleBasedSelector` implements such logic, but the evaluation script instantiates it and then uses a separate inline utility decision; its `decide` method is not called in that loop. The compiler likewise directly ranks two utilities and omits that dead-band. Its QoS threshold is used for reporting, not a hard feasibility guard. `RuleBasedSelector.decide` holds four subsequent calls before permitting another change, unlike the environment's pre-increment counter allowing a change at t+4. Resolve which baseline is intended before claiming identical guard semantics. Utility terms also mix QoS approximately in [0,1] with an efficiency term scaled to [0,100]; nominal .3/.7 coefficients do not mean a comparable 30/70 contribution.

**High — manuscript/forecast/reward drift.** User-supplied costs, local manuscript costs and actual transition costs differ. The excellent later forecast metric is not the forecast artifact used in original policy training. The B2 sweep threshold differs from every archived B1/B2 snapshot. These are scientific configuration changes, not cosmetic path fixes.

**High — model error is shared but does not cancel.** Controllers query the same deterministic fitted models, so comparisons are internally conditional on that twin. Policies select different modes/loads and can exploit different surrogate errors. Shared errors need not cancel, and offline optimization may exploit them most strongly. The recovered DPDK threshold errors show why low R² cannot be dismissed as harmless stability.

**High — incomplete provenance.** External measurement CSVs and fitting scripts recover much, but the exact original QoS-label builder, complete physical campaign manifest, original spatial traffic aggregation inputs, authenticated submission snapshot, original dependency lock and best-checkpoint normalization snapshots are not established. External copies must be recorded by identity, not merged indiscriminately into the old pipeline.

**High — online causality and continuity are unproven.** Slot-t load is available to a decision scored on slot t. No sub-slot degradation response or transition interruption exists. The offline QoS penalty is soft, and cooldown may block a protective change. Avoid calling this a proven online safety mechanism. Startup timing is not end-to-end session continuity.

**Required claim corrections:**

- The supplied PPO violation mean .00333 is greater than rule-based .0014: about **0.193 percentage points** higher, roughly 2.38×. Say “low absolute violation rates, with a higher observed mean than the rule-based baseline,” not “without increasing violations.” The currently stored abstract already says “low QoS violation rate,” rather than the exact stronger phrase in the supplied concern; nevertheless all comparative wording needs this qualification.
- YouTube mean energy 124.47 Wh is **0.90 Wh higher** than the 123.57 Wh rule baseline, about .73%. It supports transfer relative to static provisioning, not energy superiority over rule-based control on that trace. This is same-dataset, same-city, same-period cross-service transfer, not cross-dataset generalization.
- The exact planner optimizes discounted reward, including SEC, QoS and reconfiguration penalties. Its reward value is an upper bound under the modeled finite-horizon problem; its 116.21 Wh energy is **not a mathematically certified energy minimum** or a physical deployment bound.
- The local manuscript clearly calls power process-attributed and describes Scaphandre/RAPL-derived estimates. Preserve that qualification in abstract, tables and captions. GPU training-energy logs concern a different quantity. Whole-server electrical savings were **NOT TESTED**.
- The current paper's main FF-PPO switch count 16.4±3.5 differs from the archived Main aggregate 16.8±3.9; 16.4±3.5 is the YouTube aggregate. This is another table-provenance inconsistency.
- A 400 Mbps traffic cap and k_max=2 do not certify an operational envelope. External measurements extend to 5,000 Mbps, but their coverage, packet characteristics, saturation errors and multi-instance physical behavior need validation before extrapolating the policy claim. Current code has no enforced validated upper bound; its commented capacity guard and 700-Mbps setting are not evidence of one.

## 6. Minimum recovery plan

These are proposed next-stage actions, **not implemented fixes**. Preserve the historical artifacts and retain separate labels for historical reproduction and corrected revision results.

1. **Establish one authoritative experiment manifest.** Associate each manuscript table/figure with exact input CSV/hash, original timestamp set, code revision, checkpoint, normalization file, forecast identity, reward semantics and reported units. Use the supplied specification as the reference, while explicitly recording conflicting local source values. The audit's 87-file hash manifest supplies a starting point.
2. **Recover missing provenance as dependencies.** Record the external measurement CSVs and candidate fitting scripts without merging newer code. Identify the actual QoS_w builder, measurement campaign identifiers, raw traffic aggregation path, submitted PDF/source revision and historical environment. Recover checkpoint-time normalization snapshots if they exist; otherwise document that exact selection-time observations cannot yet be reconstructed.
3. **Close the numerical replay gap.** In an isolated environment, after authorization for any dependency installation, reproduce profiling inference with serialized-model versions and compare per-load prediction vectors. Accept identical timestamps/actions first, then justify numerical limits from precision/serialization. Do not adopt the audit's slightly different energies as new historical truth.
4. **Reconcile the historical baseline generator.** Recover the implementation that produced 123.923325 Wh, then explain the 720/721 slicing, configuration and forecast differences. The current utility compiler with a 720-row slice does not reproduce that number. Preserve its archived value as historical evidence until explained.
5. **Define a corrected evaluation contract.** Decide observation timing, one timestamp-identical horizon, proper recurrent state, calendar alignment, forecast logging, baseline action parity/dead-band, cooldown tick convention, reward launch costs and units. Add focused contract checks only after approval; do not silently “fix” the historical pipeline before reproducing it.
6. **Separate training, validation and final testing.** Select forecast models, hyperparameters and checkpoints using training/validation only, save checkpoint-paired normalizers, and keep final testing untouched. Establish repeatable seed execution and freeze a revised configuration. The old selected checkpoints can support historical analysis but cannot become independent held-out evidence retroactively.
7. **Run a small corrected baseline/policy verification before scaling up.** Verify deterministic baselines on exactly identical timestamps, then the selected saved seed-42 policy under explicitly labeled corrected semantics. Only then authorize the full revised training campaign and reviewer experiments.

The smallest current blocker is not missing PPO weights: it is the lack of one consistent experiment identity linking the published comparisons. A manifest and reconciliation come before more model training.

## 7. Proposed revision experiments

Reuse is feasible subject to the recovery gate above. None of the following training/sensitivity/alternative experiments was launched.

| Order | Experiment | Design and outputs | Dependency / interpretation |
|---|---|---|---|
| 1 | Reward-component diagnostics | Bin realized and counterfactual rewards by offered load; report distributions of alpha×SEC, QoS shortfall and executed-change costs; separate blocked requests; include zero/near-zero synthetic loads and epsilon checks | Existing logs largely sufficient; counterfactual states require consistent twin. A diagnostic is not a new trained-policy sensitivity result |
| 2 | DPDK QoS reliability | Reproduce residuals with fixed measurement identities; compare train-mean/constant and simple fitted models; inspect threshold confusion and errors by load/campaign; evaluate whether plausible residuals change preferred configurations | Fitted models/data recovered. Keep additional fitting confined to training/validation; correlated-run splitting and QoS-label provenance need resolution |
| 3 | Independent PPO seeds | Prespecify 10, 15 or 20 total seeds per selected variant with equal interactions, proper recurrent evaluation and independent validation/final test; report distributions, effects and confidence intervals | Existing five labels are available, but do not pool old leaked/defective results with a corrected campaign. Increasing total from five would mean 5/10/15 additional runs per variant only if the underlying protocol remains comparable |
| 4 | Reconfiguration/reward-weight sensitivity | Start with one-at-a-time 0.5×,1×,2× around each of the five agreed weights; add a prespecified switch-to-scale cost-ratio study if needed. Use matched seed lists and retrain each setting; report energy, SEC, violations, mode shares, scaling, switches, blocked actions and reward components | For five seeds, baseline plus ten noncentral settings is 55 runs per policy family. This is an optional budget proposal, not a launched sweep. `sec_scale` needs sweep support. Relative switch/launch semantics must be resolved first |
| 5 | Dueling DQN comparison | Same twin, action set, observation information, reward and cooldown; equal environment-interaction and tuning budgets; same train/validation/final-test conditions; report sample efficiency and wall time as well as final metrics | Needs an actual dueling-network implementation. PPO is on-policy and DQN off-policy; matching interaction budgets does not automatically match optimizer work, so report both. Do not force standard TD3 into this discrete task |
| 6 | Traffic diversity | First verify no-retraining YouTube transfer with frozen preprocessing; optionally add one independently sourced period, location or dataset | Existing files only establish two service traces. Any new source must have compatible units, aggregation, valid load envelope and clearly stated transfer category |

For every reward setting, **evaluation-only reweighting of existing trajectories is an accounting/counterfactual analysis**. It does not measure how a policy trained with those weights would behave. New trained-policy sensitivity results require retraining.

A fair scaling baseline should be included before interpreting horizontal-scaling preference as an algorithmic advantage. Equal action availability is necessary but not sufficient: it must receive comparable observation timing, use the same actuation constraints, and have its own reasonable validation-based parameter selection.

## 8. Estimated work

Engineering estimates are indicative working days for someone familiar with the project; they exclude new physical profiling/data-access delays and are not runtime guarantees. There are 27 calendar days from this audit date to the stated deadline.

| Task | Effort | Indicative engineering time | Major dependency |
|---|---|---:|---|
| Artifact/table/timestamp manifest and historical reconciliation | Medium | 1–3 days | Authoritative submitted document and baseline provenance |
| Numerical inference/version reproduction | Medium, possibly high | 1–3 days | Historical package/backend versions; authorized isolated environment if needed |
| Evaluation-contract corrections and focused checks | Medium–high | 2–4 days | Decisions on timing, costs, recurrence, validation split and baseline parity |
| Reward diagnostics and units explanation | Low–medium | .5–1.5 days | Agreed semantics and aligned logs |
| Additional-seed orchestration and statistics | Medium | 1–2 days plus training | Frozen corrected protocol, sample-size plan and checkpoint-normalizer pairing |
| Five-weight sensitivity campaign | Medium | 2–3 days plus training | Sweep mapping for alpha; agreed ratios and meaningful launch costs |
| DPDK residual/simple-baseline/decision-impact analysis | Medium–high | 2–4 days | QoS-label provenance, campaign grouping, uncertainty method |
| Dueling DQN baseline | High | 3–5 days plus training/tuning | Verified implementation and fair information/budget controls |
| Additional traffic evaluation | Medium–high | 1–3 days after data availability | Independent trace, unit/scaling provenance and validated model coverage |
| Titles/acronyms/Section 7 organization | Low | .5–1 day | Submitted manuscript identity; final claims depend on corrected evidence |
| Expanded verified related-work comparison | Medium | 1–3 days | Access to primary papers and editorial synthesis |
| Live sub-slot safety or service-continuous switching | High | Not reliably estimable from repository | Physical orchestration/session-transfer implementation and testbed access; not a prerequisite for honestly describing the present offline scope |

Existing metadata suggests roughly 7.5 minutes for one archived feed-forward run and approximately 12 minutes for recurrent runs on the original machine. These are historical planning inputs only. Hardware availability, validation frequency and revised workloads can change runtime substantially; no new training throughput benchmark was run.

## 9. Exactly one immediate technical next action

**Create a read-only, timestamp-indexed provenance manifest for the manuscript's main comparison table, linking every row to its exact 720- or 721-slot CSV, checkpoint/normalizer bundle, forecast file and configuration hash.**

This directly resolves the most consequential ambiguity before any implementation decision: which experiment each reported number describes. Start from the archived 720-slot baselines, the B1/B2 five-seed outputs, the offline-optimal CSV and the hash manifest produced by this audit. Do not retrain or edit results as part of that step.

**Audit complete. Stop here; no reviewer fixes or manuscript changes have been implemented.**

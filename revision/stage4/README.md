# Stage 4: full-budget convergence evaluation

No Stage 3 source, frozen twin, reward, score, baseline rule or evaluation
contract is changed. `protocol.json` was written before forecasting/training;
`training_protocol_frozen.json` additionally pins the resulting forecast ledger
and implementation before the ten full controller runs.

Exactly two LSTM candidates retain Stage 3's 96-slot, 128/64 stacked hidden
architecture and six known calendar features in the calendar variant. Maximum
100 epochs, patience15, scaled-MSE min-delta1e-6, plateau reduction after6 epochs,
lr.001 down to.00001, seed42, batch128. Full-training-prefix candidates stop and
select using Netflix validation only. No controller result selects forecasts.

For each expanding prefix, a chronological inner holdout (last20% of available
sequence targets) supplies stopping information. Scalers and gradient-fitting
labels use only the earlier prefix subset. Both gradient and stopping labels
precede every training forecast issued from that prefix. No outer validation
labels are used for prefix early stopping. The first96 scored training slots
retain the explicit Stage3 persistence cold start. Validation model selection
uses validation labels, so validation accuracy is exploratory/selected, not
untouched evidence; input windows remain causal. The validation ledger records
parameter-fit and model-selection cutoffs separately. A persistence alternative
is saved explicitly rather than silently replacing the LSTM.

Five seeds42,123,456,789,1024 per algorithm are freshly trained for exactly300000
interactions. Stage3 PPO hyperparameters/normalization, four collectors, hybrid
observation and recurrent state handling are retained. The last rollout is248
steps/environment rather than1024, giving992 interactions after299008. Its
advantages and optimizer updates are computed normally, with a final32-example
minibatch when necessary. No parameter resets, environment reset, dropped tail
samples or unreported303104-step overshoot occurs.

Checkpoints occur every24576 interactions (six regular rollouts), plus32768 for
a within-forecast matched pilot-budget contrast, and300000 after the final update.
Each policy is paired with its current training-only VecNormalize state. Selection
maximizes validation total reward with earlier checkpoint as tie-breaker.

`training_updates.jsonl` records optimizer losses, entropy loss, approximateKL,
clip fraction, explained variance, update count and actual rollout size **after**
each optimizer update, with the correct interaction timestamp. Standard SB3 CSV
logs are also retained; the SB3 loop dumps logs before updates and can label the
previous update's diagnostics at a later rollout timestamp. Use the explicit
JSONL for precise alignment. Per-collector Monitor CSVs and full-episode JSONL
record stochastic training rewards, energy, QoS penalties and orchestration
behavior. Four identical-workload collectors are not physical independent samples.
The incomplete final120-slot episode per collector is retained separately.

Independent seed jobs may run concurrently in separate CPU processes, each with
one Torch thread. No training state is shared. A failed seed is recorded and is
not silently replaced, removed or retried. All intermediate bundles are preserved.

Forecast sensitivity replays fixed selected policies/normalizers with Stage3 or
persistence forecasts, and fixed Stage3 policies with Stage4 forecasts. This
measures inference sensitivity; the ten primary training runs do not identify a
complete training-budget-by-forecast factorial effect. A local scaling diagnostic
compares executed power with holding the previous USR level at the same slot's
realized load. It is accounting evidence, not a feasible reoptimized no-scaling
policy or a causal scaling ablation. No extra RL training is hidden in diagnostics.

Commands (from repository root, fresh output workspace):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage4.forecast
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest revision/tests -q -p no:cacheprovider > revision/reports/stage4_preflight_tests.log
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage4.launch
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage4.evaluate
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage4.evidence
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest revision/tests -q -p no:cacheprovider
```

Writers reject existing model/manifest paths. Preserve all evidence and use a
new workspace for reproduction, with pinned prior artifacts and external data.
No final-test environment, outcome query, manuscript edit or Git mutation is
part of these commands. See the consolidated report for interpretation and the
actual test/execution records.

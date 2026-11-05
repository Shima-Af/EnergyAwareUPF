# Corrected controller pilot v1

This package explicitly imports `comcom.revised_twin.v1`; it does not call the
historical training/evaluation entry points. See `protocol.json` for the fixed
four-run budget and all pilot choices. Nothing here authorizes final testing.

The command sequence from the repository root is:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage3.forecast
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest revision/tests -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage3.pilot
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage3.sensitivity
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage3.report
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest revision/tests -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage3.seal
```

Experiment writers use exclusive creation for manifests/checkpoints; reproduction
requires a fresh output namespace/workspace containing the frozen prior artifacts.
Do not delete or overwrite prior evidence to rerun an experiment. The source CSVs
still require external backup. Scientific dependencies are pinned by manifests
and recorded runtime versions; no package installation is needed.

## Timeline and forecasting

The approved training/validation split and 95-slot history/96-slot predictor
window are reused. Forecast `[t]` is issued at interval start `t`, observes only
intervals through `t-1`, and predicts the subsequent outcome load `[t]`. Calendar
features refer to the already known decision time. A ledger records the exact
forecast input cutoff and training-label cutoff separately.

The recovered historical saved LSTM is SpatialDropout(.1), LSTM(128), LSTM(64),
Dropout(.2), Dense(64,ReLU), Dropout(.2), Dense(1,linear); the training script
fits a whole-series MinMaxScaler, then reuses a chronological evaluation suffix
for early stopping. A later predictor/calendar family was selected by test RMSE.
Neither pipeline or archived forecast column is reused as corrected evidence.

For this fixed pilot, the hidden widths are retained. Recurrent dropout is set
to zero rather than historical .2; input/output dropout remains .2. This permits
an efficient CPU recurrent implementation and is an explicit regularization
change. Scaling divides by the current training prefix's maximum, rather than
fitting a full-series MinMaxScaler. The forecast head is softplus in scaled-load
units to prevent negative forecast loads by construction. These are documented
pilot departures, not a historical training reproduction.

Exactly two eight-epoch candidates (load only, and known decision-calendar
features) are fitted to labels 96..2015. Profiling-model sweeps are unrelated to
this temporal traffic training. Validation RMSE selects the forecast candidate;
there is no epoch search or final-test evaluation. The selected architecture is
then independently fitted at prefix ends 192,608,1024,1440,1856 to forecast only
later training blocks. Each fit uses only its earlier prefix, including its
scaler. The initial 96 training decisions, indices 96..191, use explicitly
labelled persistence: no earlier sequence-model training set exists. This is
5% of scored training steps, not an undisclosed replacement of the selected LSTM.
All 504 validation decisions use the selected full-training-prefix LSTM.

This is a blocked expanding-window forecast, not a model refitted at every slot.
Later within-block completed loads enter the input history while weights/scaler
stay frozen at the block's earlier cutoff. Workload design scaling remains the
approved training-peak-to-400-Mbps transformation; it is not a fitted forecasting
parameter or a claim of a live deployment calibration procedure.

## Environment and normalizer

The 109-element hybrid vector contains, in exact `feature_names()` order:
completed load/400; previous score, attributed power and SEC; a previous-outcome
mask; three executed-configuration indicators; cooldown/4; four decision-calendar
sin/cos values; 95 completed loads/400; one causal target forecast/400. Reset uses
DPDK, an eligible cooldown, and missing previous-outcome mask. It does not invent
prior simulated telemetry. The actual scored interval load never enters that
interval's observation.

`ControlEnv` delegates actuation and accounting to the preserved corrected
foundation. Each episode traverses every authorized training or validation
index, without dropping a final slot. No final-test environment can be created.
Predictions cached for authorized workload loads are checked exactly against
all corresponding real frozen-twin outcomes before training. Rule decisions
query that same twin at the causal forecast load, with the domain checks intact.

VecNormalize learns running observation moments only from training. Rewards
remain unnormalized. Evaluation loads the policy and normalizer saved together,
sets normalizer `training=False`, and never updates moments. Observation clipping
at 10 is an explicit numerical setting of VecNormalize, not physical output
repair. Every validation checkpoint is evaluated after the previous rollout's
optimizer update; the final post-update endpoint is included. Selection maximizes
validation total reward, with earlier checkpoint as deterministic tie-breaker.
All evaluated checkpoints and their paired normalizers are preserved.

Four parallel environments repeat the same deterministic training trajectory;
they are rollout collectors, not four independent workloads. Independent units
in the pilot are the two explicitly seeded training runs per algorithm. PPO and
RecurrentPPO share the hybrid observation and interaction budgets; the pilot
therefore does not isolate the historical FF-hybrid versus recurrent-history
pairing or perform a complete observation ablation.

## Baselines and sensitivity

All static and learned controllers use the same 504-slot validation horizon,
DPDK initial configuration, actions, reward and cooldown. The rule uses forecast
QoS >=.90 as a *point-model* feasibility preference and minimizes power among
those actions; if none is feasible it chooses maximum predicted QoS then lower
power. A prespecified 2% relative-power deadband retains the current feasible
configuration for small savings. All commands pass through the same actuator;
no rule action bypasses cooldown. This is a transparent new three-action curve
baseline, not a claim to reconstruct the historical two-action utility baseline.
Static controllers receive the common observation interface but ignore signals.

Sensitivity re-executes frozen selected policies/normalizers on validation with
retained alternative USR neural power and/or training-load-mean QoS. Earlier
telemetry changes accordingly; requested actions can therefore change. The rule
also queries alternative curves. Each scenario has its own identity. There is no
retraining, checkpoint reselection, forecast change, or controller-based surrogate
selection. This tests conditional sensitivity, not uncertainty calibration or
actual physical QoS risk.

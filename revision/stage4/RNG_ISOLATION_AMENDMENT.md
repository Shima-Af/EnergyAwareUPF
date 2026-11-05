# Stage 4 execution correction: isolate validation RNG

The completed initial campaign remains unchanged under `models/stage4` and
`reports/stage4`. A direct diagnostic proved that SB3 checkpoint loading changes
Python, NumPy and CPU Torch RNG state (`reports/stage4/evaluation_rng_probe.json`).
It calls `_setup_model`, resets the saved seed, and initializes layers before
restoring weights. Loading a separate validation model during training therefore
changes later training exploration/shuffle draws. This is an implementation
side effect, not a forecasting or reward choice.

The corrected campaign uses the same ten seed/algorithm jobs, 300000 interactions,
forecasts, hyperparameters, checkpoints and model-selection criterion. A context
manager restores all three RNG states on normal return and exceptions around
**the entire checkpoint capture/evaluation**. No forecasts are refitted. The
wrapper reuses the frozen trainer with separate model/report output roots; no
original Stage3 or initial Stage4 source is edited. An execution amendment and
all wrapper/source hashes are saved before the correction runs start.

Primary results belong to `models/stage4_rng_isolated` and
`reports/stage4_rng_isolated`. The first 24576-interaction checkpoint should match
the initial campaign exactly: it precedes the first RNG-perturbing validation
load. Later trajectories are allowed to diverge. Initial and corrected runs with
the same seed are not independent replicates and must not be pooled as ten seeds.

The original snapshot was retained after removing its own empty-file self-entry;
that bookkeeping correction does not change any prior artifact. The complete
initial campaign has a separate preservation hash snapshot.

Corrected command sequence:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest revision/tests -q -p no:cacheprovider > revision/reports/stage4_isolated_preflight_tests.log
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage4.isolated_launch
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage4.isolated_evidence evaluate
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m revision.stage4.isolated_evidence plot
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest revision/tests -q -p no:cacheprovider > revision/reports/stage4_final_tests.log
```

The corrected preflight includes explicit real-model RNG-isolation regression
tests. Both full campaigns and the reason for correction must appear in the
single Stage4 report. No rerun is selected because it produces a favorable PPO
result, and no additional hyperparameter or seed search is authorized.

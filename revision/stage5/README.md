# Stage 5 reproduction

This namespace adds baseline request logic and validation replay only. No `src/`, historical model, prior experiment, reward, forecast or selected policy is edited. Runtime initialization precedes SB3 imports. Inputs require the preserved revision artifacts and `.venv` versions pinned in the frozen twin/Stage 4 manifests.

In a **fresh workspace with the prior-stage artifacts but without generated Stage 5 outputs**, execute from the repository root:

```bash
.venv/bin/python -m revision.stage5.freeze
.venv/bin/python -m pytest revision/tests/test_stage5.py -q -p no:cacheprovider
.venv/bin/python -m revision.stage5.evaluate
.venv/bin/python -m revision.stage5.plots
```

`freeze` requires the preserved `reports/stage5/sources` primary-paper/main extracts; the historical source inventory and uploaded-source reconciliation document their origins/hashes. All scientific writers refuse overwrites. Repeating an existing command is not a request to replace an artifact. Exact scientific replay requires the pinned twin, Stage 4 forecasts, selected checkpoint bundles and corrected inference runtime. The final freeze manifest is the package-level identity; `baseline_frozen.json` is the earlier pre-evaluation definition freeze.

`baselines.py`: SLICES three-configuration extension and canonical historical binary adapter. `evaluate.py`: common causal validation runner and exact Stage 4 policy replay; never `.learn()` or checkpoint reselection. B is imported directly from Stage 3. `plots.py`: PNG/PDF timelines, comparison and threshold plots plus machine-readable hysteresis regions. `THRESHOLD_HYSTERESIS_AUDIT.md`: source distinctions, crossover construction and the unexecuted stronger-rule proposal.

Unit tests include synthetic scores/loads; no scientific fit occurs. The verification command for this stage also runs the existing revision suite while deselecting the three cases that actually train synthetic neural/RL models, honoring the no-training boundary:

```bash
.venv/bin/python -m pytest revision/tests -q -p no:cacheprovider \
  -k 'not exact_budget_trains_short_last_rollout_and_retains_normalizer and not small_same_seed_training_is_reproducible'
```

The ledger/hash/replay checks do not reopen the traffic final test. `inputs()` materializes authorized train/validation horizons; `run_controller` rejects other reporting roles, and the twin cache rejects inputs outside its train/validation census. The existing data splitter can read the complete original CSV for partition construction; no final-test targets, forecasts or controller outcomes are scored or used to construct/select these rules. No external service upload, Git commit or push is part of reproduction.

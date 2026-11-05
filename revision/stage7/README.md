# Stage7 reproduction

This namespace implements the frozen2×4×5 factorial: PPO/RecurrentPPO × instantaneous/history/forecast/hybrid × seeds42/123/456/789/1024,300000 interactions each. It preserves all prior code, data, forecast services, twin models, policies and reports. See `protocol.json`, `OBSERVATION_SPEC.md` and `reports/stage7/protocol_frozen.json` for scientific definitions. `horizons()` reads only authorized first2520 source rows and validates their Stage6 fingerprints. No forecasting model is refitted.

In a fresh workspace retaining all earlier artifacts but without Stage7 generated outputs:

```bash
.venv/bin/python -m revision.stage7.prepare
.venv/bin/python -m pytest revision/tests/test_stage7.py revision/tests/test_data.py revision/tests/test_contract.py revision/tests/test_selection.py revision/tests/test_stage4_rng_isolation.py -q -p no:cacheprovider
.venv/bin/python -m revision.stage7.freeze
.venv/bin/python -m revision.stage7.launch
.venv/bin/python -m revision.stage7.evaluate
.venv/bin/python -m revision.stage7.analyze
```

Scientific writers refuse overwrites. The launcher has12 independent CPU workers and no outcome-dependent scheduling/retry. Each worker fixes Python/NumPy/Torch seeds, one Torch thread, deterministic operations and CPU. The original exact-budget algorithm classes and optimizer settings are reused. A separate Stage7 environment/vector adapter implements W96; no old source is monkeypatched. Checkpoint capture is exception-safe RNG-isolated and records complete paired normalizers after optimizer updates. Fourteen checkpoints/run include32768 and exact300000. All starts, failures and completions remain in JSONL/log files.

After all40 Netflix-only selections are frozen, evaluation scores all seven baseline versions/static choices and all40 selected/end policies on both validation workloads. Baseline families are original SLICES26 plus recalibrated-v2, immutable Stage4 strong rule, and historical binary rule. Exact Stage6 calibration coefficients are used internally and displayed rounded14.268831/8.353154. No threshold/rule objective is tuned. No learned checkpoint is selected using YouTube outcomes.

Selected Netflix worker traces are retained. Sequential post-training replays verify their metrics exactly and create separate `selected_replay` traces for comparable inference timings. Inference timings include vector construction, frozen normalization, policy prediction and recurrent-state audit; external forecast service execution and outcome-model simulation are excluded. The Stage6 provider-cost measurements remain a separate component. Whole-worker peak RSS includes imported frameworks; policy parameter bytes are reported separately and do not represent total activation/optimizer memory.

Primary inference uses selected Netflix energy and five seed-block contrasts (four pair differences plus forecast×recurrence interaction), exact two-sided sign-flip p-values and Holm adjustment. Conditional Student-t intervals are descriptive with n5. No equivalence margin is invented. Endpoints, secondary schemas, SEC/reward and YouTube comparisons are reported descriptively. Source/seed/horizon limitations accompany all conclusions.

The recorded original preflight included two existing20-interaction synthetic budget-boundary fixtures plus zero-update checkpoint/recurrent-state fixtures, separate from the40 scientific fits/12000000 scientific interactions. The original broad Stage4 test file also invoked a legacy helper that parsed the full CSV, including unscored final-test rows. This breached the literal no-access boundary; no final performance was evaluated or used in fitting/selection. See `DATA_ACCESS_ADDENDUM.json`. The safe command above excludes that broad legacy file; use only its explicitly named synthetic budget test if needed. Remaining checks use prefix-only readers, including a regression test forbidding the legacy full-file reader. A failed first preflight used an excessively strict decimal-float equality in one synthetic assertion; its log is preserved and only that assertion was corrected before any scientific training.

All560 checkpoint pairs, optimizer/episode/Monitor logs, selected and endpoint traces, providers/cache identities and post-training tests are sealed in the final Stage7 manifest. The root consolidated report is `revision/FINAL_OBSERVATION_ABLATION.md`. No manuscript edit, final-test evaluation, commit or push is part of the command sequence. Stop after the report; no later reviewer campaign is automatically authorized.

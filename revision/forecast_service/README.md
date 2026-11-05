# Workload-specific causal forecasting service v1

The controller-facing output stays `revision.evaluation.contract.Forecast`; a provider accepts 96 **completed** quarter-hour loads and a known decision timestamp. It issues the load for the interval starting at that decision epoch. Models consume no realized target-interval throughput. Calendar features are known at issuance.

```python
from revision.forecast_service.core import load_service
provider = load_service('netflix')  # or 'youtube'
forecast = provider.predict(context)
# context.decision_at, context.completed_interval_at,
# context.forecast_input_mbps (96 completed loads)
```

`provider.fit` exposes parameter provenance; selected service and model manifests pin identity and selection cutoff. Both simple and learned providers implement this contract. The existing forecast-ledger adapter can read the new CSV schema without changing policy observation dimensions. No existing controller imports/defaults are redirected here.

Workload magnitudes retain the original corrected offline study's **common Netflix-training multiplier**. The new forecast normalizers are separately fitted to each permitted prefix. This distinguishes the frozen experiment's unit/load mapping from a forecast preprocessing scaler; do not refit either on the final test. YouTube forecast learning is newly authorized in this task, while the earlier policy transfer protocol is preserved unchanged. Any future transfer study must disclose target-workload analytics training.

`raw_prefix` stops after 2,520 authorized rows; no final-test target is parsed. Each source record's timestamp is preserved; no interpolation, sorting, backward filling, archived forecasts or scaled-column reuse occurs. Fingerprints cover authorized data prefixes rather than reading final-test values for a new source hash.

`protocol.json` fixes the five families, seed, inputs, budgets, three development blocks, calibration block, selection rule, cold start and proposed margin **before fits**. Neural epochs are chosen using the earlier prefix's last 96 targets, then a fresh initialization with the same seed learns all available prefix targets for that epoch count. Boosting has fixed iteration/complexity settings and no internal random early-stop split. Poisson's exponential output and neural softplus heads are nonnegative by construction. Invalid outputs raise errors, never clipping or substitution.

Causal training predictions choose a provider using only **completed earlier** development blocks. Persistence supplies the minimum-history cold start and the first comparative block. The eventual family is not backfilled into earlier training observations. At 1,680 the family is frozen using development only; 1,680–2,016 is a separate calibration block. Full-prefix fits then use training only. Validation is explicitly secondary and does not select families, epochs, margins or configurations.

Prescribed command sequence in a fresh output workspace retaining prior-stage artifacts (writers refuse overwrites):

```bash
.venv/bin/python -m pytest revision/tests/test_forecast_service.py -q -p no:cacheprovider
.venv/bin/python -m revision.forecast_service.freeze
.venv/bin/python -m revision.forecast_service.run --workload netflix
.venv/bin/python -m revision.forecast_service.run --workload youtube
.venv/bin/python -m revision.forecast_service.audit_existing
.venv/bin/python -m revision.forecast_service.analyze
```

Netflix/YouTube worker processes may run concurrently; each uses one deterministic CPU thread and independent output directories. Their fixed experiment is 30 learned artifacts: 20 neural models, each with an inner-stopping stage and a full-prefix refit, plus ten boosted models. All scientific starts/completions/failures remain in per-workload JSONL ledgers. No unsuccessful candidate is silently retried or discarded.

New service ledgers preserve index, issued/target/observed timestamps, fit/selection cutoffs, source load, issued prediction, model identity, workload and cold-start status. Final full-prefix `in_sample` scores are explicitly descriptive; they cannot pass the causal `batch` guard and use a separately named diagnostic method. Runtime latency measures 40 warm single-forecast calls including input validation/features, after three warmups; these are local CPU measurements, not an NWDAF deployment benchmark. Ordered batch reloads are checked exactly; bitwise equality across batch sizes or hardware is not promised.

The margin proposal is a **new specification only**: fixed k=1 times selected-family calibration OOF MAE per workload. It explicitly departs from SLICES in-sample MAE to reduce optimism. The literal in-sample MAE alternative and approximate published 26/24.70 ratio are also reported. MAE is not a quantile/coverage guarantee; final refitting can shift residuals. All Stage 5 rule versions, 26-Mbps parameter and prior results remain unchanged.

No PPO/RecurrentPPO/DQN training, baseline optimization/replay, twin/reward change, traffic final-test evaluation, manuscript edit, commit or push is part of this task. A later controller campaign must be separately authorized and compare all methods using matched newly frozen forecast streams and declared baseline versions.

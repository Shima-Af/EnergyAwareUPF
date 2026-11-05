# Stage7 causal observation specification v1

The submitted Sections4.3/6/7.6 define four schema families. The historical `src/environment.py` does not consistently augment common telemetry: its forecast-only base replaces load with forecast, and its history/hybrid bases replace the scalar load with history; optional additional fields vary. The submitted text uses interval-t indicators and t+1 forecasts. Under the already approved start-of-interval convention, completed slot-t outcomes are unavailable when choosing the action for t. Stage7 retains the manuscript's four **information families** with causal timing, rather than copying that indexing.

At decision t: common load is load[t−1], previous QoS/power/SEC are the outcome of the executed action for t−1, and the one-step forecast targets interval t using information through t−1. At reset, previous outcome is missing and mask0; no unexecuted telemetry is invented. Configuration is eligible DPDK, cooldown counter4. Actual load[t] enters only after action selection, guard and execution. All schemas have identical common telemetry, four known decision-calendar coordinates, three available actions and common horizons.

| Order, zero-based | Feature | Value before VecNormalize |
|---|---|---|
|0|`completed_load`|load[t−1]/400|
|1|`previous_qos`|Q[t−1], or0 at reset|
|2|`previous_power`|Attributed W[t−1], or0|
|3|`previous_sec`|P[t−1]/max(load[t−1],1e−6), or0|
|4|`previous_valid`|1 if previous outcome exists, otherwise0|
|5|`configuration_dpdk`|Executed-configuration indicator|
|6|`configuration_usr1`|Executed-configuration indicator|
|7|`configuration_usr2`|Executed-configuration indicator|
|8|`cooldown`|Pre-actuation elapsed_slots/4|
|9|`time_sin`|sin(2π decision-day fraction)|
|10|`time_cos`|cos(2π decision-day fraction)|
|11|`weekday_sin`|sin(2π weekday/7)|
|12|`weekday_cos`|cos(2π weekday/7)|

History adds, in ascending time order, **96** fields `completed_history_0` through `completed_history_95`: load[t−96]/400 through load[t−1]/400. The last value intentionally duplicates the common completed-load scalar, because every schema retains that common scalar. Forecast adds one final field `causal_target_forecast` = forecast[t]/400. Dimensions: instantaneous13; history109; forecast14; hybrid110. Names and order are machine-readable in `prepared.json` and every paired checkpoint manifest.

The manuscript specifies a96-sample *forecasting* window and a symbolic policy W. Stage7 explicitly adopts the user's W=96 decision. Stage4 actually used95 policy-history values and109 hybrid inputs; Stage7 uses96 and110. Stage4 therefore remains a historical reference, not a controlled forecast-quality experiment against Stage7. Within Stage7, history/hybrid differences isolate inclusion of the frozen external forecast under matched common information; input dimensionality and parameter counts necessarily differ slightly and are reported.

Workload construction, normalization, reward, cooldown and startup assumptions stay fixed. Each policy's observation normalizer is learned only on Netflix training, saved with each checkpoint, and frozen for both validation workloads. Reward normalization is off. The legacy clipping of normalized *observations* at10 is retained; physical outputs/forecasts are not clipped. Policy normalization never adapts to YouTube.

The Netflix training ledger has912 persistence rows (576 minimum-history cold start and further persistence while comparative evidence is unavailable) and1008 boosting rows; these provider changes are preserved without backfilling. Both validation ledgers use their own full-prefix Poisson service. Transfer means a Netflix-trained policy using YouTube-trained external analytics; no YouTube controller training or checkpoint selection occurs.

The new isolated runner permits the specifically authorized YouTube validation role only after all40 Netflix selections are frozen. This is an explicit Stage7 extension of the earlier generic contract's final-report-only transfer gate, not a change to that older code. Outcome/event accounting still calls the exact same shared `observe`, `actuate`, `account`, `Step` and `EvaluationResult` interfaces. Netflix equivalence with the generic runner is tested. No final-test horizon is accepted.

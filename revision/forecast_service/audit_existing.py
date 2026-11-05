"""Read-only original-prefix diagnosis; no fit, controller call or test-period load."""
import json
import numpy as np
import pandas as pd
from .core import ROOT,OUT,data,dump,sha256,forecast_metrics,_runtime

def main():
    times,loads,multiplier=data('netflix')
    old=pd.read_csv(ROOT/'revision/reports/stage4_rng_isolated/forecasts.csv')
    tr=old[old['index']<2016];prefixes=[]
    for path in sorted((ROOT/'revision/models/stage4/forecast').glob('prefix_*/manifest.json'),key=lambda p:int(p.parent.name.split('_')[1])):
        m=json.loads(path.read_text());cut=m['selection_stop_exclusive'];gradient=m['parameter_targets_half_open'];g=tr[tr.selection_stop_exclusive==cut]
        if len(g)==0:continue
        ix=g['index'].to_numpy(int);scale=m['scale_training_max_mbps']
        prefixes.append({'origin':cut,'gradient_targets':gradient[1]-gradient[0],
          'gradient_last_index':gradient[1]-1,'stopping_targets':m['early_stopping_targets_half_open'],
          'most_recent_observed_rows_not_used_for_gradient':cut-gradient[1],'epochs_run':m['epochs_run'],'selected_epoch':m['best_epoch'],
          'scale_mbps':scale,'evaluation_rows':len(g),'actual_mean_mbps':float(g.actual_mbps.mean()),
          'prediction_mean_mbps':float(g.forecast_mbps.mean()),'actual_above_fit_max_rows':int(sum(loads[ix]>scale)),
          **forecast_metrics(g.actual_mbps,g.forecast_mbps)})
    _,tf,_,_=_runtime();directory=ROOT/'revision/models/stage4/forecast/lstm_calendar';m=json.loads((directory/'manifest.json').read_text())
    model=tf.keras.models.load_model(directory/'model.keras',compile=False)
    from revision.stage3.forecast import predictions
    p=predictions(model,m['scale_training_max_mbps'],loads,times,list(range(2016,2520)),True)
    reference=old[old['index']>=2016].forecast_mbps.to_numpy(float)
    target_match=bool(np.array_equal(loads[old['index'].to_numpy(int)],old.actual_mbps.to_numpy(float)))
    # CSV decimal round trips may differ by last-bit float parsing. Retain magnitudes.
    target_max=float(np.max(abs(loads[old['index'].to_numpy(int)]-old.actual_mbps.to_numpy(float))))
    result={'original_validation_model_parameters':model.count_params(),'prefixes':prefixes,
      'source_workload_multiplier':multiplier,'stage4_actual_loads_exact_csv_roundtrip':target_match,
      'stage4_actual_load_max_csv_difference_mbps':target_max,'original_ordered_validation_reload_max_difference_mbps':float(np.max(abs(p-reference))),
      'original_ordered_validation_reload_metrics':forecast_metrics(loads[2016:2520],p),
      'all_training_fit_and_stopping_cutoffs_causal':bool(((tr.fit_stop_exclusive<=tr['index'])&(tr.selection_stop_exclusive<=tr['index'])).all()),
      'all_timestamps_one_interval_aligned':bool(((pd.to_datetime(old.target_timestamp)-pd.to_datetime(old.observed_through))==pd.Timedelta(minutes=15)).all()),
      'old_family_selection':'Stage4 chose calendar architecture and validation checkpoint using external validation; its training-prefix parameter fits were causal, family selection was not training-only.',
      'conclusions':{'confirmed_implementation_defect':'No target-shift, duplicated log transform or wrong denormalization is demonstrated in corrected Stage4 inference. Historical preprocessing had separate known leakage.',
       'confirmed_data_limit':'First neural prefix has only76 gradient targets; most recent20 stopping rows are excluded from gradient fitting. Later prefixes also omit a substantial recent stopping suffix.',
       'supported_mechanisms':'Limited history, high capacity, early stopping on tiny blocks, no full-prefix refit and training-vs-validation distribution differences are supported contributors. They are not experimentally isolated causal effect sizes.',
       'new_changes':'At least7 days history, past-only96-slot stopping suffix, then complete-prefix refit; training-only family selection. No claim one change alone repairs forecasting.'},
      'reference_hashes':{str(path.relative_to(ROOT)):sha256(path) for path in [directory/'model.keras',directory/'manifest.json',ROOT/'revision/reports/stage4_rng_isolated/forecasts.csv']}}
    dump(OUT/'stage4_forecasting_audit.json',result)
    print(json.dumps({'parameters':model.count_params(),'prefixes':prefixes,'max_reload_difference':result['original_ordered_validation_reload_max_difference_mbps']},indent=2),flush=True)

if __name__=='__main__':main()

"""Post-selection reliability diagnostics. No fitting, selection changes or controllers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import ROOT,OUT,MODELS,CFG,data,forecast_metrics,dump,write_csv,_runtime,sha256


def describe(g):
    e=np.asarray(g.forecast_mbps)-np.asarray(g.actual_mbps)
    return {**forecast_metrics(g.actual_mbps,g.forecast_mbps),'median_error_mbps':float(np.median(e)),
       'p05_error_mbps':float(np.quantile(e,.05)),'p95_error_mbps':float(np.quantile(e,.95)),
       'p95_absolute_error_mbps':float(np.quantile(abs(e),.95)),'max_absolute_error_mbps':float(max(abs(e)))}


def main():
    figdir=OUT/'figures';figdir.mkdir();comparisons={};strata=[];thresholds=[];costs={};hist_rows=[]
    freeze=json.loads((ROOT/'revision/reports/stage5/baseline_frozen.json').read_text())
    boundaries=[r['right'] for r in freeze['slices_regions'][:-1]]+[r['right'] for r in freeze['comcom_utility_regions'][:-1]]
    audit=json.loads((OUT/'stage4_forecasting_audit.json').read_text())
    for name in CFG['workloads']:
        out=OUT/name;times,loads,_=data(name);sel=json.loads((out/'selection_frozen.json').read_text());chosen=sel['selected_family']
        back=pd.read_csv(out/'candidate_backtests.csv');val=pd.read_csv(out/'validation_candidates.csv');train=pd.read_csv(out/'training_forecasts.csv')
        subsets={'development':back[back.kind=='development'],'calibration':back[back.kind=='calibration'],'secondary_validation':val}
        scores={role:{family:describe(g) for family,g in d.groupby('family',sort=False)} for role,d in subsets.items()}
        scores['prequential_training_service']=describe(train);scores['selected_family']=chosen
        comparisons[name]=scores
        for role,d in subsets.items():
            for family,g in d.groupby('family',sort=False):
                g=g.copy();g['increment']=g.actual_mbps.to_numpy()-loads[g['index'].to_numpy(int)-1]
                groups=[('all',g)]
                for lo,hi in [(0,50),(50,100),(100,200),(200,400),(400,float('inf'))]:
                    groups.append((f'load_{lo}_{hi}',g[(g.actual_mbps>=lo)&(g.actual_mbps<hi)]))
                groups += [('rising',g[g.increment>0]),('falling',g[g.increment<0]),('unchanged',g[g.increment==0]),('rise_over30',g[g.increment>30]),('fall_over30',g[g.increment< -30])]
                for day,h in g.groupby(g.target_timestamp.str[:10]):groups.append(('day_'+day,h))
                for label,h in groups:
                    if len(h):strata.append({'workload':name,'role':role,'family':family,'stratum':label,**describe(h)})
                for boundary in boundaries:
                    actual=g.actual_mbps.to_numpy();pred=g.forecast_mbps.to_numpy();near=g[abs(g.actual_mbps-boundary)<=10]
                    thresholds.append({'workload':name,'role':role,'family':family,'boundary_mbps':boundary,'rows':len(g),
                      'crossing_disagreement_rows':int(sum((actual>=boundary)!=(pred>=boundary))),
                      'predicted_above_actual_below_rows':int(sum((pred>=boundary)&(actual<boundary))),
                      'predicted_below_actual_above_rows':int(sum((pred<boundary)&(actual>=boundary))),
                      'near_band_width_each_side_mbps':10.,'near_rows':len(near),
                      'near_mae_mbps':float(abs(near.forecast_mbps-near.actual_mbps).mean()) if len(near) else None,
                      'near_rmse_mbps':float(np.sqrt(np.mean((near.forecast_mbps-near.actual_mbps)**2))) if len(near) else None})
        models=list((MODELS/name).rglob('manifest.json'));learned=[];epochs=0
        for p in models:
            m=json.loads(p.read_text())
            if m.get('model_file'):
                learned.append(m)
                if (p.parent/'history.csv').exists():epochs+=len(pd.read_csv(p.parent/'history.csv'))
        costs[name]={'learned_artifacts':len(learned),'neural_optimization_stages':2*sum(m['family'] in ('gru','lstm') for m in learned),
          'boosted_fits':sum(m['family']=='boosted_poisson' for m in learned),'neural_epochs_total':epochs,
          'total_fit_seconds':float(sum(m['training_seconds'] for m in learned)),
          'maximum_neural_epoch_budget':1200,'minimum_history_slots':672}
        fig,axes=plt.subplots(2,1,figsize=(13,7))
        d=val[val.family==chosen];x=pd.to_datetime(d.target_timestamp)
        axes[0].plot(x,d.actual_mbps,label='Realized',color='black',linewidth=.8)
        for family in [chosen,'persistence','lstm']:
            h=val[val.family==family]
            if family==chosen or family not in [chosen]:axes[0].plot(pd.to_datetime(h.target_timestamp),h.forecast_mbps,label=family,linewidth=.7,alpha=.8)
        axes[0].set(ylabel='Workload Mbps',title=name+' — secondary validation, family already frozen');axes[0].legend()
        axes[1].plot(pd.to_datetime(train.target_timestamp),train.forecast_mbps-train.actual_mbps,linewidth=.7)
        axes[1].axhline(0,color='black',linewidth=.5);axes[1].set(ylabel='Prequential forecast − actual Mbps',xlabel='Source clock')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(figdir/f'{name}_forecast_reliability.{ext}',dpi=160)
        plt.close(fig)
        # Existing saved Stage4 model reference: never used in new family selection.
        directory=ROOT/'revision/models/stage4/forecast/lstm_calendar';m=json.loads((directory/'manifest.json').read_text())
        _,tf,_,_=_runtime();model=tf.keras.models.load_model(directory/'model.keras',compile=False)
        from revision.stage3.forecast import predictions
        p=predictions(model,m['scale_training_max_mbps'],loads,times,list(range(2016,2520)),True)
        hist_rows.append({'workload':name,'reference':'old Stage4 Netflix-trained, validation-selected calendar LSTM','role':'secondary_validation_descriptive_only',
          'workload_specific_training':name=='netflix',**forecast_metrics(loads[2016:2520],p)})
        if name=='netflix':
            old=pd.read_csv(ROOT/'revision/reports/stage4_rng_isolated/forecasts.csv');g=old[old['index']<2016]
            scores['old_stage4_training_ledger_descriptive']=describe(g)
        cal=subsets['calibration'];cal=cal[cal.family==chosen];margin=json.loads((out/'hysteresis_recalibration_proposal.json').read_text())
        margin['empirical_calibration_fraction_absolute_error_le_margin']=float(np.mean(abs(cal.forecast_mbps-cal.actual_mbps)<=margin['proposed_margin_mbps']))
        # Save supplementary evidence without overwriting the frozen margin proposal.
        dump(out/'calibration_reliability.json',{'selected_family':chosen,'signed_residuals':describe(cal),
           'empirical_fraction_covered_by_proposed_mae_margin':margin['empirical_calibration_fraction_absolute_error_le_margin'],
           'future_coverage_guarantee':False,'final_refit_distribution_may_differ':True})
    write_csv(OUT/'reliability_by_stratum.csv',strata);write_csv(OUT/'decision_boundary_errors.csv',thresholds);write_csv(OUT/'historical_lstm_reference.csv',hist_rows)
    dump(OUT/'comparison_metrics.json',comparisons);dump(OUT/'computational_cost.json',costs)
    print(json.dumps({'comparisons':comparisons,'costs':costs,'historical_reference':hist_rows},indent=2),flush=True)

if __name__=='__main__':main()

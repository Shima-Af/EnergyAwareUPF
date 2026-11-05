"""Fixed training-only development, calibration, OOF service and secondary validation."""
import argparse,json,time,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .core import CFG,BASE,OUT,MODELS,data,data_hash,fit,simple,Provider,select,forecast_metrics,dump,sha256,ROOT,write_csv


def row(i,p,family,provider,times,loads,selection_stop,kind):
    return {'index':int(i),'forecast_timestamp':times[i].isoformat(),'target_timestamp':times[i].isoformat(),
      'observed_through':times[i-1].isoformat(),'fit_stop_exclusive':provider.meta['fit_stop_exclusive'],
      'selection_stop_exclusive':int(selection_stop),'forecast_mbps':float(p),'actual_mbps':float(loads[i]),
      'family':family,'model_identity':provider.identity,'model_manifest':str((provider.directory/'manifest.json').relative_to(ROOT)),
      'workload':provider.meta['workload'],'kind':kind}


def append(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workload',choices=CFG['workloads'],required=True);args=parser.parse_args();name=args.workload
    out=OUT/name;out.mkdir();models=MODELS/name;models.mkdir();times,loads,multiplier=data(name)
    frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    for p,h in frozen['implementation_hashes'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Code changed after protocol freeze')
    dump(out/'data_manifest.json',{'authorized_prefix_rows':2520,'training_rows':2016,'validation_rows':504,
      'authorized_prefix_sha256':data_hash(loads,times,2520),'training_prefix_sha256':data_hash(loads,times,2016),
      'workload_scale_multiplier':multiplier,'training_peak_mbps':float(max(loads[:2016])),
      'training_start':times[0].isoformat(),'training_end_inclusive':times[2015].isoformat(),
      'validation_start':times[2016].isoformat(),'validation_end_inclusive':times[2519].isoformat(),
      'source':CFG['datasets'][name],'archived_forecasts_used':False,'final_test_rows_parsed':0})
    simple_providers={f:simple(f,name,models/f) for f in ['persistence','seasonal_daily']}
    dev={f:[] for f in CFG['families']};candidate_rows=[];service_rows=[];fold_records=[]
    for i in range(96,672):service_rows.append(row(i,loads[i-1],'persistence',simple_providers['persistence'],times,loads,0,'minimum_history_cold_start'))
    selected=None
    for stop in [672,1008,1344,1680]:
        end=stop+336;role='calibration' if stop==1680 else 'development'
        # The issuing provider is chosen before this block's targets are scored.
        prior_scores={f:forecast_metrics([r['actual_mbps'] for r in records],[r['forecast_mbps'] for r in records]) for f,records in dev.items() if records}
        issuing=selected if selected else (select(prior_scores) if prior_scores else 'persistence')
        choice_stop=stop if prior_scores else 0
        providers=dict(simple_providers)
        for family in ['lstm','gru','boosted_poisson']:
            path=models/f'prefix_{stop}'/family;append(out/'fit_ledger.jsonl',{'status':'started','family':family,'fit_stop':stop,'role':role})
            try:
                providers[family]=fit(family,name,loads,times,stop,path)
                append(out/'fit_ledger.jsonl',{'status':'completed','family':family,'fit_stop':stop,'role':role,'manifest_sha256':providers[family].identity})
            except Exception as exc:
                append(out/'fit_ledger.jsonl',{'status':'failed','family':family,'fit_stop':stop,'error':repr(exc),'traceback':traceback.format_exc()});raise
        for family,provider in providers.items():
            predictions=provider.batch(loads,times,range(stop,end))
            records=[row(i,p,family,provider,times,loads,stop,role) for i,p in zip(range(stop,end),predictions)]
            candidate_rows+=records
            if role=='development':dev[family]+=records
            fold_records.append({'role':role,'origin':stop,'family':family,**forecast_metrics(loads[stop:end],predictions)})
            if family==issuing:
                service_rows.extend([{**r,'selection_stop_exclusive':choice_stop,'kind':'past_selected_provider'} for r in records])
        if stop==1344:
            scores={f:forecast_metrics([r['actual_mbps'] for r in records],[r['forecast_mbps'] for r in records]) for f,records in dev.items()}
            selected=select(scores)
            dump(out/'selection_frozen.json',{'selected_family':selected,'selection_cutoff_exclusive':1680,
              'selection_cutoff_timestamp':times[1680].isoformat(),'development_half_open':[672,1680],
              'development_metrics':scores,'calibration_results_seen':False,'external_validation_seen':False,
              'tie_break':'persistence, seasonal_daily, boosted_poisson, gru, lstm','no_search_expansion':True})
            print(name,'FAMILY FROZEN',selected,scores,flush=True)
    write_csv(out/'candidate_backtests.csv',candidate_rows);write_csv(out/'rolling_metrics.csv',fold_records)
    write_csv(out/'training_forecasts.csv',service_rows)
    providers={f:simple(f,name,models/'full'/f,1680,times) for f in ['persistence','seasonal_daily']}
    for family in ['lstm','gru','boosted_poisson']:
        append(out/'fit_ledger.jsonl',{'status':'started','family':family,'fit_stop':2016,'role':'full_training_prefix'})
        try:
            providers[family]=fit(family,name,loads,times,2016,models/'full'/family)
            append(out/'fit_ledger.jsonl',{'status':'completed','family':family,'fit_stop':2016,'role':'full_training_prefix','manifest_sha256':providers[family].identity})
        except Exception as exc:
            append(out/'fit_ledger.jsonl',{'status':'failed','family':family,'fit_stop':2016,'error':repr(exc)});raise
    validation=[];metrics={};latencies={}
    for family,provider in providers.items():
        p=provider.batch(loads,times,range(2016,2520));repeat=Provider(provider.directory).batch(loads,times,range(2016,2520))
        if not np.array_equal(p,repeat):raise ValueError('Ordered reload predictions changed')
        validation += [row(i,v,family,provider,times,loads,2016 if family in ('lstm','gru') else 1680,'secondary_validation_only') for i,v in zip(range(2016,2520),p)]
        insample=provider.diagnostic_in_sample(loads,times,range(96,2016))
        metrics[family]={'secondary_validation':forecast_metrics(loads[2016:2520],p),
           'training_in_sample_descriptive':forecast_metrics(loads[96:2016],insample),
           'training_seconds':provider.meta.get('training_seconds',0.),
           'artifact_bytes':sum(f.stat().st_size for f in provider.directory.iterdir() if f.is_file()),'exact_ordered_reload':True}
        from types import SimpleNamespace
        context=SimpleNamespace(decision_at=times[2016],completed_interval_at=times[2015],forecast_input_mbps=tuple(loads[1920:2016]))
        for _ in range(3):provider.predict(context)
        timing=[]
        for _ in range(40):
            start=time.perf_counter();provider.predict(context);timing.append((time.perf_counter()-start)*1000)
        latencies[family]={'single_prediction_p50_ms':float(np.median(timing)),'single_prediction_p95_ms':float(np.percentile(timing,95)),
                           'samples':40,'warmups':3,'includes_window_validation_and_feature_construction':True}
    write_csv(out/'validation_candidates.csv',validation)
    val_selected=[r for r in validation if r['family']==selected]
    write_csv(out/'validation_forecasts.csv',val_selected)
    write_csv(out/'forecasts.csv',service_rows+val_selected)
    chosen=providers[selected];dump(out/'service.json',{'version':CFG['id'],'workload':name,'selected_family':selected,
      'provider_manifest':str((chosen.directory/'manifest.json').relative_to(ROOT)),'provider_manifest_sha256':chosen.identity,
      'selection_manifest':str((out/'selection_frozen.json').relative_to(ROOT)),'selection_sha256':sha256(out/'selection_frozen.json'),
      'input_contract':'96 completed quarter-hour loads in common workload Mbps and known decision timestamp',
      'output_contract':'revision.evaluation.contract.Forecast; current target interval, observed-through previous completed interval',
      'serving_role':'Diagnostic only, no automatic controller or baseline promotion','final_test_evaluation':False})
    full_metrics={'candidate_metrics':metrics,'latency':latencies,'selected_family':selected,
      'prequential_training':forecast_metrics([r['actual_mbps'] for r in service_rows],[r['forecast_mbps'] for r in service_rows]),
      'minimum_history_cold_start_slots':576,'first_comparative_block_persistence_slots':336,
      'selection_frozen_before_calibration':True}
    dump(out/'metrics.json',full_metrics)
    cal=[r for r in candidate_rows if r['kind']=='calibration' and r['family']==selected]
    margin=forecast_metrics([r['actual_mbps'] for r in cal],[r['forecast_mbps'] for r in cal])['mae_mbps']
    dump(out/'hysteresis_recalibration_proposal.json',{'status':'proposal_only_no_baseline_changed','workload':name,'k':1.,
      'k_source':'Declared a priori unit error multiplier, not fitted to control performance',
      'calibration_half_open':[1680,2016],'calibration_rows':len(cal),'oof_mae_mbps':margin,'proposed_margin_mbps':margin,
      'literal_in_sample_margin_mbps':metrics[selected]['training_in_sample_descriptive']['mae_mbps'],
      'approximate_paper_ratio_26_over_24_70':26/24.70,'historical_26_automatically_reused':False,
      'limitation':'Empirical same-workload calibration under a preceding-prefix model; neither coverage guarantee nor identical to the final refitted model. Affects three-region entry interiors and needs a new baseline version before future comparison.'})
    print(name,'COMPLETE',selected,full_metrics['prequential_training'],'margin',margin,flush=True)

if __name__=='__main__':main()

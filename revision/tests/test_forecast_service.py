"""No optimization fixtures: causality, split policy and provider reproducibility."""
import json
from datetime import datetime,timedelta
from types import SimpleNamespace as S
import numpy as np
import pytest
from revision.forecast_service.core import CFG,BASE,OUT,MODELS,ROOT,raw_prefix,data,windows,tabular,simple,Provider,select,sha256


def fixture():
    times=tuple(datetime(2019,4,1)+timedelta(minutes=15*i) for i in range(300))
    return times,np.arange(300,dtype=float)+1


def test_windows_exclude_target_and_future_and_match_stage4():
    times,loads=fixture();before=windows(loads,times,[100,150],300.)
    poisoned=loads.copy();poisoned[150:]=1e10
    assert np.array_equal(before[1],windows(poisoned,times,[150],300.)[0])
    assert before.shape==(2,96,7)
    assert before[0,0,0]==pytest.approx(loads[4]/300.)
    assert before[0,-1,0]==pytest.approx(loads[99]/300.)
    from revision.stage3.forecast import arrays
    assert np.array_equal(before,arrays(loads,times,[100,150],300.,True))
    assert tabular(before).shape==(2,106)


def test_prefix_reader_prohibits_final_test_and_uses_raw_fields():
    with pytest.raises(ValueError):raw_prefix('netflix',2521)
    times,loads,scale=data('netflix');assert len(times)==len(loads)==2520
    assert max(loads[:2016])==400.
    yt,yl,ys=data('youtube');assert yt==times and ys==scale
    assert times[2016]==datetime(2019,4,22)
    assert times[2519]==datetime(2019,4,27,5,45)


def test_protocol_separates_development_calibration_and_secondary_validation():
    assert CFG['development_origins']==[672,1008,1344]
    assert CFG['development_origins'][-1]+336==CFG['calibration_half_open'][0]
    assert CFG['calibration_half_open'][1]==CFG['validation_half_open'][0]==2016
    assert CFG['validation_half_open'][1]==2520
    assert len(CFG['families'])==5
    assert CFG['hysteresis_proposal']['k']==1.
    assert select({'gru':{'rmse_mbps':1},'persistence':{'rmse_mbps':1}})=='persistence'


@pytest.mark.parametrize('family,offset',[('persistence',1),('seasonal_daily',96)])
def test_parameter_free_forecast_interface_deterministic_and_reload(tmp_path,family,offset):
    times,loads=fixture();p=simple(family,'fixture',tmp_path/family)
    context=S(decision_at=times[120],completed_interval_at=times[119],forecast_input_mbps=tuple(loads[24:120]))
    f=p.predict(context)
    assert f==Provider(p.directory).predict(context)
    assert f.load_mbps==loads[120-offset]
    assert f.target_at==f.issued_at==times[120] and f.observed_through==times[119]
    context.forecast_input_mbps=tuple(loads[25:120])
    with pytest.raises(ValueError):p.predict(context)


def test_artifact_identity_failure_before_deserialization(tmp_path):
    p=simple('persistence','fixture',tmp_path/'provider')
    path=p.directory/'manifest.json';m=json.loads(path.read_text());m['implementation_sha256']='bad';path.write_text(json.dumps(m))
    with pytest.raises(ValueError):Provider(p.directory)


def test_model_training_cutoff_checked_without_fitting(tmp_path):
    times,loads=fixture();p=simple('persistence','fixture',tmp_path/'p',selection_stop=150,times=times)
    with pytest.raises(ValueError):p.batch(loads,times,[149])
    assert p.batch(loads,times,[150])[0]==loads[149]
    c=S(decision_at=times[149],completed_interval_at=times[148],forecast_input_mbps=tuple(loads[53:149]))
    with pytest.raises(ValueError):p.predict(c)


def test_completed_forecasts_are_causal_and_selection_is_not_backfilled():
    for name in CFG['workloads']:
        path=OUT/name/'service.json'
        if not path.exists():pytest.skip('Scientific outputs not generated yet')
        import pandas as pd
        service=json.loads(path.read_text());sel=json.loads((OUT/name/'selection_frozen.json').read_text())
        assert sel['selection_cutoff_exclusive']==1680 and not sel['calibration_results_seen'] and not sel['external_validation_seen']
        tr=pd.read_csv(OUT/name/'training_forecasts.csv');assert tr['index'].tolist()==list(range(96,2016))
        assert (tr.fit_stop_exclusive<=tr['index']).all() and (tr.selection_stop_exclusive<=tr['index']).all()
        assert set(tr[tr['index']<1008].family)=={'persistence'}
        assert set(tr[tr['index']>=1680].family)=={service['selected_family']}
        assert (pd.to_datetime(tr.target_timestamp)-pd.to_datetime(tr.observed_through)==pd.Timedelta(minutes=15)).all()
        assert tr[tr['index']<672].shape[0]==576
        ledger=[json.loads(l) for l in (OUT/name/'fit_ledger.jsonl').read_text().splitlines()]
        assert len([r for r in ledger if r['status']=='started'])==15
        assert len([r for r in ledger if r['status']=='completed'])==15
        assert not [r for r in ledger if r['status']=='failed']


def test_real_selected_artifact_reload_and_future_poisoning():
    for name in CFG['workloads']:
        path=OUT/name/'service.json'
        if not path.exists():pytest.skip('Scientific outputs not generated yet')
        import pandas as pd
        s=json.loads(path.read_text());manifest=ROOT/s['provider_manifest']
        assert sha256(manifest)==s['provider_manifest_sha256']
        p=Provider(manifest.parent);times,loads,_=data(name)
        pred=p.batch(loads,times,[2016,2017])
        poisoned=loads.copy();poisoned[2017:]=1e8
        assert p.batch(loads,times,[2017])[0]==p.batch(poisoned,times,[2017])[0]
        assert np.array_equal(pred,Provider(manifest.parent).batch(loads,times,[2016,2017]))
        v=pd.read_csv(OUT/name/'validation_forecasts.csv');assert len(v)==504
        assert v['index'].tolist()==list(range(2016,2520))
        m=json.loads((OUT/name/'metrics.json').read_text())
        assert all(x['exact_ordered_reload'] for x in m['candidate_metrics'].values())


def test_training_rejects_validation_period_before_writing(tmp_path):
    from revision.forecast_service.core import fit
    times,loads=fixture()
    for family in ['lstm','gru','boosted_poisson']:
        with pytest.raises(ValueError):fit(family,'fixture',loads,times,2520,tmp_path/family)
        assert not (tmp_path/family).exists()


def test_completed_services_bind_identity_and_preserve_contract():
    from revision.forecast_service.core import load_service
    for name in CFG['workloads']:
        if not (OUT/name/'service.json').exists():pytest.skip('Scientific outputs pending')
        p=load_service(name);times,loads,_=data(name)
        context=S(decision_at=times[2016],completed_interval_at=times[2015],forecast_input_mbps=tuple(loads[1920:2016]))
        first=p.predict(context);assert first==p.predict(context)
        assert first.target_at==context.decision_at and np.isfinite(first.load_mbps) and first.load_mbps>=0
        # Existing ledger consumer accepts the same timestamp/forecast schema.
        from revision.stage3.pipeline import ForecastLedger
        ledger=ForecastLedger(OUT/name/'forecasts.csv')
        for i in [96,671,672,1008,1680,2015,2016,2519]:
            c=S(decision_at=times[i],completed_interval_at=times[i-1])
            assert ledger.at(i,c).target_at==times[i]


def test_real_model_artifact_tampering_and_zero_load_outputs(tmp_path):
    for name in CFG['workloads']:
        path=MODELS/name/'full'/'boosted_poisson'/'manifest.json'
        if not path.exists():pytest.skip('Scientific outputs pending')
        times,loads,_=data(name)
        for family in ['lstm','gru','boosted_poisson']:
            provider=Provider(MODELS/name/'full'/family)
            c=S(decision_at=times[2016],completed_interval_at=times[2015],forecast_input_mbps=(0.,)*96)
            assert np.isfinite(provider.predict(c).load_mbps) and provider.predict(c).load_mbps>=0
        m=json.loads(path.read_text());directory=tmp_path/name;directory.mkdir()
        (directory/'manifest.json').write_text(json.dumps(m));(directory/m['model_file']).write_bytes(b'corrupt')
        with pytest.raises(ValueError):Provider(directory)


def test_saved_scalers_stopping_and_parameter_rows_use_only_past():
    for name in CFG['workloads']:
        if not (OUT/name/'service.json').exists():pytest.skip('Scientific outputs pending')
        times,loads,_=data(name)
        from revision.forecast_service.core import data_hash
        for path in (MODELS/name).rglob('manifest.json'):
            m=json.loads(path.read_text())
            if not m.get('model_file'):continue
            stop=m['fit_stop_exclusive']
            assert stop in [672,1008,1344,1680,2016]
            assert m['scaler_fit_stop_exclusive']==stop and m['scale']==float(max(loads[:stop]))
            assert m['parameter_targets_half_open']==[96,stop]
            assert m['training_prefix_sha256']==data_hash(loads,times,stop)
            assert m['last_training_timestamp']==times[stop-1].isoformat()
            assert not m['uses_validation_or_final_test_for_fit']
            if m['family'] in ('lstm','gru'):
                assert m['inner_stopping_targets_half_open']==[stop-96,stop]
                import pandas as pd
                h=pd.read_csv(path.parent/'history.csv')
                assert len(h[h.phase=='past_only_refit'])==m['best_epoch']<=60
                assert len(h[h.phase=='inner_stopping'])<=60
        selection=json.loads((OUT/name/'selection_frozen.json').read_text())
        assert selection['selected_family']==select(selection['development_metrics'])
        assert selection['development_metrics']['persistence']['n']==1008


def test_parser_never_converts_final_test_targets(tmp_path,monkeypatch):
    import revision.forecast_service.core as core
    path=tmp_path/core.CFG['datasets']['netflix'];path.parent.mkdir(parents=True)
    start=datetime(2019,4,1)
    with path.open('w') as f:
        f.write('timestamp,Traffic_bps\n')
        for i in range(2520):f.write(f'{start+timedelta(minutes=15*i)},1000000\n')
        f.write('FORBIDDEN_FINAL_TEST_TIMESTAMP,FORBIDDEN_FINAL_TEST_TARGET\n')
    monkeypatch.setattr(core,'ROOT',tmp_path)
    times,loads=core.raw_prefix('netflix',2520)
    assert len(times)==2520 and np.all(loads==1.)


def test_each_training_provider_choice_uses_only_completed_earlier_blocks():
    import pandas as pd
    for name in CFG['workloads']:
        if not (OUT/name/'service.json').exists():pytest.skip('Scientific outputs pending')
        back=pd.read_csv(OUT/name/'candidate_backtests.csv');tr=pd.read_csv(OUT/name/'training_forecasts.csv')
        from revision.forecast_service.core import forecast_metrics
        for origin in [672,1008,1344,1680]:
            earlier=back[(back['index']<origin)&(back.kind=='development')]
            scores={family:forecast_metrics(g.actual_mbps,g.forecast_mbps) for family,g in earlier.groupby('family')}
            expected=select(scores) if scores else 'persistence'
            assert set(tr[(tr['index']>=origin)&(tr['index']<origin+336)].family)=={expected}

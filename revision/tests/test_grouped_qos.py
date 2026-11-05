"""Task 2H artifact and grouped-protocol tests; no extra neural training runs."""

import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest

from revision.evaluation.twin import _runtime,sha256
from revision.profiling.neural_qos import BoundedQoSModel,SimpleQoSModel,TARGET
from revision.profiling.power_model import LOAD,ProfilingPartition
from revision.profiling.train_grouped_qos import CONFIG,fit_one,score_predictions

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'revision/reports/task2h'


@pytest.fixture(scope='module',params=['DPDK','USR'])
def selected(request):
    info=json.loads((REPORT/'selection_frozen.json').read_text())['modes'][request.param]
    return BoundedQoSModel(info['selected_manifest'])


def test_exact_group_memberships_are_disjoint_complete_and_chronological():
    protocol=json.loads((REPORT/'prespecified_protocol.json').read_text())
    config=json.loads(CONFIG.read_text())
    assert protocol['protocol_sha256']==sha256(CONFIG)
    for source in config['datasets']:
        f=pd.read_csv(Path(config['source_directory'])/source['file'])
        roles=protocol['source_rows'][source['mode']]
        assert sum(len(v) for v in roles.values())==len(f)
        assert len(set(i for v in roles.values() for i in v))==len(f)
        ends=[]
        for role,idx in roles.items():
            part=f.iloc[idx]
            assert set(part['Test Set'])==set(config['split'][role])
            ends.append((part.timestamp.min(),part.timestamp.max()))
        assert ends[0][1]<ends[1][0] and ends[1][1]<ends[2][0]


def test_eight_scientific_runs_logged_with_no_retries():
    ledger=[json.loads(s) for s in (REPORT/'candidate_runs.jsonl').read_text().splitlines()]
    started=[r for r in ledger if r['status']=='started']
    terminal=[r for r in ledger if r['status'] in ('completed','failed')]
    assert len(started)==len(terminal)==8
    assert len({(r['mode'],r['architecture'],r['seed']) for r in started})==8
    assert all(r['status']=='completed' for r in terminal)


def test_selection_uses_validation_only_and_losing_candidates_have_no_test_scores():
    selection=json.loads((REPORT/'selection_frozen.json').read_text())
    assert not selection['internal_test_scored_before_selection']
    assert not selection['search_expanded'] and not selection['traffic_used_for_selection']
    metrics=json.loads((REPORT/'metrics.json').read_text())
    for mode,s in selection['modes'].items():
        winner=min(s['validation_scores'],key=lambda n:s['validation_scores'][n]['rmse'])
        assert winner==s['selected_name']
        assert sha256(Path(s['selected_manifest']))==s['selected_manifest_sha256']
        for name in s['validation_scores']:
            assert ('internal_test' in metrics['modes'][mode][name])==(name==winner)


def test_preprocessing_uses_only_training_sweeps(selected):
    config=json.loads(CONFIG.read_text())
    source=next(s for s in config['datasets'] if s['mode']==selected.metadata['mode'])
    frame=pd.read_csv(Path(config['source_directory'])/source['file'])
    x=np.log1p(frame[frame['Test Set'].isin([1,2,3])][LOAD].to_numpy())
    scaler=selected.metadata['input_scaler']
    assert scaler['mean']==x.mean() and scaler['scale']==x.std()
    assert scaler['fit_rows']==len(x) and scaler['fit_sweeps']==[1,2,3]


def test_inference_and_reload_are_exactly_repeatable_and_bounded(selected):
    x=np.geomspace(.01,5000.,2103)
    a=selected.predict(x)
    np.testing.assert_array_equal(a,selected.predict(x))
    np.testing.assert_array_equal(a,BoundedQoSModel(selected.manifest_path).predict(x))
    assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    assert selected.network.layers[-1].get_config()['activation']=='sigmoid'
    assert selected.metadata['target_transform']=='identity_score'


def test_unmeasured_or_invalid_domain_is_rejected_without_clipping(selected):
    for load in [0.,-.1,5001.,np.nan,np.inf]:
        with pytest.raises(ValueError,match='domain'):selected.predict([load])


def test_artifact_tampering_rejected_before_loading(selected,tmp_path):
    shutil.copy(selected.manifest_path,tmp_path/'manifest.json')
    shutil.copy(selected.model_path,tmp_path/'model.keras')
    with (tmp_path/'model.keras').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='hash mismatch'):BoundedQoSModel(tmp_path/'manifest.json')


def test_already_loaded_artifact_identity_detects_manifest_changes(selected,tmp_path):
    shutil.copy(selected.manifest_path,tmp_path/'manifest.json')
    shutil.copy(selected.model_path,tmp_path/'model.keras')
    model=BoundedQoSModel(tmp_path/'manifest.json')
    with (tmp_path/'manifest.json').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='changed'):model.verify_artifacts()


def test_baselines_cannot_fit_validation_or_test():
    for role in ['validation','internal_test']:
        part=ProfilingPartition(role,(4,),pd.DataFrame({'Test Set':[4,4],LOAD:[1.,2.],TARGET:[.7,.9]}))
        with pytest.raises(ValueError,match='TRAIN'):SimpleQoSModel.fit_training(part,'training_mean_constant',(1.,2.))


def test_simple_references_are_training_only_and_bounded():
    f=pd.DataFrame({'Test Set':[1,1,2,2],LOAD:[1.,10.,1.,10.],TARGET:[.2,.8,.4,1.]})
    part=ProfilingPartition('train',(1,2),f)
    a=SimpleQoSModel.fit_training(part,'training_mean_constant',(1.,10.))
    b=SimpleQoSModel.fit_training(part,'training_load_mean_pwl',(1.,10.))
    np.testing.assert_allclose(a.predict([1.,5.,10.]),[.6,.6,.6])
    np.testing.assert_allclose(b.predict([1.,5.5,10.]),[.3,.6,.9])
    np.testing.assert_array_equal(b.predict([1.,10.]),SimpleQoSModel(json.loads(json.dumps(b.metadata))).predict([1.,10.]))
    with pytest.raises(ValueError):b.predict([0.])


def test_learning_and_checkpoint_interfaces_reject_internal_test_without_training(tmp_path):
    _,tf,_,_=_runtime();config=json.loads(CONFIG.read_text())
    f=pd.DataFrame({'Test Set':[5],LOAD:[1.],TARGET:[.9]})
    test=ProfilingPartition('internal_test',(5,),f)
    val=ProfilingPartition('validation',(5,),f)
    for train,validation in [(test,val),(val,test)]:
        with pytest.raises(ValueError,match='TRAIN'):
            fit_one(train,validation,config['candidates_per_mode'][0],42,config,tmp_path/'forbidden',tf,config['datasets'][0])


def test_reject_all_baseline_does_not_report_zero_false_safe_risk():
    f=pd.DataFrame({TARGET:[.85,.95]})
    result=score_predictions(f,np.asarray([.7,.7]))
    assert result['false_safe']==0 and result['predicted_safe']==0
    assert result['false_safe_fraction_predicted_safe'] is None
    assert result['predicted_safe_coverage']==0


def test_variance_identity_retains_small_batch_position_effects():
    from revision.diagnostics.summarize_grouped_qos import decompose_prediction_variance
    f=pd.DataFrame({LOAD:[1.,1.,2.,2.],TARGET:[.7,.9,.8,1.],
                    'predicted_qos':[.8,.8+6e-8,.9,.9]})
    d=decompose_prediction_variance(f)
    assert d['max_same_load_prediction_span']>0
    assert d['actual_model_sse']==pytest.approx(d['within_load_sse']+d['load_mean_misfit_sse']+d['numerical_prediction_cross_term'],abs=1e-14)

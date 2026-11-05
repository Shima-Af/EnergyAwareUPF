"""Final frozen twin contract checks, including real saved-model integration."""

import json
from pathlib import Path
import numpy as np
import pytest

from revision.evaluation.revised_twin import CONFIG,RevisionTwin
from revision.evaluation.twin import ROOT,sha256
from revision.evaluation.validate_static import run_static_validation,validation_inputs


@pytest.fixture(scope='module')
def frozen_twin():return RevisionTwin()


def test_four_models_code_score_and_weights_are_pinned(frozen_twin):
    assert set(frozen_twin.components)=={'dpdk_power','usr_power','dpdk_qos','usr_qos'}
    config=json.loads(CONFIG.read_text())
    assert not config['controller_results_used_for_selection']
    assert config['selection_precedes_workload_diagnostics']
    for item in config['models'].values():
        for a in item['artifacts']:assert sha256(ROOT/a['path'])==a['sha256']
    frozen_twin.verify_artifacts()


@pytest.mark.parametrize('load',[.02,1.,8.72,20.,100.,400.,5000.])
def test_all_actions_are_finite_physical_outcomes(frozen_twin,load):
    for action in range(3):
        r=frozen_twin.raw_prediction(action,load)
        assert not r.failures and r.power_w>=0 and 0<=r.qos<=1
        assert r.load_mbps==load


@pytest.mark.parametrize('load',[.02,10.,40.,100.,400.])
def test_balanced_load_sum_power_and_per_instance_qos(frozen_twin,load):
    one=frozen_twin.outcome(1,load/2);two=frozen_twin.outcome(2,load)
    assert two.power_w==2*one.power_w
    assert two.qos==one.qos


def test_repeated_and_reloaded_scalar_inference_identical(frozen_twin):
    reloaded=RevisionTwin()
    assert frozen_twin.identity==reloaded.identity
    for action in range(3):
        for load in (8.72,40.,400.):
            assert frozen_twin.outcome(action,load)==reloaded.outcome(action,load)==reloaded.outcome(action,load)


@pytest.mark.parametrize('action',[True,False,-1,3,1.5,'1'])
def test_bad_actions_rejected(frozen_twin,action):
    with pytest.raises(ValueError):frozen_twin.outcome(action,20.)


@pytest.mark.parametrize('action,load',[(0,0.),(1,0.),(2,.01),(0,5001.),(0,np.nan),(1,np.inf),(2,-1.)])
def test_unmeasured_or_invalid_inputs_fail_without_repair(frozen_twin,action,load):
    with pytest.raises(ValueError):frozen_twin.outcome(action,load)


def test_tampered_config_cannot_load_model(tmp_path):
    cfg=json.loads(CONFIG.read_text());cfg['models']['usr_power']['artifacts'][0]['sha256']='0'*64
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='identity mismatch'):RevisionTwin(path)


def test_static_validation_contract_horizon_energy_events_and_repeat(frozen_twin):
    horizon,contract,meta=validation_inputs()
    summary,_,_=run_static_validation(frozen_twin,horizon,contract,meta['protocol_sha256'])
    assert not summary['final_test_evaluated'] and summary['all_three_successful']
    for i,label in enumerate(['DPDK','USR(1)','USR(2)']):
        r=summary['baselines'][label]
        assert r['slots']==504 and r['deterministic_repeat_equal']
        assert r['type_switches']==(0 if i==0 else 1)
        assert r['usr_instances_launched']==i
        assert r['blocked_requests']==0


def test_final_evidence_never_uses_traffic_final_test():
    path=ROOT/'revision/reports/final_twin/integration/input_census.json'
    data=json.loads(path.read_text())
    for mode,d in data.items():
        assert set(d['partitions'])=={'train','validation'}
        for p in d['partitions'].values():
            for a in p['actions'].values():assert a['all_source_rows']['invalid']==0


def test_dpdk_power_validation_selection_and_budget():
    base=ROOT/'revision/reports/final_twin/dpdk_power'
    selection=json.loads((base/'selection_frozen.json').read_text())
    runs=json.loads((base/'neural_runs.json').read_text())
    assert [(r['seed'],r['status']) for r in runs]==[(42,'completed'),(123,'completed')]
    assert sum(r['epochs'] for r in runs)<=800
    scores=selection['validation_scores']
    assert selection['selected_name']==min(scores,key=lambda name:scores[name]['rmse_w'])
    assert not selection['internal_test_scored_before_selection']
    assert not selection['controller_results_used'] and not selection['search_expanded']


def test_dpdk_power_exact_disjoint_training_rows_and_scaler():
    import pandas as pd
    protocol=json.loads((ROOT/'revision/reports/final_twin/dpdk_power/protocol.json').read_text())
    rows=protocol['source_rows']
    assert not set(rows['train']) & set(rows['validation'])
    assert not (set(rows['train'])|set(rows['validation'])) & set(rows['internal_test'])
    config=protocol['config']
    frame=pd.read_csv(config['source_csv'])
    manifest=json.loads((ROOT/'revision/models/final_dpdk_power/neural_seed_123/manifest.json').read_text())
    train=frame.iloc[rows['train']]
    assert set(train['Test Set'])=={1,2,3}
    assert set(frame.iloc[rows['validation']]['Test Set'])=={4}
    assert set(frame.iloc[rows['internal_test']]['Test Set'])=={5}
    assert manifest['training_source_rows']==rows['train']
    transformed=np.log1p(train['Objective Throughput (DL/UL)'].to_numpy())
    assert manifest['input_scaler']['mean']==float(transformed.mean())
    assert manifest['input_scaler']['scale']==float(transformed.std())

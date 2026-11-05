"""Versioned score/protocol verification; no model learning in Task 2G."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[2]
SPEC=ROOT/'revision/specifications/qos_score_v1.json'
PROTOCOL=ROOT/'revision/configs/qos_validation_v1.json'


def from_spec(frame,spec):
    components=[]
    for item in spec['components']:
        if item['id']=='throughput':
            value=np.clip(frame[item['numerator_column']]/(frame[item['denominator_column']]+item['denominator_epsilon_mbps']),0,1)
        else:
            raw=frame[item['column']]*item.get('input_multiplier',1.)
            cap=item['normalization_cap'];value=1-np.clip(raw,0,cap)/cap
        components.append(value.to_numpy())
    weights=np.asarray([c['weight'] for c in spec['components']]);weights/=weights.sum()
    return (np.stack(components,axis=1)*weights).sum(axis=1)


@pytest.mark.parametrize('mode',['dpdk','oai'])
def test_versioned_spec_reproduces_all_historical_labels(mode):
    frame=pd.read_csv(f'/home/ubuntu/DDT_UPF_Selection_DRL/data/processed/{mode}_processed.csv')
    spec=json.loads(SPEC.read_text())
    np.testing.assert_allclose(from_spec(frame,spec),frame.QoS_w,atol=2e-16,rtol=0)


def test_score_threshold_implies_component_bounds_but_not_full_delivery():
    spec=json.loads(SPEC.read_text());weights=np.asarray([c['weight'] for c in spec['components']])
    lower=1-(1-spec['threshold']['value'])/weights
    np.testing.assert_allclose(lower,[.75,.5,.5,.5],atol=1e-15)
    assert np.dot(weights,[.75,1,1,1])==pytest.approx(.9)
    assert np.dot(weights,[1,.5,1,1])==pytest.approx(.9)


def test_two_point_five_percent_loss_can_pass_composite_target():
    spec=json.loads(SPEC.read_text())
    frame=pd.DataFrame({'DN_Rx_Mbits/s':[100.+1e-9],'Objective Throughput (DL/UL)':[100.],
       'downlink_delay_ms':[0.],'RAN_jitter':[0.],'RAN_PLR_interval':[.025]})
    assert from_spec(frame,spec)[0]==pytest.approx(.9)
    assert .025/1e-6==pytest.approx(25000)


def test_exceeding_a_historical_cap_cannot_pass_the_exact_score_target():
    weights=np.asarray([.4,.2,.2,.2])
    for component in (1,2,3):
        scores=np.ones(4);scores[component]=0.
        assert np.dot(weights,scores)<.9


def test_protocol_freezes_sweeps_and_limits_learning_budget():
    p=json.loads(PROTOCOL.read_text())
    assert p['split']=={'train':[1,2,3],'validation':[4],'internal_test':[5]}
    assert not p['random_row_splitting']
    assert len(p['datasets'])*len(p['candidates_per_mode'])*len(p['seeds'])==p['total_scientific_nn_fits']==8
    assert p['selection']['partition']=='validation'
    assert not p['internal_test']['select_or_expand_search_after_test']
    assert not p['models_trained_in_task_2g']
    assert all(c['output_activation']=='sigmoid' and c['hidden_units']==[64,32,16] for c in p['candidates_per_mode'])


def test_raw_unit_evidence_reports_missing_matches_instead_of_imputing():
    data=json.loads((ROOT/'revision/reports/task2g/raw_unit_counter_summary.json').read_text())
    for mode,d in data.items():
        assert d['max_throughput_conversion_difference_mbps']==0
        assert d['max_jitter_difference_us']==0
        assert d['ran_jitter_equals_raw_Avg_us']+d['unmatched_raw_fields']['raw_ran_jitter_us']==d['rows']
        assert d['dn_rx_mbps_equals_raw_kbps_div1000']+d['unmatched_raw_fields']['raw_dn_rx_kbps']==d['rows']

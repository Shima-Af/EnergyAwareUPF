"""No training or final-test parsing: trace provenance, map and workload metrics."""
from datetime import datetime,timedelta
import csv,json,math
import numpy as np
import pandas as pd
import pytest
from revision.stage9.core import segments,relu_pieces
from revision.stage9.run import CFG,OUT
from revision.forecast_service.core import data
from revision.evaluation.twin import ROOT,sha256


def fixture():
    times=tuple(datetime(2019,4,1)+timedelta(minutes=15*i) for i in range(2520))
    # Deterministic fixture, never an experimental workload.
    return np.arange(2520,dtype=float)+1,times


@pytest.mark.parametrize('name',list(CFG['scenarios']))
def test_training_selection_never_uses_validation_values(name):
    loads,times=fixture();spec=CFG['scenarios'][name]
    a=segments(loads,times,name,spec)
    poison=loads.copy();poison[2016:]=1e9
    b=segments(poison,times,name,spec)
    for field in ['indices','factors','loads']:
        assert np.array_equal(a['train'][field],b['train'][field])
    assert np.array_equal(a['validation']['indices'],b['validation']['indices'])
    assert np.array_equal(a['validation']['factors'],b['validation']['factors'])
    assert set(a['train']['indices']).isdisjoint(a['validation']['indices'])
    assert np.all(a['train']['indices']<2016) and np.all(a['validation']['indices']<2520)
    assert np.array_equal(loads,fixture()[0])


def test_rapid_window_is_natural_contiguous_earliest_tie_and_mixed_joins_flagged():
    loads,times=fixture()
    rapid=segments(loads,times,'rapid',CFG['scenarios']['rapid_real_day'])
    assert rapid['train']['indices'].tolist()==list(range(96,192))
    assert rapid['validation']['indices'].tolist()==list(range(2016,2112))
    mixed=segments(loads,times,'mixed',CFG['scenarios']['mixed_daily'])
    for d in mixed.values():
        assert np.array_equal(d['joins'][1:],np.diff(d['factors'])!=0)
        assert np.array_equal(d['loads'],loads[d['indices']]*d['factors'])


def test_scalar_geometry_finds_all_relu_segments():
    class Dense:
        def __init__(self,w,b,activation):self.w=np.array(w);self.b=np.array(b);self.activation=activation
        def get_weights(self):return self.w,self.b
        def get_config(self):return {'activation':self.activation}
    class Net:
        layers=[Dense([[1,-1]],[-1,2],'relu'),Dense([[1],[1]],[0],'sigmoid')]
    pieces=relu_pieces(Net(),{'mean':0.,'scale':1.},[.01,20.])
    assert len(pieces)==3
    for l,h,m,b in pieces:
        z=(l+h)/2
        assert float((m*z+b)[0])==pytest.approx(max(0,z-1)+max(0,2-z))


def test_no_final_suffix_accepted():
    loads,times=fixture()
    with pytest.raises(ValueError):
        segments(np.r_[loads,10],times+times[:1],'original',CFG['scenarios']['original'])


@pytest.mark.parametrize('workload',['netflix','youtube'])
def test_real_scenarios_are_exact_trace_transformations_and_metrics(workload,monkeypatch):
    from revision.evaluation.data import Timeline
    def forbidden(*args,**kwargs):raise AssertionError('Full-file reader forbidden')
    monkeypatch.setattr(Timeline,'read_csv',forbidden)
    times,loads,_=data(workload)
    frame=pd.read_csv(OUT/'scenario_samples.csv');frame=frame[frame.workload==workload]
    metrics=json.loads((OUT/'scenario_metrics.json').read_text())
    actions=json.loads((OUT/'action_metrics.json').read_text())
    for name,spec in CFG['scenarios'].items():
        expected=segments(loads,times,name,spec)
        for role,d in expected.items():
            sub=frame[(frame.scenario==name)&(frame.role==role)]
            assert sub.source_index.tolist()==d['indices'].tolist()
            assert sub.source_timestamp.tolist()==sub.scenario_timestamp.tolist()==d['timestamps']
            assert np.allclose(sub.load_mbps,d['loads'],rtol=1e-15,atol=1e-14)
            assert sub.load_mbps.between(.02,5000).all()
            m=next(r for r in metrics if (r['workload'],r['scenario'],r['role'])==(workload,name,role))
            q=sub[[f'qos_{a}' for a in range(3)]].to_numpy();bad=q<.9
            assert sum(m['risk_counts'].values())==len(sub)
            assert m['risk_counts']['all_unsafe']==int(bad.all(axis=1).sum())
            assert m['risk_counts']['avoidable']==int((bad.any(axis=1)&~bad.all(axis=1)).sum())
            for a in range(3):
                value=next(r for r in actions if (r['workload'],r['scenario'],r['role'],r['action'])==(workload,name,role,a))
                assert value['violations']==int(bad[:,a].sum())
                assert value['energy_wh']==pytest.approx(float(sub[f'power_{a}_w'].sum()*.25),abs=1e-10)
                assert np.allclose(sub[f'qos_penalty_{a}'],30*np.maximum(0,.9-q[:,a]),atol=1e-13)


def test_frozen_protocol_map_and_profile_evidence():
    frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    assert frozen['protocol_sha256']==sha256(ROOT/'revision/stage9/protocol.json')
    for name,digest in frozen['implementation_hashes'].items():assert sha256(ROOT/name)==digest
    regions=json.loads((OUT/'qos_regions.json').read_text())
    intervals=regions['regions'];assert intervals[0]['lower_mbps']==.02 and intervals[-1]['upper_mbps']==5000
    assert all(a['upper_mbps']==b['lower_mbps'] for a,b in zip(intervals,intervals[1:]))
    assert any(r['risk_region']=='all_unsafe' for r in intervals)
    assert any(r['risk_region']=='all_safe' for r in intervals)
    checks=json.loads((OUT/'inference_profile_checks.json').read_text())
    assert all(c['class_disagreements']==0 for c in checks)
    assert max(c['max_qos_difference'] for c in checks)<2e-6
    assert max(c['max_power_difference_w'] for c in checks)<2e-6


def test_context_boundary_joins_keep_natural_and_imposed_components_separate():
    frame=pd.read_csv(OUT/'context_boundary_joins.csv')
    assert len(frame)==32
    assert np.allclose(frame.transformed_ramp_mbps,
                       frame.natural_component_mbps+frame.imposed_component_mbps,atol=1e-10)
    assert frame[frame.scenario=='mixed_daily'].artificial_factor_join.all()
    assert not frame[frame.scenario!='mixed_daily'].artificial_factor_join.any()

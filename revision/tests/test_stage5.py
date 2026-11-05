"""Focused request-rule, shared-actuator and frozen replay contract checks."""
import json
from types import SimpleNamespace as S
from pathlib import Path
import numpy as np
import pytest
from revision.stage5.baselines import SlicesExtendedManual,ComcomHistorical,Region,utility,profile_regions
from revision.evaluation.contract import Outcome,State,Contract,actuate
from revision.evaluation.twin import ROOT,sha256

class Twin:
    def __init__(self,outcomes):self.outcomes=outcomes
    def outcome(self,a,load):return self.outcomes[a]


def manual(regions,margin=26.):
    return SlicesExtendedManual(Twin([Outcome(1,.9)]*3),regions,0.,2.,margin)


def test_manual_utility_has_consistent_score_scales():
    assert utility(.9,1.,performance_weight=.2,energy_weight=.8,minimum=0,maximum=2)==58.
    assert utility(.9,3.,performance_weight=.2,energy_weight=.8,minimum=0,maximum=2)==18.


def test_historical_preserves_mixed_scales_binary_actions_and_dpdk_tie():
    twin=Twin([Outcome(1.,.9),Outcome(1.01,1.),Outcome(0.,1.)])
    c=ComcomHistorical(twin)
    o=S(forecast=S(load_mbps=20.),state=State(2,4))
    assert c.predict(o,deterministic=True)==0 # efficiency dominates even better USR QoS
    assert c.utilities(20.)==pytest.approx([.3*.9+.7*100*(3.58-1)/3.58,.3+.7*100*(3.58-1.01)/3.58])
    assert ComcomHistorical(Twin([Outcome(1,.9)]*3)).predict(o,deterministic=True)==0


@pytest.mark.parametrize('load,expected',[(59.,0),(60.,0),(85.,0),(86.,0),(86.001,1),(113.999,1),(114.,0),(120.,0)])
def test_three_region_hysteresis_requires_interior(load,expected):
    c=manual([Region(.02,60.,2),Region(60.,140.,1),Region(140.,400.,0)])
    c.utilities=lambda x:[0.,2.,0.] # middle winner at these loads
    if load<60:c.utilities=lambda x:[0.,0.,2.]
    # Region USR2 ends at60:59 is held; middle interior is (86,114).
    assert c.request(load,0)==expected


def test_overlapping_hysteresis_eliminates_narrow_region_without_forcing_usr2():
    c=manual([Region(.02,70.,1),Region(70.,100.,2),Region(100.,400.,0)])
    c.utilities=lambda x:[0.,0.,1.]
    for load in [70.,80.,85.,99.,100.]:assert c.request(load,0)==0


def test_hysteresis_uses_executed_state_not_last_rejected_request():
    c=manual([Region(.02,100.,1),Region(100.,400.,0)])
    c.utilities=lambda x:[0.,1.,0.]
    assert c.request(20.,0)==1
    # A denied command does not mutate the request rule's state.
    assert c.request(90.,0)==0
    assert c.request(90.,1)==1
    with pytest.raises(ValueError):c.request(401.,0)


def test_region_detection_preserves_nonmonotone_regions_and_omits_dominated_action():
    f=lambda x:np.array([0.,1.-(x-2)**2,-10.])
    grid=np.linspace(.02,4,100)
    regions=profile_regions(grid,[f(x) for x in grid],f)
    assert [r.action for r in regions]==[0,1,0]
    assert [r.right for r in regions[:-1]]==pytest.approx([1.,3.],abs=1e-5)


def test_shared_actuator_four_slot_gap_and_binary_rule_from_usr2():
    state=State(0,4);contract=Contract()
    t=actuate(state,2,contract);assert t.executed==2 and t.usr_instances_launched==2
    for _ in range(3):
        t=actuate(t.next_state,0,contract);assert t.blocked and t.executed==2
    t=actuate(t.next_state,0,contract);assert not t.blocked and t.executed==0


def test_frozen_sources_and_profiling_only_regions():
    p=ROOT/'revision/reports/stage5/baseline_frozen.json'
    if not p.exists():pytest.skip('Profiling freeze not executed yet')
    f=json.loads(p.read_text())
    from revision.stage5.evaluate import verify_frozen
    verify_frozen(f)
    assert not f['validation_used_for_rule_selection'] and not f['construction_uses_workload_outcomes']
    assert f['configuration']['slices_extended_manual']['margin_mbps']==26
    assert f['configuration']['comcom_historical']['active_margin_mbps']==0
    assert sha256(ROOT/'revision/reports/stage5/profiling_curves.csv')==f['profiling_grid_sha256']


def test_post_run_same_horizons_repeats_and_unchanged_stage4_results():
    p=ROOT/'revision/reports/stage5/comparison.json'
    if not p.exists():pytest.skip('Stage 5 validation replay not executed yet')
    import pandas as pd
    f=json.loads(p.read_text());rows=f['controllers'];assert len(rows)==16
    assert len({m['horizon_identity'] for m in rows.values()})==1
    assert all(m['slots']==504 and m['deterministic_repeat_equal'] for m in rows.values())
    reference=None
    for name,m in rows.items():
        data=pd.read_csv(p.parent/f'{name}_steps.csv')
        horizon=data[['timestamp','source_index','actual_load_mbps','forecast_load_mbps']]
        if reference is None:reference=horizon
        else:pd.testing.assert_frame_equal(reference,horizon)
        assert np.allclose(data.power_w*.25,data.energy_wh,rtol=0,atol=1e-14)
        assert sum(m['action_counts'])==504
        if name.startswith(('ppo_','recurrent_ppo_')):assert m['stage4_metrics_exactly_reproduced']
    assert rows['comcom_historical']['action_counts'][2]==0
    assert f['all_static_actions_predicted_safe']


def test_historical_loop_cooldown_equals_common_actuator_for_all_binary_requests():
    from itertools import product
    for commands in product((0,1),repeat=9):
        current=0;cooldown=0;state=State(0,4)
        for requested in commands:
            if requested!=current and cooldown<=0:
                current=requested;cooldown=4
            cooldown=max(0,cooldown-1)
            t=actuate(state,requested,Contract());state=t.next_state
            assert t.executed==current


def test_historical_plot_selector_not_called_by_evaluation_loop():
    import ast
    root=ROOT/'revision/reports/stage5/sources'
    tree=ast.parse((root/'main_evaluation_baseline-rule.py').read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr=='decide' for n in ast.walk(tree))
    assert 'allow_multi_oai: False' in (root/'main_config.yaml').read_text()
    assert 'margin_mbps: 23.0' in (root/'main_config.yaml').read_text()
    assert '0.3 * perf_d' in (root/'main_compile_baselines_table.py').read_text()


def test_uploaded_slices_source_parameters_and_identity():
    p=ROOT/'revision/reports/stage5/uploaded_source_reconciliation.json'
    if not p.exists():pytest.skip('User source not uploaded yet')
    f=json.loads(p.read_text())
    assert sha256(ROOT/f['uploaded_path'])==f['uploaded_sha256']
    assert not f['baseline_definition_changed']

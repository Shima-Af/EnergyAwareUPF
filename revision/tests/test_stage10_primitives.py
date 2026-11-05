"""Synthetic arithmetic only; fixture budgets are NOT scientific service policy."""
from dataclasses import replace
import math
import pytest
from revision.stage10.primitives import Budgets,KPIs,compliance,activation_burst_wh,transition_energy_wh,serving_compliance


@pytest.fixture
def budgets():return Budgets(.95,1.,10.,.01,'fraction',None)


@pytest.mark.parametrize('field,value',[('throughput_fraction',.94),('delay_ms',1.01),('jitter_us',10.1),('loss',.0101)])
def test_no_good_kpi_can_compensate_for_one_failure(budgets,field,value):
    result=compliance(replace(KPIs(1.,0.,0.,0.),**{field:value}),budgets)
    assert not result['compliant'] and sum(result['kpi_violations'])==1
    assert result['severity']>0


def test_boundary_equality_and_severity(budgets):
    assert compliance(KPIs(.95,1.,10.,.01),budgets)['compliant']
    result=compliance(KPIs(.475,1.25,15.,.02),budgets)
    assert result['normalized_exceedances']==pytest.approx((.5,.25,.5,1.))
    assert result['severity']==1.


@pytest.mark.parametrize('kwargs',[{'loss_limit':0},{'loss_units':'percent'},
                                  {'loss_units':'packets_per_sample'}, {'delay_ms':math.nan}])
def test_ambiguous_or_undefined_policy_rejected(budgets,kwargs):
    with pytest.raises(ValueError):replace(budgets,**kwargs)


def test_burst_ratio_and_two_launches_are_charged_once():
    one=activation_burst_wh(burst_ratio=.37,attributed_idle_w=.015,per_instance_seconds=3.3,launched_instances=1)
    two=activation_burst_wh(burst_ratio=.37,attributed_idle_w=.015,per_instance_seconds=3.3,launched_instances=2)
    assert one==pytest.approx(.37*.015*3.3/3600) and two==2*one
    result=transition_energy_wh(outgoing_w=2,incoming_w=1,outgoing_seconds=6.6,slot_seconds=900,activation_wh=two)
    assert result['operational_wh']==pytest.approx((2*6.6+893.4)/3600)
    assert result['total_wh']==result['operational_wh']+two


def test_no_physical_launch_energy_for_draining_without_launch():
    assert activation_burst_wh(burst_ratio=.37,attributed_idle_w=.015,per_instance_seconds=3.3,launched_instances=0)==0
    with pytest.raises(ValueError):transition_energy_wh(outgoing_w=1,incoming_w=2,outgoing_seconds=901,slot_seconds=900,activation_wh=0)


def test_safe_incoming_cannot_hide_outgoing_startup_failure(budgets):
    old=KPIs(.5,0,0,0);new=KPIs(1,0,0,0)
    assert not serving_compliance(old,new,budgets,outgoing_seconds=24,slot_seconds=900)['compliant']
    assert serving_compliance(old,new,budgets,outgoing_seconds=0,slot_seconds=900)['compliant']


def test_phase_severity_is_worst_not_time_diluted(budgets):
    old=KPIs(1,2,0,0);new=KPIs(1,0,0,.03)
    result=serving_compliance(old,new,budgets,outgoing_seconds=3.3,slot_seconds=900)
    assert result['severity']==pytest.approx(2.)
    assert result['kpi_violations']==(False,True,False,True)

"""Synthetic kernel/oracle tests and integrity checks; no policy training."""
from datetime import datetime,timedelta
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from revision.evaluation.contract import State,Forecast
from revision.evaluation.data import Timeline,Partition,DataSpec,EvaluationHorizon
from revision.stage10.completion.engine import Engine,Dynamics,Reward,PhysicalOutcome
from revision.stage10.completion.oracles import solve
from revision.stage10.completion.intervals import difference,candidate_fraction,authenticate_population
from revision.stage10.completion.environment import ControlEnv


class Fixture:
    synthetic=True
    def outcome(self,a,load):
        return PhysicalOutcome([.8,.02*load,.025*load][a],1.,[0.,load/10,load/20][a],0.,0.)


def engine():return Engine(Fixture(),Dynamics(.82,.015,0.),Reward(.3,'train','fixture'),synthetic=True)


def test_unknown_and_reset_counters_are_never_zero_labels():
    t=datetime(2025,1,1)
    for previous,current,dt in [(None,10,3),(10,5,3),(10,20,6),(10,float('nan'),3)]:
        value,status=difference(previous,current,t,t+timedelta(seconds=dt))
        assert value is None and status!='valid'
    assert difference(10,10,t,t+timedelta(seconds=3))==(0,'valid')
    assert candidate_fraction(0,0)[0] is None
    assert candidate_fraction(6,5)[0] is None
    with pytest.raises(ValueError):authenticate_population(definition='',source='',same_direction=True,same_interval=True,same_population=False,reset_rule_verified=False)


def test_real_engine_fails_without_physical_target_authentication():
    class Unverified(Fixture):provenance={}
    with pytest.raises(ValueError,match='loss_denominator_verified'):
        Engine(Unverified(),Dynamics(.82,.015,0.),Reward(.3,'train','fixture'))


def test_launch_scaling_and_safe_incoming_still_exposes_outgoing_violations():
    e=engine();r=e.step(State(0,4),2,100.)
    assert r['outgoing_serving_seconds']==6.6
    assert r['activation_wh']==pytest.approx(2*.37*.015*3.3/3600)
    assert r['ready_idle_wh']==pytest.approx(.015*3.3/3600)
    r=e.step(State(1,4),2,60.)
    assert r['outgoing_serving_seconds']==3.3
    assert r['incoming'].policy()['compliant'] and not r['compliant']
    assert r['unsafe_serving_seconds']==3.3
    assert e.step(State(2,4),1,60.)['outgoing_serving_seconds']==0


def test_blocked_change_has_no_burst_or_extra_change_penalty():
    r=engine().step(State(1,1),0,60.)
    assert r['transition'].blocked and r['activation_wh']==0
    assert r['reward_components']['stability_cost']==0
    assert r['reward']==-sum(r['reward_components'].values())


def test_energy_bound_prevents_sec_overwhelming_avoidable_violation():
    r=Reward(.3,'train','fixture')
    safe=sum(r.terms(.3,False,0,True).values())
    unsafe=sum(r.terms(.00001,True,.001,False).values())
    assert safe<unsafe
    with pytest.raises(ValueError):Reward(.3,'validation','wrong role')


@pytest.mark.parametrize('objective',['reward','energy','qos_constrained_energy'])
@pytest.mark.parametrize('actions',[(0,1),(0,1,2)])
def test_transition_oracle_matches_all_requested_sequences(objective,actions):
    e=engine();loads=[10.,60.,20.,100.,10.];paths=[]
    for requests in product(actions,repeat=len(loads)):
        s=State(0,4);cost=Fraction();changes=0;seq=[]
        for t,(a,load) in enumerate(zip(requests,loads)):
            step=e.step(s,a,load)
            if objective=='qos_constrained_energy' and not step['compliant']:break
            value=-.995**t*step['reward'] if objective=='reward' else step['total_wh']
            cost+=Fraction.from_float(value);tr=step['transition'];changes+=tr.executed!=s.action
            seq.append(tr.executed);s=tr.next_state
        else:paths.append((cost,changes,tuple(seq)))
    result=solve(e,loads,objective,actions=actions)
    assert result['feasible']==bool(paths)
    if paths:
        exact=Fraction(result['exact_numerator'],result['exact_denominator'])
        if objective=='reward':exact=-exact
        assert (exact,result['reconfigurations'],tuple(result['actions']))==min(paths)


@pytest.mark.parametrize('schema,n',[('instant',13),('history',109),('forecast',14),('hybrid',110)])
def test_environment_is_causal_and_uses_common_scoring_times(schema,n):
    start=datetime(2019,4,1);times=tuple(start+timedelta(minutes=15*i) for i in range(105))
    def horizon(target):
        x=[10.]*105;x[96]=target
        return EvaluationHorizon(Partition(Timeline('netflix',times,tuple(x),timedelta(minutes=15),'fixture'),'train',0,105),DataSpec(96,96))
    class Ledger:
        def at(self,i,c):return Forecast(c.decision_at,c.completed_interval_at,c.decision_at,c.history_mbps[-1])
    a=ControlEnv(horizon(10.),engine(),Ledger(),schema);b=ControlEnv(horizon(100.),engine(),Ledger(),schema)
    oa,_=a.reset();ob,_=b.reset();assert len(oa)==n and np.array_equal(oa,ob)
    assert a.step(1)[1]!=b.step(1)[1]


def test_saved_independent_components_have_no_loss_or_transfer_training():
    root=Path(__file__).resolve().parents[2];out=root/'revision/reports/stage10/completion'
    f=pd.read_csv(out/'rebuilt_forecasts.csv')
    assert len(f)==11712 and f.source_index.max()==2519
    assert (f.fit_stop_exclusive<=f.source_index).all()
    for (w,scenario,role),g in f.groupby(['workload','scenario','role']):
        assert g.source_index.tolist()==list(range(96,2016) if role=='train' else range(2016,2520))
        assert w=='netflix' or (g.fit_stop_exclusive==2016).all()
    meta=list((root/'revision/models/stage10/kpi').rglob('manifest.json'))
    assert len(meta)==6
    for p in meta:
        m=json.loads(p.read_text());assert m['target'] in ('delay_ms','jitter_us','throughput_fraction')
        assert 2700<=len(m['train_source_rows'])<=2760


def test_active_flow_offer_and_counter_unknowns_are_explicit():
    root=Path(__file__).resolve().parents[2];out=root/'revision/reports/stage10/completion'
    checks=pd.read_csv(out/'active_offer_mapping.csv')
    assert len(checks)==440 and checks.matches.all()
    assert set(checks[checks.side=='dn'].direction)=={'downlink'}
    assert set(checks[checks.side=='ue'].direction)=={'uplink'}
    f=pd.read_csv(out/'interval_feasibility.csv')
    assert f.physical_loss_fraction.isna().all() and not f.loss_population_authenticated.any()
    assert f.loc[f.receiver_status!='numeric_only_population_unverified','candidate_receiver_expected_fraction'].isna().all()


def test_model_reload_domain_and_missing_loss_gate():
    from revision.stage10.completion.provider import FrozenKPIModel,KPIProvider
    root=Path(__file__).resolve().parents[2]
    for p in (root/'revision/models/stage10/kpi').rglob('manifest.json'):
        model=FrozenKPIModel(p)
        x=[.01,1.,50.,400.,5000.]
        assert np.array_equal(model.predict(x),FrozenKPIModel(p).predict(x))
        with pytest.raises(ValueError):model.predict([0])
    auth={k:True for k in ['throughput_direction_verified','throughput_offered_mapping_verified','loss_numerator_verified',
                           'loss_denominator_verified','loss_timebase_verified','loss_direction_verified']}
    with pytest.raises(ValueError,match='Missing authenticated KPI'):
        KPIProvider({'dpdk':object(),'usr':object()},{'dpdk':{},'usr':{}},auth)


def test_constrained_oracle_does_not_hide_initial_outgoing_failure():
    class BadInitial(Fixture):
        def outcome(self,a,load):return PhysicalOutcome(1.,1.,10. if a==0 else 0.,0.,0.)
    e=Engine(BadInitial(),Dynamics(.82,.015,0.),Reward(.3,'train','fixture'),synthetic=True)
    assert not solve(e,[10.,10.],'qos_constrained_energy')['feasible']


def test_reward_calibration_is_training_only_and_uses_no_sec():
    root=Path(__file__).resolve().parents[2]
    r=json.loads((root/'revision/reports/stage10/completion/reward_calibration_v2.json').read_text())
    assert r['role']=='train' and r['calibration_transitions']==4*1920*9
    assert r['energy_scale_wh']>0 and r['no_sec']
    assert r['activation_energy_included_once'] and not r['old_launch_reward_penalties_used']
    assert not r['full_reward_diagnostics_complete']


def test_training_gate_fails_before_creating_scientific_run_directory(tmp_path):
    from revision.stage10.completion.train import fit
    class Provider(Fixture):provenance={}
    e=engine();e.provider=Provider()
    with pytest.raises(ValueError,match='loss_denominator_verified'):
        fit(tmp_path/'forbidden',e,{}, {},algorithm='ppo',schema='instant',seed=42)
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.parametrize('recurrent',[False,True])
def test_new_validation_keeps_recurrent_state_and_normalizer_frozen(recurrent):
    from revision.stage10.completion.train import evaluate
    from types import SimpleNamespace
    class Normalizer:
        obs_rms={'count':1234}
        def normalize_obs(self,x):return x
    class Model:
        calls=0
        def predict(self,x,**kwargs):
            if recurrent:
                assert bool(kwargs['episode_start'][0])==(self.calls==0)
                assert kwargs['state']==(None if self.calls==0 else self.calls)
            self.calls+=1
            return np.array([0]),self.calls
    class Ledger:
        def at(self,i,c):return Forecast(c.decision_at,c.completed_interval_at,c.decision_at,10.)
    start=datetime(2019,4,1);times=tuple(start+timedelta(minutes=15*i) for i in range(105))
    h=EvaluationHorizon(Partition(Timeline('netflix',times,(10.,)*105,timedelta(minutes=15),'fixture'),'train',0,105),DataSpec(96,96))
    result=evaluate(Model(),Normalizer(),h,engine(),Ledger(),'hybrid',recurrent)
    assert len(result['steps'])==9 and result['violations']==0


def test_strong_and_legacy_adapters_use_new_point_score_and_common_actuator():
    from types import SimpleNamespace
    from revision.stage10.completion.baselines import PointScoreTwin,Strong,legacy_rules
    e=engine();o=SimpleNamespace(state=State(0,4),forecast=SimpleNamespace(load_mbps=10.))
    p=PointScoreTwin(e).outcome(1,10.)
    assert p.qos==e.provider.outcome(1,10.).policy()['qos']
    rules=legacy_rules(e,regions=[{'left':.02,'right':5000.,'action':1}],power_minimum=0,power_maximum=5.,training_margin=10.)
    assert len(rules)==7 and rules['static_2'].predict(o,deterministic=True)==2
    chosen=Strong(e).predict(o,deterministic=True)
    assert chosen in (0,1,2) and e.step(o.state,chosen,10.)['total_wh']>0


@pytest.mark.parametrize('fail',[False,True])
def test_checkpoint_validation_preserves_all_rng_streams_even_on_exception(tmp_path,fail):
    import random,pickle,torch
    from types import SimpleNamespace
    from revision.stage10.completion.train import Callback
    start=datetime(2019,4,1);times=tuple(start+timedelta(minutes=15*i) for i in range(105))
    h=EvaluationHorizon(Partition(Timeline('netflix',times,(10.,)*105,timedelta(minutes=15),'fixture'),'train',0,105),DataSpec(96,96))
    class Ledger:
        def at(self,i,c):return Forecast(c.decision_at,c.completed_interval_at,c.decision_at,10.)
    class Norm:
        obs_rms={'count':999}
        def normalize_obs(self,x):return x
        def save(self,path):Path(path).write_bytes(pickle.dumps(self.obs_rms))
    norm=Norm()
    class Model:
        num_timesteps=24576
        def get_env(self):return norm
        def save(self,path):Path(path).write_bytes(b'synthetic-test-not-SB3-policy')
        def predict(self,x,**kwargs):
            random.random();np.random.rand();torch.rand(1)
            if fail:raise RuntimeError('fixture validation failure')
            return np.array([0]),None
    e=engine();e.provider.provenance={'synthetic':True}
    callback=Callback(tmp_path,e,{'fixture':(h,h,Ledger())},'instant',False);callback.model=Model()
    before=(pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state()),torch.random.get_rng_state().clone())
    if fail:
        with pytest.raises(RuntimeError,match='fixture validation failure'):callback.capture()
    else:callback.capture()
    assert pickle.dumps(random.getstate())==before[0] and pickle.dumps(np.random.get_state())==before[1]
    assert torch.equal(torch.random.get_rng_state(),before[2])


def test_completed_scenario_horizons_and_ledgers_never_use_final_reader(monkeypatch):
    from revision.stage10.completion.data import horizons,Ledger
    def forbidden(*args,**kwargs):raise AssertionError('Full CSV reader prohibited')
    monkeypatch.setattr(Timeline,'read_csv',forbidden)
    for scenario in ['original','low','moderate_overload','mixed']:
        for workload in ['netflix','youtube']:
            train,val=horizons(workload,scenario);ledger=Ledger(workload,scenario)
            assert len(val.indices)==504 and len(val.partition.timeline.timestamps)==2520
            assert (train is None)==(workload=='youtube')
            for i in [2016,2519]:
                assert ledger.at(i,val.context(i)).target_at==val.partition.timeline.timestamps[i]
                assert 'actual_mbps' not in ledger.rows[i]
            if train is not None:
                assert len(train.indices)==1920
                assert ledger.at(96,train.context(96)).observed_through==train.partition.timeline.timestamps[95]

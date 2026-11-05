"""Causal W96 schema, actuator, RNG and frozen forecast integration contracts."""
from datetime import datetime,timedelta
from dataclasses import replace,asdict
from types import SimpleNamespace as S
import json,pickle
import numpy as np
import pytest
from revision.stage7.common import CFG,ROOT,OUT,MODELS,horizons,sha256
from revision.stage7.pipeline import feature_names,vector,ControlEnv,Ledger,run_controller
from revision.evaluation.contract import State,Outcome,Forecast,observe,Contract,evaluate
from revision.evaluation.selection import Candidate
from revision.evaluation.data import Timeline,Partition,DataSpec,EvaluationHorizon


class Twin:
    identity='a'*64
    def outcome(self,action,load):return Outcome(1.+action+load/100.,.95)


class FakeLedger:
    digest='b'*64
    def at(self,i,c):return Forecast(c.decision_at,c.completed_interval_at,c.decision_at,300.+i)


class Controller:
    def reset(self,*,seed):self.i=0
    def predict(self,o,*,deterministic):self.i+=1;return (self.i//3)%3


def fixture_horizon():
    start=datetime(2019,4,1);times=tuple(start+timedelta(minutes=15*i) for i in range(120))
    t=Timeline('netflix',times,tuple(float(i+1) for i in range(120)),timedelta(minutes=15),'fixture')
    return EvaluationHorizon(Partition(t,'train',0,120),DataSpec(96,96))


@pytest.mark.parametrize('schema,dimension',[('instant',13),('history',109),('forecast',14),('hybrid',110)])
def test_all_schema_common_signals_dimensions_and_order(schema,dimension):
    h=fixture_horizon();c=h.context(96);f=FakeLedger().at(96,c);o=observe(c,schema,State(2,3),Outcome(4.,.92),f)
    x=vector(o);assert len(x)==dimension==len(feature_names(schema))
    assert x[:13]==pytest.approx([96/400.,.92,4,4/96,1,0,0,1,.75,c.decision_calendar.time_sin,c.decision_calendar.time_cos,c.decision_calendar.weekday_sin,c.decision_calendar.weekday_cos])
    if schema in ('history','hybrid'):assert x[13:109]==pytest.approx(np.arange(1,97)/400.)
    else:assert o.history_mbps==()
    if schema in ('forecast','hybrid'):assert x[-1]==pytest.approx(396/400.)
    else:assert o.forecast is None


@pytest.mark.parametrize('schema',['instant','history','forecast','hybrid'])
def test_reset_step_last_slot_and_missing_previous_telemetry(schema):
    h=fixture_horizon();env=ControlEnv(h,Twin(),FakeLedger(),Contract(),schema)
    x,_=env.reset(seed=42);assert x[1:5].tolist()==[0,0,0,0]
    assert x[5:9].tolist()==[1,0,0,1]
    obs,reward,end,trunc,info=env.step(2)
    assert info['index']==96 and info['outcome']['power_w']==pytest.approx(3.97,rel=0,abs=1e-14)
    assert obs[0]==pytest.approx(97/400.) and obs[1]==pytest.approx(.95) and obs[4]==1
    # t+1,t+2,t+3 blocked; t+4 eligible, common semantics independent of schema.
    for i in range(3):
        _,_,_,_,info=env.step(0);assert info['transition']['blocked']
    _,_,_,_,info=env.step(0);assert not info['transition']['blocked']
    while not end:obs,reward,end,trunc,info=env.step(0)
    assert info['index']==119 and env.position==24 and obs.shape==(len(feature_names(schema)),)


def test_target_load_cannot_affect_its_observation_or_history():
    h=fixture_horizon();poisoned=list(h.partition.timeline.load_mbps);poisoned[96]=99999.
    changed=replace(h,partition=replace(h.partition,timeline=replace(h.partition.timeline,load_mbps=tuple(poisoned))))
    for schema in CFG['schemas']:
        a=ControlEnv(h,Twin(),FakeLedger(),Contract(),schema);b=ControlEnv(changed,Twin(),FakeLedger(),Contract(),schema)
        assert np.array_equal(a.reset()[0],b.reset()[0])
        assert a.step(1)[4]['outcome']!=b.step(1)[4]['outcome']


def test_frozen_data_horizons_and_provider_changes():
    train,val,c=horizons('netflix');yt,transfer,_=horizons('youtube')
    assert yt is None and len(train.indices)==1920 and len(val.indices)==len(transfer.indices)==504
    assert val.timestamps==transfer.timestamps and train.spec==DataSpec(96,96,1)
    ledger=Ledger(ROOT/CFG['forecast_ledgers']['netflix'])
    assert sum(r['family']=='persistence' for i,r in ledger.rows.items() if i<2016)==912
    for i in [96,672,1008,1344,1680,2016,2519]:assert ledger.at(i,(train if i<2016 else val).context(i)).target_at==val.partition.timeline.timestamps[i]
    assert CFG['recalibrated_margins']['netflix']==pytest.approx(14.268831,abs=1e-6)
    assert CFG['recalibrated_margins']['youtube']==pytest.approx(8.353154,abs=1e-6)


def test_new_runner_equals_shared_contract_on_netflix():
    _,val,contract=horizons('netflix');ledger=Ledger(ROOT/CFG['forecast_ledgers']['netflix'])
    from revision.stage3.pipeline import LedgerForecaster
    for schema in CFG['schemas']:
        name='fixture';candidate=Candidate(name,(('bundle','c'*64),('twin',Twin.identity),('forecast',ledger.digest)))
        expected=evaluate(val,Controller(),Twin(),candidate=candidate,contract=contract,schema=schema,forecaster=LedgerForecaster(ledger,val))
        actual=run_controller(val,Controller(),Twin(),ledger,contract,name,'c'*64,schema)
        assert expected==actual


def test_transfer_and_final_testing_guards():
    train,val,c=horizons('netflix');_,yt,_=horizons('youtube');ledger=Ledger(ROOT/CFG['forecast_ledgers']['youtube'])
    if not (OUT/'all_netflix_selections_frozen.json').exists():
        with pytest.raises(ValueError):run_controller(yt,Controller(),Twin(),ledger,c,'x','d'*64,'hybrid',transfer=True)
    with pytest.raises(ValueError):run_controller(train,Controller(),Twin(),ledger,c,'x','d'*64,'instant')
    with pytest.raises(ValueError):run_controller(replace(val,partition=replace(val.partition,role='final_test')),Controller(),Twin(),ledger,c,'x','d'*64,'instant')
    from revision.stage7.train import Callback
    with pytest.raises(ValueError):Callback(None,'ppo','history',42,yt,Twin(),ledger,c)
    with pytest.raises(ValueError):ControlEnv(replace(yt,partition=replace(yt.partition,role='train')),Twin(),ledger,c,'history')


@pytest.mark.parametrize('algorithm',['ppo','recurrent_ppo'])
def test_zero_update_policy_bundle_reload_schema_and_rng(tmp_path,algorithm):
    import torch
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    from revision.stage7.train import save_bundle,load_bundle,Callback
    from revision.stage4.rng_isolation import isolated_rng
    from revision.tests.test_stage4_rng_isolation import state,same
    torch.set_num_threads(1);train,val,c=horizons('netflix');ledger=Ledger(ROOT/CFG['forecast_ledgers']['netflix']);schema='history'
    env=VecNormalize(DummyVecEnv([lambda:ControlEnv(train,Twin(),ledger,c,schema)]),norm_reward=False)
    cls=RecurrentPPO if algorithm=='recurrent_ppo' else PPO
    model=cls('MlpLstmPolicy' if algorithm=='recurrent_ppo' else 'MlpPolicy',env,n_steps=8,batch_size=8,seed=42,device='cpu',
              policy_kwargs={'lstm_hidden_size':8} if algorithm=='recurrent_ppo' else {})
    path=tmp_path/'bundle';save_bundle(model,env,path,algorithm,schema,42,Twin(),ledger,c)
    before=state()
    with isolated_rng():
        policy=load_bundle(path,algorithm,schema,Twin(),c);policy.reset(seed=0);o=observe(val.context(2016),schema,State(0,4),None,ledger.at(2016,val.context(2016)))
        a=policy.predict(o,deterministic=True);b=policy.predict(o,deterministic=True)
        assert policy.calls==2
        if algorithm=='recurrent_ppo':assert min(policy.state_norms)>0
        norm=pickle.dumps(policy.normalizer.obs_rms);policy.predict(o,deterministic=True);assert norm==pickle.dumps(policy.normalizer.obs_rms)
        policy.normalizer.close()
    assert same(before,state()) and hasattr(Callback.capture,'__wrapped__')
    with pytest.raises(ValueError):load_bundle(path,algorithm,'forecast',Twin(),c)
    env.close()


def test_cache_identity_and_real_frozen_twin_agreement():
    if not (OUT/'prepared.json').exists():pytest.skip('Preparation not completed')
    from revision.stage7.pipeline import CachedTwin
    from revision.evaluation.revised_twin import RevisionTwin
    twin=CachedTwin();real=RevisionTwin()
    assert twin.identity==real.identity
    for name in ['netflix','youtube']:
        _,val,_=horizons(name)
        for i in [2016,2519]:
            load=val.partition.timeline.load_mbps[i]
            for a in range(3):assert twin.outcome(a,load)==real.outcome(a,load)


def test_unused_forecast_is_not_exposed_to_instant_or_history():
    h=fixture_horizon()
    class Changed(FakeLedger):
        def at(self,i,c):return Forecast(c.decision_at,c.completed_interval_at,c.decision_at,9999.)
    for schema in ['instant','history']:
        a=ControlEnv(h,Twin(),FakeLedger(),Contract(),schema);b=ControlEnv(h,Twin(),Changed(),Contract(),schema)
        assert np.array_equal(a.reset()[0],b.reset()[0])
        for action in [0,2,1,1,0]:
            xa,ra,_,_,_=a.step(action);xb,rb,_,_,_=b.step(action)
            assert np.array_equal(xa,xb) and ra==rb


def test_prespecified_paired_tests_and_holm():
    from revision.stage7.analyze import paired,holm
    assert paired([1]*5)['exact_two_sided_sign_flip_p']==.0625
    assert paired([0]*5)['exact_two_sided_sign_flip_p']==1.
    assert paired([1,2,3,4,5])['exact_two_sided_sign_flip_p']==paired([-1,-2,-3,-4,-5])['exact_two_sided_sign_flip_p']
    d={str(i):{'exact_two_sided_sign_flip_p':.0625} for i in range(5)};holm(d)
    assert all(v['holm_adjusted_p']==.3125 for v in d.values())


def test_completed_campaign_preserves_all40_selections_endpoints_and_normalizers():
    if not (OUT/'worker_results.json').exists():pytest.skip('Scientific campaign in progress')
    from revision.stage7.common import directory
    workers=json.loads((OUT/'worker_results.json').read_text());assert len(workers)==40 and all(w['exit_code']==0 for w in workers)
    for algorithm in CFG['algorithms']:
        for schema in CFG['schemas']:
            for seed in CFG['seeds']:
                path=directory(algorithm,schema,seed);selection=json.loads((path/'selection.json').read_text())
                assert selection['actual_timesteps']==300000 and not selection['youtube_used'] and not selection['final_test_used']
                assert len(selection['validation_scores'])==14
                assert [s['timesteps'] for s in selection['validation_scores']]==CFG['checkpoint_timesteps']
                assert selection['selected_checkpoint']==min(selection['validation_scores'],key=lambda s:(-s['total_reward'],s['timesteps']))['checkpoint']
                assert len((path/'training_updates.jsonl').read_text().splitlines())==74
                assert len((path/'episodes.jsonl').read_text().splitlines())==156
                remainder=json.loads((path/'training_remainder.json').read_text())['partial_episodes_not_independent']
                assert [r['length'] for r in remainder]==[120]*4
                for s in selection['validation_scores']:
                    checkpoint=ROOT/s['checkpoint'];m=json.loads((checkpoint/'bundle.json').read_text())
                    assert m['schema']==schema and m['features']==feature_names(schema)
                    assert m['normalizer_count']==pytest.approx(s['timesteps']+4+.0001,abs=1e-7)
                    for f,h in m['artifacts'].items():assert sha256(checkpoint/f)==h
                result=json.loads((path/'result.json').read_text())
                assert result['deterministic_repeat_equal'] and result['optimizer_updates']==74
                if algorithm=='recurrent_ppo':assert result['recurrent_calls']==504 and result['recurrent_state_norm_min']>0


def test_completed_comparisons_share_horizons_and_transfer_cannot_select():
    path=OUT/'comparison.json'
    if not path.exists():pytest.skip('Post-training evaluation pending')
    import pandas as pd
    comparison=json.loads(path.read_text());assert not comparison['final_test_used'] and not comparison['youtube_used_for_selection']
    frozen=json.loads((OUT/'all_netflix_selections_frozen.json').read_text());assert frozen['count']==40
    for workload,d in comparison['workloads'].items():
        assert len(d['controllers_selected'])==47 and len(d['controllers_endpoint'])==40
        assert len({m['horizon_identity'] for m in d['controllers_selected'].values()})==1
        for name,m in d['controllers_selected'].items():
            rows=pd.read_csv(OUT/workload/f'{name}_selected_steps.csv')
            assert rows.timestamp.tolist()==d['timestamps'] and rows.source_index.tolist()==list(range(2016,2520))
            assert m['energy_wh']==pytest.approx(math_fsum(rows.power_w*.25),abs=1e-10)
            assert m['total_reward']==pytest.approx(math_fsum(rows.reward),abs=1e-9)
            assert sum(m['action_counts'])==m['slots']==504
        for item in frozen['selections'].values():
            p=ROOT/item['selected'];assert sha256(p/'bundle.json')==item['selected_bundle_sha256']


def math_fsum(values):
    import math
    return math.fsum(values)


def test_stage7_preparation_inputs_do_not_use_legacy_full_file_reader(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Legacy full-file reader invoked')
    monkeypatch.setattr(Timeline,'read_csv',forbidden)
    for name in ['netflix','youtube']:
        train,val,_=horizons(name)
        assert len(val.partition.timeline.timestamps)==2520
        assert val.partition.stop==2520

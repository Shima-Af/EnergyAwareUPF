"""Preflight controller verification; no PPO parameter updates in these tests."""
import json,pickle
from datetime import datetime
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import revision.stage3  # Establish frozen TensorFlow CPU profile before SB3.
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from revision.evaluation.contract import State,Forecast,Outcome,observe,actuate,account,Contract
from revision.stage3.common import inputs,OUT,MODELS,CFG
from revision.stage3.forecast import arrays,predictions
from revision.stage3.pipeline import ControlEnv,PolicyController,RuleController,ForecastLedger,vector,feature_names

class Twin:
    def outcome(self,a,load):return Outcome([1.,.7,.6][a], [.99,.95,.92][a])

class Ledger:
    def at(self,i,c):return Forecast(c.decision_at,c.completed_interval_at,c.decision_at,c.forecast_input_mbps[-1])

@pytest.fixture
def env(prepared,contract):return ControlEnv(prepared.horizons['train'],Twin(),Ledger(),contract,schema='instant')


def test_environment_no_final_test(prepared,contract):
    with pytest.raises(ValueError):ControlEnv(prepared.horizons['final_test'],Twin(),Ledger(),contract)


def test_vector_causal_completed_telemetry_and_initial_mask():
    train,_,contract=inputs();c=train.context(96)
    f=Forecast(c.decision_at,c.completed_interval_at,c.decision_at,123.)
    o=observe(c,'hybrid',State(0,4),None,f);v=vector(o)
    assert v.shape==(109,) and len(feature_names())==109
    assert v[0]==np.float32(c.history_mbps[-1]/400)
    np.testing.assert_allclose(v[13:108],np.asarray(c.history_mbps)/400,rtol=np.finfo(np.float32).eps,atol=0)
    assert v[4]==0 and all(v[1:4]==0) and v[5:8].tolist()==[1,0,0]
    assert v[-1]==np.float32(123/400)
    p=replace(o,previous_outcome=Outcome(.8,.95));v=vector(p)
    assert v[4]==1 and v[1]==np.float32(.95)
    assert v[3]==np.float32(.8/c.history_mbps[-1])


def test_terminal_scoring_has_no_extra_slot(env):
    obs,_=env.reset();indices=[]
    while True:
        obs,r,done,_,info=env.step(2);indices.append(info['index'])
        if done:break
    assert indices==list(env.horizon.indices)
    assert np.array_equal(obs,np.zeros_like(obs))


def test_environment_outcomes_reward_and_blocked_actions(env):
    env.reset();_,r,_,_,info=env.step(2)
    assert info['transition']['usr_instances_launched']==2
    assert r==pytest.approx(-100*.6/env.horizon.partition.timeline.load_mbps[env.indices[0]]-.024)
    for _ in range(3):
        _,_,_,_,info=env.step(0)
        assert info['transition']['blocked'] and info['accounting']['orchestration_term']==0
    _,_,_,_,info=env.step(0)
    assert not info['transition']['blocked'] and info['accounting']['orchestration_term']==.03

@pytest.mark.parametrize('before,after,cost',[(0,1,.012),(0,2,.024),(1,2,.012),(2,1,.003),(1,0,.03),(2,0,.03)])
def test_launch_switch_scale_equations(before,after,cost):
    c=Contract();t=actuate(State(before,4),after,c);r=account(Outcome(1,.8),10,900,t,c)
    assert r.orchestration_term==cost and r.energy_wh==.25
    assert r.qos_term==pytest.approx(3.)


def test_forecast_windows_ignore_target_and_future():
    _,v,_=inputs();t=v.partition.timeline;loads=np.asarray(t.load_mbps)
    indices=list(range(96,100));a=arrays(loads,t.timestamps,indices,400,True)
    poisoned=loads.copy();poisoned[100:]=1e9
    assert np.array_equal(a,arrays(poisoned,t.timestamps,indices,400,True))
    for k,i in enumerate(indices):assert np.array_equal(a[k,:,0],loads[i-96:i].astype(np.float32)/400)
    poisoned=loads.copy();poisoned[96]=1e9
    assert np.array_equal(a[0],arrays(poisoned,t.timestamps,[96],400,True)[0])


def test_forecast_ledger_cutoffs_and_full_common_horizon():
    train,val,_=inputs();ledger=ForecastLedger()
    assert set(ledger.rows)==set(train.indices)|set(val.indices)
    for h in [train,val]:
        for i in h.indices:
            c=h.context(i);f=ledger.at(i,c)
            assert f.target_at==c.decision_at and f.observed_through<c.decision_at
            assert int(ledger.rows[i]['fit_stop_exclusive'])<=i
    assert sum(r['kind']=='explicit_cold_start_persistence' for r in ledger.rows.values())==96
    i=2016;ledger.rows[i]['fit_stop_exclusive']='2017'
    with pytest.raises(ValueError):ledger.at(i,val.context(i))


def test_forecast_saved_fit_rows_and_scalers():
    train,_,_=inputs();loads=np.asarray(train.partition.timeline.load_mbps)
    for path in (MODELS/'forecast').glob('*/manifest.json'):
        m=json.loads(path.read_text());stop=m['fit_source_rows_half_open'][1]
        assert stop<=2016 and m['target_rows_half_open']==[96,stop]
        assert m['scale_training_max_mbps']==float(max(loads[:stop]))
        assert not m['uses_validation_labels']
    selected=json.loads((OUT/'forecast_selection.json').read_text())
    assert selected['selected']==min(selected['validation_metrics'],key=lambda n:(selected['validation_metrics'][n]['rmse_mbps'],n))


def test_forecast_reloaded_selected_model_matches_ledger():
    from revision.evaluation.twin import _runtime
    _,tf,_,_=_runtime();train,val,_=inputs()
    selected=json.loads((OUT/'forecast_selection.json').read_text())['selected']
    path=MODELS/'forecast'/selected;manifest=json.loads((path/'manifest.json').read_text())
    model=tf.keras.models.load_model(path/'model.keras',compile=False)
    timeline=val.partition.timeline
    p=predictions(model,manifest['scale_training_max_mbps'],timeline.load_mbps,timeline.timestamps,list(val.indices),selected=='lstm_calendar')
    ledger=ForecastLedger()
    assert np.array_equal(p,np.asarray([float(ledger.rows[i]['forecast_mbps']) for i in val.indices],dtype=np.float32))


def test_rule_has_scaling_and_shared_guard(prepared,contract):
    h=prepared.horizons['train'];c=h.context(next(iter(h.indices)));rule=RuleController(Twin())
    o=observe(c,'hybrid',State(0,4),None,Ledger().at(0,c))
    assert rule.predict(o,deterministic=True)==2
    blocked=replace(o,state=State(1,1))
    assert actuate(blocked.state,rule.predict(blocked,deterministic=True),contract).blocked


def test_recurrent_adapter_retains_state_and_resets_episode():
    class Norm:
        training=False
        def normalize_obs(self,x):return x
    class Model:
        def __init__(self):self.inputs=[]
        def predict(self,x,state,episode_start,deterministic):
            self.inputs.append((state,bool(episode_start[0])))
            s=0 if state is None else state[0][0]
            return np.array([2]),(np.array([s+1.]),np.array([s+2.]))
    train,_,_=inputs();c=train.context(96);o=observe(c,'hybrid',State(0,4),None,Ledger().at(96,c))
    m=Model();a=PolicyController(m,Norm(),True);a.reset(seed=42)
    a.predict(o,deterministic=True);a.predict(o,deterministic=True)
    assert m.inputs[0]==(None,True) and m.inputs[1][0] is not None and not m.inputs[1][1]
    a.reset(seed=123);a.predict(o,deterministic=True);assert m.inputs[-1]==(None,True)


def test_normalizer_saved_state_frozen_during_inference(env,tmp_path):
    norm=VecNormalize(DummyVecEnv([lambda:env]),norm_reward=False)
    norm.reset();norm.step([1]);norm.save(tmp_path/'norm.pkl')
    frozen=VecNormalize.load(tmp_path/'norm.pkl',DummyVecEnv([lambda:env]));frozen.training=False
    before=pickle.dumps(frozen.obs_rms);x=np.ones((1,13),np.float32)
    p=frozen.normalize_obs(x);q=frozen.normalize_obs(x)
    assert np.array_equal(p,q) and pickle.dumps(frozen.obs_rms)==before
    assert np.array_equal(p,norm.normalize_obs(x))

@pytest.mark.parametrize('algorithm',['ppo','recurrent_ppo'])
def test_real_policy_checkpoint_normalizer_reload_without_training(algorithm,tmp_path):
    import torch
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    from revision.stage3.pilot import save_bundle,load_bundle
    torch.set_num_threads(1)
    train,val,contract=inputs();ledger=ForecastLedger()
    twin=Twin();twin.identity='a'*64
    normalizer=VecNormalize(DummyVecEnv([lambda:ControlEnv(train,twin,ledger,contract)]),norm_reward=False)
    cls=RecurrentPPO if algorithm=='recurrent_ppo' else PPO
    model=cls('MlpLstmPolicy' if algorithm=='recurrent_ppo' else 'MlpPolicy',normalizer,
        n_steps=8,batch_size=8,device='cpu',seed=42,
        policy_kwargs={'lstm_hidden_size':8} if algorithm=='recurrent_ppo' else {})
    directory=tmp_path/'checkpoint'
    save_bundle(model,normalizer,directory,algorithm,42,ledger,twin,contract)
    a=load_bundle(directory,algorithm,val,twin,ledger,contract)
    b=load_bundle(directory,algorithm,val,twin,ledger,contract)
    a.reset(seed=42);b.reset(seed=42);ctx=val.context(2016)
    o=observe(ctx,'hybrid',State(0,4),None,ledger.at(2016,ctx))
    for _ in range(3):assert a.predict(o,deterministic=True)==b.predict(o,deterministic=True)
    if algorithm=='recurrent_ppo':
        assert len(a.state_norms)==3 and a.state_norms==b.state_norms and min(a.state_norms)>0
    before=pickle.dumps(a.normalizer.obs_rms);a.predict(o,deterministic=True)
    assert pickle.dumps(a.normalizer.obs_rms)==before


def test_checkpoint_evaluation_after_updates_and_includes_endpoint():
    from revision.stage3.pilot import SelectionCallback
    callback=SelectionCallback(None,'ppo',42,None,None,None,None)
    seen=[];callback.capture=lambda:seen.append(callback.num_timesteps)
    callback.num_timesteps=0;callback._on_rollout_start()
    callback.num_timesteps=8192
    assert callback._on_step() and seen==[]
    callback._on_rollout_start();assert seen==[8192]
    # learn() explicitly captures the last post-update endpoint.
    callback.num_timesteps=32768;callback.capture();assert seen==[8192,32768]


@pytest.mark.skipif(not (OUT/'controller_comparison.json').exists(),reason='Post-pilot artifact verification')
def test_completed_pilot_fixed_budget_checkpoint_pairs_and_validation_selection():
    from pathlib import Path
    from revision.evaluation.twin import ROOT,sha256
    from revision.stage3.pilot import load_bundle
    _,val,c=inputs();ledger=ForecastLedger();twin=Twin()
    frozen=json.loads((OUT/'pilot_protocol_frozen.json').read_text());twin.identity=frozen['frozen_twin']['twin_id']
    lines=[json.loads(s) for s in (OUT/'candidate_runs.jsonl').read_text().splitlines()]
    assert len(lines)==8 and sum(r['status']=='completed' for r in lines)==4
    for algorithm in CFG['algorithms']:
        for seed in CFG['seeds']:
            directory=MODELS/'controllers'/algorithm/f'seed_{seed}'
            selection=json.loads((directory/'selection.json').read_text())
            assert selection['actual_timesteps']==32768 and not selection['final_test_used']
            scores=selection['validation_scores'];assert [s['timesteps'] for s in scores]==[8192,16384,24576,32768]
            best=min(scores,key=lambda s:(-s['total_reward'],s['timesteps']))
            assert selection['selected_checkpoint']==best['checkpoint']
            for score in scores:
                path=ROOT/score['checkpoint'];m=json.loads((path/'bundle.json').read_text())
                assert m['timesteps']==score['timesteps']
                for artifact,digest in m['artifacts'].items():assert sha256(path/artifact)==digest
                a=load_bundle(path,algorithm,val,twin,ledger,c)
                assert a.normalizer.obs_rms.count==pytest.approx(score['timesteps']+4+.0001,abs=1e-7)
                assert not a.normalizer.training and not a.normalizer.norm_reward


@pytest.mark.skipif(not (OUT/'controller_comparison.json').exists(),reason='Post-pilot artifact verification')
def test_pilot_all_controllers_share_horizon_accounting_and_actuator():
    import math
    from revision.evaluation.contract import State,actuate,Contract
    c=json.loads((OUT/'controller_comparison.json').read_text())
    _,val,contract=inputs()
    assert c['validation_only'] and c['horizon']==val.identity
    for name,r in c['controllers'].items():
        steps=pd.read_csv(OUT/f'{name}_steps.csv')
        assert steps.source_index.tolist()==list(val.indices)
        assert steps.timestamp.tolist()==[t.isoformat() for t in val.timestamps]
        assert math.fsum(steps.power_w*.25)==pytest.approx(r['energy_wh'],abs=1e-10)
        state=State(0,4)
        for _,row in steps.iterrows():
            transition=actuate(state,int(row.requested),contract)
            assert transition.executed==row.executed and transition.blocked==row.blocked
            state=transition.next_state


@pytest.mark.skipif(not (OUT/'twin_sensitivity.json').exists(),reason='Post-pilot artifact verification')
def test_sensitivity_fixed_models_and_validation_horizons():
    d=json.loads((OUT/'twin_sensitivity.json').read_text())
    base=json.loads((OUT/'controller_comparison.json').read_text())['controllers']
    assert d['validation_only'] and d['no_retraining_or_reselection']
    assert set(d['scenarios'])=={'usr_neural_power','load_mean_qos','both'}
    for scenario,rows in d['scenarios'].items():
        assert set(rows)==set(base)
        for n,r in rows.items():
            assert r['slots']==504 and r['horizon_identity']==base[n]['horizon_identity']
            assert r['energy_wh']>=0 and r['min_predicted_qos']>=0

"""Full-budget protocol and optimizer-boundary verification; synthetic fits only."""
import json,pickle
from pathlib import Path
import numpy as np
import pytest
import revision.stage4
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from revision.evaluation.twin import ROOT,sha256
from revision.stage3.pipeline import ControlEnv,ForecastLedger
from revision.stage4.common import CFG,OUT,MODELS,inputs
from revision.stage4.train import ExactPPO,ExactRecurrentPPO,ConvergenceCallback


def test_stage4_preserves_frozen_contract_baseline_and_hyperparameters():
    old=json.loads((ROOT/'revision/stage3/protocol.json').read_text())
    assert CFG['requested_timesteps']==300000 and CFG['seeds']==[42,123,456,789,1024]
    for key in ['n_envs','n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','ent_coef','clip_range','target_kl','schema','rule']:
        assert CFG[key]==old[key]
    assert CFG['checkpoint_timesteps'][-1]==300000 and 32768 in CFG['checkpoint_timesteps']
    assert all(t%4096==0 for t in CFG['checkpoint_timesteps'][:-1])
    assert len(CFG['forecast']['candidates'])==2 and CFG['forecast']['max_epochs']==100
    assert CFG['final_test_allowed'] is False


@pytest.mark.parametrize('recurrent',[False,True])
def test_exact_budget_trains_short_last_rollout_and_retains_normalizer(prepared,contract,recurrent):
    from revision.tests.test_stage3 import Twin,Ledger
    import torch
    torch.set_num_threads(1)
    env=VecNormalize(DummyVecEnv([lambda:ControlEnv(prepared.horizons['train'],Twin(),Ledger(),contract,schema='instant') for _ in range(2)]),norm_reward=False)
    cls=ExactRecurrentPPO if recurrent else ExactPPO
    model=cls('MlpLstmPolicy' if recurrent else 'MlpPolicy',env,n_steps=8,batch_size=4,n_epochs=1,device='cpu',seed=42,
        policy_kwargs={'lstm_hidden_size':8} if recurrent else {})
    model.stage4_updates=[];model.stage4_training_log=None
    class Callback(BaseCallback):
        def _on_step(self):return True
    model.learn(20,callback=Callback())
    assert model.num_timesteps==20
    assert [(r['timesteps'],r['rollout_steps_per_environment']) for r in model.stage4_updates]==[(16,8),(20,2)]
    assert model.n_steps==8  # Regular rollout hyperparameter is not silently changed.
    assert env.obs_rms.count==pytest.approx(22.0001)
    if recurrent:assert model.rollout_buffer.hidden_state_shape==(2,1,2,8)
    assert 'train/entropy_loss' in model.stage4_updates[-1]
    assert 'train/approx_kl' in model.stage4_updates[-1]
    assert 'train/clip_fraction' in model.stage4_updates[-1]


def test_callback_schedule_matches_all_post_update_boundaries():
    callback=ConvergenceCallback(None,'ppo',42,None,None,None,None)
    seen=[];callback.capture=lambda:seen.append(callback.num_timesteps)
    for t in range(0,299009,4096):callback.num_timesteps=t;callback._on_rollout_start()
    callback.num_timesteps=300000;callback.capture()
    assert seen==CFG['checkpoint_timesteps']


@pytest.mark.skipif(not (OUT/'forecasts.csv').exists(),reason='Forecast fitting not complete')
def test_forecast_parameter_and_early_stop_provenance_is_causal():
    ledger=ForecastLedger(OUT/'forecasts.csv');train,val,_=inputs()
    assert set(ledger.rows)==set(train.indices)|set(val.indices)
    for i in train.indices:
        r=ledger.rows[i]
        assert int(r['fit_stop_exclusive'])<=i and int(r['selection_stop_exclusive'])<=i
        ledger.at(i,train.context(i))
    for i in val.indices:
        assert int(ledger.rows[i]['fit_stop_exclusive'])==2016
        ledger.at(i,val.context(i))
    for path in (MODELS/'forecast').glob('*/manifest.json'):
        m=json.loads(path.read_text());fit_stop=m['fit_source_rows_half_open'][1]
        assert fit_stop==m['scale_fit_stop_exclusive'] and fit_stop<=2016
        assert m['scale_training_max_mbps']==max(train.partition.timeline.load_mbps[:fit_stop])
        assert m['parameter_targets_half_open'][1]<=m['early_stopping_targets_half_open'][0]
        if path.parent.name.startswith('prefix_'):
            cutoff=int(path.parent.name.split('_')[-1]);assert m['selection_stop_exclusive']==cutoff
            assert m['early_stopping_targets_half_open'][1]<=cutoff
        assert sha256(path.parent/'model.keras')==m['model_sha256']
    chosen=json.loads((OUT/'forecast_selection.json').read_text())
    assert chosen['selected']==min(chosen['validation_metrics'],key=lambda name:(chosen['validation_metrics'][name]['rmse_mbps'],name))
    assert not chosen['no_controller_selection'] is False


@pytest.mark.skipif(not (OUT/'controller_comparison.json').exists(),reason='Post-training evidence')
def test_all_ten_runs_exact_budget_checkpoints_selection_and_optimizer_logs():
    for alg in CFG['algorithms']:
        for seed in CFG['seeds']:
            path=MODELS/'controllers'/alg/f'seed_{seed}'
            runs=[json.loads(s) for s in (path/'run_ledger.jsonl').read_text().splitlines()]
            assert [r['status'] for r in runs]==['started','completed']
            selected=json.loads((path/'selection.json').read_text());scores=selected['validation_scores']
            assert selected['actual_timesteps']==300000 and not selected['final_test_used']
            assert [s['timesteps'] for s in scores]==CFG['checkpoint_timesteps']
            assert selected['selected_checkpoint']==min(scores,key=lambda s:(-s['total_reward'],s['timesteps']))['checkpoint']
            updates=[json.loads(s) for s in (path/'training_updates.jsonl').read_text().splitlines()]
            assert len(updates)==74 and updates[-1]['timesteps']==300000
            assert updates[-1]['rollout_steps_per_environment']==248
            assert len((path/'episodes.jsonl').read_text().splitlines())==156
            for score in scores:
                b=ROOT/score['checkpoint'];m=json.loads((b/'bundle.json').read_text())
                assert m['phase']=='after_optimizer_update'
                assert m['normalizer_count']==pytest.approx(score['timesteps']+4+.0001,abs=1e-6)
                assert not score['normalizer_updated_by_validation']
                for name,h in m['artifacts'].items():assert sha256(b/name)==h


@pytest.mark.skipif(not (OUT/'forecasts.csv').exists(),reason='Forecast fitting not complete')
def test_stage4_selected_forecast_reloads_exactly():
    from revision.evaluation.twin import _runtime
    from revision.stage3.forecast import predictions
    _,tf,_,_=_runtime();_,val,_=inputs();timeline=val.partition.timeline
    choice=json.loads((OUT/'forecast_selection.json').read_text());path=ROOT/choice['manifest_path']
    manifest=json.loads(path.read_text());model=tf.keras.models.load_model(path.parent/'model.keras',compile=False)
    p=predictions(model,manifest['scale_training_max_mbps'],timeline.load_mbps,timeline.timestamps,list(val.indices),manifest['calendar'])
    ledger=ForecastLedger(OUT/'forecasts.csv')
    assert np.array_equal(p,np.asarray([float(ledger.rows[i]['forecast_mbps']) for i in val.indices],np.float32))


@pytest.mark.skipif(not (OUT/'controller_comparison.json').exists(),reason='Post-training common comparison')
def test_complete_validation_metrics_common_horizon_and_arithmetic():
    import math
    import pandas as pd
    data=json.loads((OUT/'controller_comparison.json').read_text());_,val,_=inputs()
    assert data['validation_only'] and data['horizon']==val.identity and len(data['controllers'])==14
    for name,r in data['controllers'].items():
        rows=pd.read_csv(OUT/f'{name}_steps.csv')
        assert rows.source_index.tolist()==list(val.indices)
        assert rows.timestamp.tolist()==[t.isoformat() for t in val.timestamps]
        assert r['energy_wh']==pytest.approx(math.fsum(rows.power_w*.25),abs=1e-10)
        assert r['total_reward']==pytest.approx(math.fsum(rows.reward),abs=1e-10)
        assert r['slots']==504 and sum(r['action_counts'])==504
    old=json.loads((ROOT/'revision/reports/stage3/controller_comparison.json').read_text())['controllers']
    for a in range(3):assert data['controllers'][f'static_{a}']['energy_wh']==old[f'static_{a}']['energy_wh']


@pytest.mark.skipif(not (OUT/'forecast_sensitivity.json').exists(),reason='Post-training forecast replay')
def test_forecast_replay_is_fixed_policy_validation_only():
    d=json.loads((OUT/'forecast_sensitivity.json').read_text())
    assert d['no_retraining_reselection']
    assert set(d['fixed_stage4_policies'])=={'stage3_lstm','persistence'}
    _,val,_=inputs()
    for scenario,rows in d['fixed_stage4_policies'].items():
        assert len(rows)==14
        for r in rows.values():assert r['horizon_identity']==val.identity and r['slots']==504
    assert len(d['fixed_stage3_policies_new_forecasts'])==4


@pytest.mark.skipif(not (OUT/'scaling_accounting.json').exists(),reason='Post-training local accounting')
def test_scaling_local_counterfactual_is_labeled_and_only_within_usr():
    data=json.loads((OUT/'scaling_accounting.json').read_text())
    assert len(data)==10
    for r in data.values():
        assert r['events']==len(r['records'])
        assert 'not a causal policy ablation' in r['interpretation']
        for event in r['records']:
            assert {event['before'],event['after']}=={1,2}
            assert event['local_energy_saved_wh']==pytest.approx((event['hold_power_w']-event['executed_power_w'])*.25)

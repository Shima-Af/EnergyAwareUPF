"""Fixed four-run pilot; checkpoints pair policies with training-only VecNormalize states."""
import json,copy,pickle,random,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from revision.evaluation.twin import ROOT,sha256
from revision.evaluation.contract import require_comparable
from revision.evaluation.static import StaticController
from revision.evaluation.validate_static import write_csv
from .common import CFG,OUT,MODELS,inputs,dump
from .pipeline import ForecastLedger,CachedTwin,ControlEnv,PolicyController,RuleController,run_controller,summarize,feature_names


def load_bundle(directory,algorithm,horizon,twin,ledger,contract):
    directory=Path(directory);meta=json.loads((directory/'bundle.json').read_text())
    for name,digest in meta['artifacts'].items():
        if sha256(directory/name)!=digest:raise ValueError('Checkpoint/normalizer hash mismatch')
    if meta['twin_id']!=twin.identity or meta['forecast_sha256']!=ledger.digest:raise ValueError('Bundle twin/forecast identity mismatch')
    dummy=DummyVecEnv([lambda:ControlEnv(horizon,twin,ledger,contract)])
    normalizer=VecNormalize.load(directory/'normalizer.pkl',dummy);normalizer.training=False;normalizer.norm_reward=False
    cls=RecurrentPPO if algorithm=='recurrent_ppo' else PPO
    model=cls.load(directory/'policy.zip',device='cpu')
    return PolicyController(model,normalizer,algorithm=='recurrent_ppo')


def save_bundle(model,normalizer,directory,algorithm,seed,ledger,twin,contract):
    directory.mkdir(parents=True)
    model.save(directory/'policy.zip');normalizer.save(directory/'normalizer.pkl')
    dump(directory/'bundle.json',{'algorithm':algorithm,'seed':seed,'timesteps':model.num_timesteps,
        'twin_id':twin.identity,'forecast_sha256':ledger.digest,'protocol_sha256':sha256(ROOT/'revision/stage3/protocol.json'),
        'contract':asdict(contract),'observation_features':feature_names(),
        'implementation_sha256':{'pipeline':sha256(ROOT/'revision/stage3/pipeline.py'),'pilot':sha256(Path(__file__))},
        'normalization':'training-only running observation mean/variance; clip 10; rewards unnormalized',
        'artifacts':{n:sha256(directory/n) for n in ['policy.zip','normalizer.pkl']}})


class SelectionCallback(BaseCallback):
    def __init__(self,directory,algorithm,seed,horizon,twin,ledger,contract):
        super().__init__();self.directory=directory;self.algorithm=algorithm;self.seed=seed
        self.horizon=horizon;self.twin=twin;self.ledger=ledger;self.contract=contract;self.scores=[]
    def _on_step(self):
        return True
    def _on_rollout_start(self):
        if self.num_timesteps and not self.num_timesteps%CFG['eval_every_timesteps']:
            self.capture()
    def capture(self):
        path=self.directory/f'checkpoint_{self.num_timesteps}'
        save_bundle(self.model,self.training_env,path,self.algorithm,self.seed,self.ledger,self.twin,self.contract)
        controller=load_bundle(path,self.algorithm,self.horizon,self.twin,self.ledger,self.contract)
        before=pickle.dumps(controller.normalizer.obs_rms)
        r=run_controller(self.horizon,controller,self.twin,self.ledger,self.contract,path.name,sha256(path/'bundle.json'))
        if pickle.dumps(controller.normalizer.obs_rms)!=before:raise ValueError('Validation changed observation normalization')
        score={'checkpoint':str(path.relative_to(ROOT)),'total_reward':r.summary['total_reward'],
              'timesteps':self.num_timesteps,'metrics':summarize(r),'normalizer_updated_by_validation':False}
        self.scores.append(score);dump(path/'validation.json',score)
        print(self.algorithm,self.seed,'checkpoint',self.num_timesteps,score['metrics'],flush=True)
        return True


def export_steps(name,result,directory):
    rows=[]
    for s in result.steps:
        rows.append({'controller':name,'timestamp':s.context.decision_at.isoformat(),'source_index':s.index,
          'actual_load_mbps':s.load_mbps,'forecast_load_mbps':s.forecast.load_mbps,
          **{k:v for k,v in asdict(s.transition).items() if k!='next_state'},
          **asdict(s.outcome),**asdict(s.accounting)})
    write_csv(directory/f'{name}_steps.csv',rows)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    train,val,contract=inputs();ledger=ForecastLedger();twin=CachedTwin()
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'pilot_protocol_frozen.json',{'config':CFG,'contract':asdict(contract),'train_horizon':train.identity,
       'validation_horizon':val.identity,'frozen_twin':twin.real.manifest,'forecast_sha256':ledger.digest,
       'cache_sha256':twin.census_sha256,'final_test_allowed':False,
       'code_hashes':{p.name:sha256(p) for p in (ROOT/'revision/stage3').glob('*.py')},
       'runtime':{'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','torch_threads':1,
         'stable_baselines3':__import__('stable_baselines3').__version__,'sb3_contrib':__import__('sb3_contrib').__version__}})
    comparison={};results=[]
    for name,controller in [(f'static_{a}',StaticController(a)) for a in range(3)]+[('rule_three_action',RuleController(twin.real))]:
        r=run_controller(val,controller,twin,ledger,contract,name,sha256(ROOT/'revision/stage3/protocol.json'))
        results.append(r);comparison[name]=summarize(r);export_steps(name,r,OUT)
    ledger_path=OUT/'candidate_runs.jsonl'
    def log(item):
        with ledger_path.open('a') as f:f.write(json.dumps(item)+'\n')
    for algorithm in CFG['algorithms']:
        for seed in CFG['seeds']:
            log({'algorithm':algorithm,'seed':seed,'status':'started'})
            directory=MODELS/'controllers'/algorithm/f'seed_{seed}'
            try:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                env=DummyVecEnv([lambda:ControlEnv(train,twin,ledger,contract) for _ in range(CFG['n_envs'])])
                norm=VecNormalize(env,norm_obs=True,norm_reward=False,clip_obs=10.,gamma=CFG['gamma'])
                cls=RecurrentPPO if algorithm=='recurrent_ppo' else PPO
                kwargs={k:CFG[k] for k in ['n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','ent_coef','clip_range','target_kl']}
                policy_kwargs={'lstm_hidden_size':128,'n_lstm_layers':1,'shared_lstm':True,'enable_critic_lstm':False} if algorithm=='recurrent_ppo' else {}
                model=cls('MlpLstmPolicy' if algorithm=='recurrent_ppo' else 'MlpPolicy',norm,seed=seed,device='cpu',verbose=0,policy_kwargs=policy_kwargs,**kwargs)
                callback=SelectionCallback(directory,algorithm,seed,val,twin,ledger,contract)
                model.learn(total_timesteps=CFG['requested_timesteps'],callback=callback)
                callback.capture()  # Final optimizer update is complete before endpoint selection.
                best=min(callback.scores,key=lambda x:(-x['total_reward'],x['timesteps']))
                dump(directory/'selection.json',{'selected_checkpoint':best['checkpoint'],'validation_scores':callback.scores,
                    'selection_objective':'validation_total_reward','final_test_used':False,'actual_timesteps':model.num_timesteps})
                path=ROOT/best['checkpoint'];controller=load_bundle(path,algorithm,val,twin,ledger,contract)
                name=f'{algorithm}_{seed}'
                r=run_controller(val,controller,twin,ledger,contract,name,sha256(path/'bundle.json'))
                replay=run_controller(val,controller,twin,ledger,contract,name,sha256(path/'bundle.json'))
                assert r==replay
                results.append(r);comparison[name]={**summarize(r),'selected_checkpoint':best['checkpoint'],
                    'actual_training_timesteps':model.num_timesteps,'deterministic_repeat_equal':True,
                    'recurrent_state_calls':controller.calls if controller.recurrent else 0,
                    'recurrent_state_norm_min':min(controller.state_norms) if controller.recurrent else None}
                export_steps(name,r,OUT);norm.close();log({'algorithm':algorithm,'seed':seed,'status':'completed','metrics':comparison[name]})
            except Exception as exc:
                log({'algorithm':algorithm,'seed':seed,'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
                raise
    common=require_comparable(tuple(results));dump(OUT/'controller_comparison.json',{'horizon':common,'validation_only':True,'controllers':comparison})
    twin.real.verify_artifacts();print('Pilot completed',comparison,flush=True)

if __name__=='__main__':main()

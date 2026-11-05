"""Exact 300k PPO budgets, complete optimizer/episode logs, and paired checkpoints."""
import argparse,json,pickle,random,traceback,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from revision.evaluation.twin import ROOT,sha256
from revision.evaluation.validate_static import diagnostic_json
from revision.stage3.pipeline import CachedTwin,ControlEnv,ForecastLedger,run_controller,summarize,feature_names
from revision.stage3.pilot import load_bundle as original_load_bundle,export_steps
from .common import CFG,OUT,MODELS,inputs,dump


def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(diagnostic_json(value),allow_nan=False)+'\n')


class ExactBudgetMixin:
    def collect_rollouts(self,env,callback,rollout_buffer,n_rollout_steps):
        remaining=self._total_timesteps-self.num_timesteps
        if remaining%self.n_envs:raise ValueError('Exact budget must be divisible by number of environments')
        steps=min(n_rollout_steps,remaining//self.n_envs)
        if steps<1:raise ValueError('Empty exact-budget rollout')
        if steps!=rollout_buffer.buffer_size:
            kwargs={'buffer_size':steps,'observation_space':self.observation_space,'action_space':self.action_space,
                'device':self.device,'gae_lambda':self.gae_lambda,'gamma':self.gamma,'n_envs':self.n_envs}
            if hasattr(rollout_buffer,'hidden_state_shape'):
                kwargs['hidden_state_shape']=(steps,*rollout_buffer.hidden_state_shape[1:])
            rollout_buffer=type(rollout_buffer)(**kwargs);self.rollout_buffer=rollout_buffer
        return super().collect_rollouts(env,callback,rollout_buffer,n_rollout_steps=steps)
    def train(self):
        super().train()
        metrics={k:(v.item() if isinstance(v,np.generic) else v) for k,v in self.logger.name_to_value.items() if k.startswith('train/')}
        record={'timesteps':self.num_timesteps,'rollout_steps_per_environment':self.rollout_buffer.buffer_size,**metrics}
        self.stage4_updates.append(record)
        if getattr(self,'stage4_training_log',None):append(self.stage4_training_log,record)


class ExactPPO(ExactBudgetMixin,PPO):pass
class ExactRecurrentPPO(ExactBudgetMixin,RecurrentPPO):pass


def load_bundle(path,algorithm,horizon,twin,ledger,contract):
    meta=json.loads((Path(path)/'bundle.json').read_text())
    if meta['algorithm']!=algorithm or meta['protocol_sha256']!=sha256(ROOT/'revision/stage4/protocol.json') or meta['contract']!=asdict(contract):
        raise ValueError('Stage 4 bundle algorithm/protocol/contract mismatch')
    return original_load_bundle(path,algorithm,horizon,twin,ledger,contract)


def save_bundle(model,normalizer,path,algorithm,seed,twin,ledger,contract):
    path.mkdir(parents=True);model.save(path/'policy.zip');normalizer.save(path/'normalizer.pkl')
    dump(path/'bundle.json',{'algorithm':algorithm,'seed':seed,'timesteps':model.num_timesteps,'phase':'after_optimizer_update',
        'twin_id':twin.identity,'forecast_sha256':ledger.digest,'protocol_sha256':sha256(ROOT/'revision/stage4/protocol.json'),
        'contract':asdict(contract),'observation_features':feature_names(),
        'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'revision/stage3/pipeline.py']},
        'normalizer_count':float(normalizer.obs_rms.count),'norm_reward':False,'rollout_buffer_steps':model.rollout_buffer.buffer_size,
        'artifacts':{name:sha256(path/name) for name in ['policy.zip','normalizer.pkl']}})


class ConvergenceCallback(BaseCallback):
    def __init__(self,directory,algorithm,seed,val,twin,ledger,contract):
        super().__init__();self.directory=directory;self.algorithm=algorithm;self.seed=seed
        self.val=val;self.twin=twin;self.ledger=ledger;self.contract=contract;self.scores=[]
        self.running=[self.empty_episode() for _ in range(CFG['n_envs'])]
    @staticmethod
    def empty_episode():
        return {'length':0,'reward':0.,'energy_wh':0.,'qos_shortfall_slots':0,'energy_term':0.,'qos_term':0.,'orchestration_term':0.,
            'type_switches':0,'scale_up':0,'scale_down':0,'blocked':0,'occupancy':[0,0,0]}
    def _on_step(self):
        for i,info in enumerate(self.locals['infos']):
            e=self.running[i];a=info['accounting'];t=info['transition']
            e['length']+=1;e['reward']+=a['reward'];e['energy_wh']+=a['energy_wh'];e['qos_shortfall_slots']+=int(a['qos_violation'])
            for k in ['energy_term','qos_term','orchestration_term']:e[k]+=a[k]
            for k,dest in [('type_switch','type_switches'),('scale_up','scale_up'),('scale_down','scale_down'),('blocked','blocked')]:e[dest]+=int(t[k])
            e['occupancy'][t['executed']]+=1
            if self.locals['dones'][i]:
                append(self.directory/'episodes.jsonl',{'timesteps':self.num_timesteps,'environment':i,**e});self.running[i]=self.empty_episode()
        return True
    def _on_rollout_start(self):
        if self.num_timesteps in CFG['checkpoint_timesteps']:self.capture()
    def capture(self):
        if self.num_timesteps in [s['timesteps'] for s in self.scores]:return
        path=self.directory/f'checkpoint_{self.num_timesteps}'
        save_bundle(self.model,self.training_env,path,self.algorithm,self.seed,self.twin,self.ledger,self.contract)
        controller=load_bundle(path,self.algorithm,self.val,self.twin,self.ledger,self.contract)
        before=pickle.dumps(controller.normalizer.obs_rms)
        result=run_controller(self.val,controller,self.twin,self.ledger,self.contract,path.name,sha256(path/'bundle.json'))
        assert pickle.dumps(controller.normalizer.obs_rms)==before
        score={'timesteps':self.num_timesteps,'checkpoint':str(path.relative_to(ROOT)),
             'total_reward':result.summary['total_reward'],'metrics':summarize(result),'normalizer_updated_by_validation':False}
        dump(path/'validation.json',score);self.scores.append(score)
        print(self.algorithm,self.seed,self.num_timesteps,'reward',score['total_reward'],'Wh',score['metrics']['energy_wh'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--algorithm',choices=CFG['algorithms'],required=True);parser.add_argument('--seed',type=int,choices=CFG['seeds'],required=True)
    args=parser.parse_args();algorithm,seed=args.algorithm,args.seed
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    train,val,contract=inputs();ledger=ForecastLedger(OUT/'forecasts.csv');twin=CachedTwin()
    directory=MODELS/'controllers'/algorithm/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=False)
    ledger_path=directory/'run_ledger.jsonl';start=time.monotonic()
    append(ledger_path,{'status':'started','algorithm':algorithm,'seed':seed,'protocol_sha256':sha256(ROOT/'revision/stage4/protocol.json')})
    try:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        def make_env(i):return lambda:Monitor(ControlEnv(train,twin,ledger,contract),str(directory/f'environment_{i}.monitor.csv'))
        env=DummyVecEnv([make_env(i) for i in range(CFG['n_envs'])]);norm=VecNormalize(env,norm_obs=True,norm_reward=False,clip_obs=10.,gamma=CFG['gamma'])
        cls=ExactPPO if algorithm=='ppo' else ExactRecurrentPPO
        kwargs={k:CFG[k] for k in ['n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','ent_coef','clip_range','target_kl']}
        pk={'lstm_hidden_size':128,'n_lstm_layers':1,'shared_lstm':True,'enable_critic_lstm':False} if algorithm=='recurrent_ppo' else {}
        model=cls('MlpPolicy' if algorithm=='ppo' else 'MlpLstmPolicy',norm,seed=seed,device='cpu',verbose=0,policy_kwargs=pk,**kwargs)
        model.stage4_updates=[];model.stage4_training_log=str(directory/'training_updates.jsonl')
        model.set_logger(configure(str(directory/'sb3_logs'),['csv']))
        callback=ConvergenceCallback(directory,algorithm,seed,val,twin,ledger,contract)
        model.learn(total_timesteps=CFG['requested_timesteps'],callback=callback);callback.capture()
        assert model.num_timesteps==300000 and [s['timesteps'] for s in callback.scores]==CFG['checkpoint_timesteps']
        dump(directory/'training_remainder.json',{'partial_episodes_not_independent':callback.running})
        best=min(callback.scores,key=lambda s:(-s['total_reward'],s['timesteps']))
        dump(directory/'selection.json',{'selected_checkpoint':best['checkpoint'],'validation_scores':callback.scores,
           'selection_objective':'validation_total_reward','final_test_used':False,'actual_timesteps':model.num_timesteps})
        path=ROOT/best['checkpoint'];controller=load_bundle(path,algorithm,val,twin,ledger,contract)
        name=f'{algorithm}_{seed}';result=run_controller(val,controller,twin,ledger,contract,name,sha256(path/'bundle.json'))
        replay=run_controller(val,controller,twin,ledger,contract,name,sha256(path/'bundle.json'));assert result==replay
        export_steps(name,result,OUT)
        metrics={**summarize(result),'selected_checkpoint':best['checkpoint'],'actual_training_timesteps':model.num_timesteps,
             'deterministic_repeat_equal':True,'recurrent_calls':controller.calls if controller.recurrent else 0}
        dump(directory/'result.json',metrics);norm.close();twin.real.verify_artifacts()
        append(ledger_path,{'status':'completed','seconds':time.monotonic()-start,'timesteps':model.num_timesteps,'metrics':metrics})
    except Exception as exc:
        append(ledger_path,{'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-start});raise

if __name__=='__main__':main()

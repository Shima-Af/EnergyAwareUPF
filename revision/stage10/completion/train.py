"""Gated definitive 40-run harness; never invoked before physical target readiness.

Inputs are scenario-keyed (training horizon, validation horizon, forecast ledger).
Only Engine/ControlEnv are used: no legacy reward, outcome twin or policy fallback.
"""
from revision.evaluation.twin import _runtime,sha256,ROOT
_runtime()
import json,math,pickle,random
from pathlib import Path
from dataclasses import asdict,is_dataclass
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from revision.stage4.train import ExactPPO,ExactRecurrentPPO
from revision.stage4.rng_isolation import preserve_rng
from revision.stage10.selected_policy import require_authenticated_targets
from .environment import ControlEnv
from .independent_fits import CFG,dump,BASE


def normalize(value):
    if is_dataclass(value):return normalize(asdict(value))
    if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [normalize(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(normalize(value),allow_nan=False)+'\n')


def authorize(engine,scenarios,freeze):
    require_authenticated_targets(engine.provider.provenance)
    if set(scenarios)!=set(CFG['scenarios']):raise ValueError('All four frozen scenarios required')
    if not freeze.get('all_eight_kpi_models_validated') or not freeze.get('full_preflight_passed'):
        raise ValueError('Authenticated full KPI validation and integration preflight required')
    if freeze.get('protocol_sha256')!=sha256(BASE/'protocol.json'):raise ValueError('Unfrozen campaign protocol')
    implementations=freeze.get('implementation_sha256',{})
    required=[BASE/'engine.py',BASE/'environment.py',BASE/'train.py',BASE/'provider.py',BASE/'data.py',ROOT/'revision/stage10/selected_policy.py']
    if any(implementations.get(str(p.relative_to(ROOT)))!=sha256(p) for p in required):
        raise ValueError('Campaign implementation is not frozen')
    expected={mode+'/'+target for mode in ['dpdk','usr'] for target in ['throughput_fraction','delay_ms','jitter_us','loss_fraction']}
    if set(freeze.get('kpi_manifests',{}))!=expected:raise ValueError('All eight authenticated model manifests required')
    for name,item in freeze['kpi_manifests'].items():
        p=ROOT/item['path']
        if sha256(p)!=item['sha256']:raise ValueError('Model manifest changed')
        m=json.loads(p.read_text())
        if m['target']!=name.split('/')[1] or sha256(p.parent/'model.joblib')!=m['model_sha256']:
            raise ValueError('Wrong KPI model or artifact')
    for name,(train,val,ledger) in scenarios.items():
        if train.partition.timeline.name!='netflix' or train.partition.role!='train' or val.partition.role!='validation' or val.partition.timeline.name!='netflix':
            raise ValueError('Only Netflix train/validation may train or select checkpoints')
        if tuple(train.indices)!=tuple(range(96,2016)) or tuple(val.indices)!=tuple(range(2016,2520)):
            raise ValueError('Frozen common scoring horizons required')


def evaluate(model,normalizer,horizon,engine,ledger,schema,recurrent):
    env=ControlEnv(horizon,engine,ledger,schema);obs,_=env.reset(seed=0)
    hidden=None;starts=np.ones(1,bool);records=[]
    before=pickle.dumps(normalizer.obs_rms)
    while True:
        x=normalizer.normalize_obs(obs[None,:])
        if recurrent:
            action,hidden=model.predict(x,state=hidden,episode_start=starts,deterministic=True);starts[:]=False
        else:action,_=model.predict(x,deterministic=True)
        obs,reward,done,_,info=env.step(int(np.asarray(action).ravel()[0]));records.append(info)
        if done:break
    assert pickle.dumps(normalizer.obs_rms)==before
    return dict(total_reward=math.fsum(r['reward'] for r in records),
        discounted_return=math.fsum(.995**t*r['reward'] for t,r in enumerate(records)),
        energy_wh=math.fsum(r['total_wh'] for r in records),
        activation_wh=math.fsum(r['activation_wh'] for r in records),
        violations=sum(not r['compliant'] for r in records),steps=records)


class Callback(BaseCallback):
    def __init__(self,path,engine,scenarios,schema,recurrent):
        super().__init__();self.path=path;self.engine=engine;self.scenarios=scenarios
        self.schema=schema;self.recurrent=recurrent;self.saved=[]
        self.episodes=[self.empty() for _ in scenarios]
    @staticmethod
    def empty():
        return dict(slots=0,reward=0.,energy_wh=0.,activation_wh=0.,violations=0,
                    type_switches=0,scale_up=0,scale_down=0,blocked=0,
                    energy_cost=0.,strict_qos_cost=0.,severity_cost=0.,stability_cost=0.,
                    severity_sum=0.,unsafe_serving_seconds=0.,startup_delay_exposure_seconds=0.,
                    occupancy=[0,0,0])
    def _on_step(self):
        for i,info in enumerate(self.locals['infos']):
            e=self.episodes[i];e['slots']+=1;e['reward']+=info['reward']
            e['energy_wh']+=info['total_wh'];e['activation_wh']+=info['activation_wh'];e['violations']+=not info['compliant']
            e['severity_sum']+=info['severity'];e['unsafe_serving_seconds']+=info['unsafe_serving_seconds']
            e['startup_delay_exposure_seconds']+=info['startup_delay_exposure_seconds']
            for key,value in info['reward_components'].items():e[key]+=value
            tr=info['transition']
            e['occupancy'][tr.executed]+=1
            for field,key in [('type_switch','type_switches'),('scale_up','scale_up'),('scale_down','scale_down'),('blocked','blocked')]:e[key]+=getattr(tr,field)
            if self.locals['dones'][i]:
                append(self.path/'episodes.jsonl',dict(timesteps=self.num_timesteps,collector=i,**e));self.episodes[i]=self.empty()
        return True
    def _on_rollout_start(self):
        if self.num_timesteps in set(range(24576,300000,24576))|{32768}:self.capture()
    def _on_training_end(self):
        self.capture()
        dump(self.path/'partial_episode_tails.json',normalize(self.episodes))
    @preserve_rng
    def capture(self):
        step=self.model.num_timesteps
        if step in [r['timesteps'] for r in self.saved]:return
        folder=self.path/f'checkpoint_{step}';folder.mkdir()
        self.model.save(folder/'policy.zip');self.training_env.save(folder/'normalizer.pkl')
        metrics={}
        for name,(_,val,ledger) in self.scenarios.items():
            metrics[name]=evaluate(self.model,self.training_env,val,self.engine,ledger,self.schema,self.recurrent)
        dump(folder/'validation.json',normalize(metrics))
        from revision.evaluation.twin import sha256
        dump(folder/'bundle.json',dict(timesteps=step,post_update=True,observation_schema=self.schema,
            artifacts={p.name:sha256(p) for p in [folder/'policy.zip',folder/'normalizer.pkl']},
            reward=asdict(self.engine.reward),dynamics=asdict(self.engine.dynamics),
            kpi_provenance=self.engine.provider.provenance))
        self.saved.append(dict(timesteps=step,score=math.fsum(r['total_reward'] for r in metrics.values())))


def fit(path,engine,scenarios,freeze,*,algorithm,schema,seed):
    authorize(engine,scenarios,freeze)
    plan=CFG['controller_protocol_prepared']
    if algorithm not in plan['algorithms'] or schema not in plan['schemas'] or seed not in plan['seeds']:
        raise ValueError('Unprescribed scientific run')
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    dump(path/'freeze.json',freeze);append(path/'run_ledger.jsonl',dict(status='started',algorithm=algorithm,schema=schema,seed=seed))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    env=VecNormalize(DummyVecEnv([lambda tr=tr,ledger=ledger:ControlEnv(tr,engine,ledger,schema)
                                  for tr,_,ledger in scenarios.values()]),norm_obs=True,norm_reward=False,clip_obs=10.)
    recurrent=algorithm=='recurrent_ppo';cls=ExactRecurrentPPO if recurrent else ExactPPO
    policy_kwargs={'lstm_hidden_size':128,'n_lstm_layers':1,'shared_lstm':True,'enable_critic_lstm':False} if recurrent else {}
    model=cls('MlpLstmPolicy' if recurrent else 'MlpPolicy',env,seed=seed,device='cpu',
         n_steps=1024,batch_size=64,n_epochs=10,learning_rate=.0001,gamma=.995,gae_lambda=.9,
         ent_coef=.15,clip_range=.2,target_kl=.03,policy_kwargs=policy_kwargs,verbose=0)
    model.stage4_updates=[];model.stage4_training_log=path/'optimizer.jsonl'
    callback=Callback(path,engine,scenarios,schema,recurrent)
    try:model.learn(total_timesteps=300000,callback=callback)
    except BaseException as error:
        append(path/'run_ledger.jsonl',dict(status='failed',timesteps=model.num_timesteps,error_type=type(error).__name__,error=str(error)))
        env.close();raise
    assert model.num_timesteps==300000
    selected=max(callback.saved,key=lambda r:(r['score'],-r['timesteps']))
    dump(path/'selection.json',dict(selection=selected,all_checkpoints=callback.saved,criterion='Netflix sum validation reward, earlier step tie-break',budget=300000))
    append(path/'run_ledger.jsonl',dict(status='completed',timesteps=300000));env.close()


def campaign(path,engine,scenarios,freeze):
    """One prescribed 2×4×5 campaign, no automatic retries or losing-run removal."""
    authorize(engine,scenarios,freeze)
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    plan=CFG['controller_protocol_prepared'];dump(path/'campaign_freeze.json',dict(protocol=plan,freeze=freeze))
    for algorithm in plan['algorithms']:
        for schema in plan['schemas']:
            for seed in plan['seeds']:
                fit(path/algorithm/schema/f'seed_{seed}',engine,scenarios,freeze,
                    algorithm=algorithm,schema=schema,seed=seed)

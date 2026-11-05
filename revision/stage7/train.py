"""Forty prescribed independent fits; RNG-isolated, validation-only checkpointing."""
import argparse,json,pickle,random,resource,time,traceback
from dataclasses import asdict
import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from revision.stage4.train import ExactPPO,ExactRecurrentPPO,ConvergenceCallback as PriorCallback,append
from revision.stage4.rng_isolation import preserve_rng,isolated_rng
from .common import CFG,BASE,OUT,MODELS,ROOT,dump,sha256,horizons,directory,run_name
from .pipeline import ControlEnv,CachedTwin,Ledger,PolicyController,feature_names,run_controller,summarize,export_steps


def verify_inputs():
    f=json.loads((OUT/'protocol_frozen.json').read_text())
    for p,h in f['implementation_hashes'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Stage7 frozen code changed: '+p)
    for p,h in f['source_hashes'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Prior source changed: '+p)
    return f


def save_bundle(model,normalizer,path,algorithm,schema,seed,twin,ledger,contract):
    path.mkdir(parents=True);model.save(path/'policy.zip');normalizer.save(path/'normalizer.pkl')
    dump(path/'bundle.json',{'algorithm':algorithm,'schema':schema,'seed':seed,'timesteps':model.num_timesteps,
      'phase':'after_optimizer_update','twin_id':twin.identity,'training_workload':'netflix',
      'forecast_sha256':ledger.digest,'protocol_sha256':sha256(BASE/'protocol.json'),'contract':asdict(contract),
      'features':feature_names(schema),'normalizer_count':float(normalizer.obs_rms.count),'norm_reward':False,
      'artifacts':{n:sha256(path/n) for n in ['policy.zip','normalizer.pkl']}})


def load_bundle(path,algorithm,schema,twin,contract):
    meta=json.loads((path/'bundle.json').read_text())
    if meta['algorithm']!=algorithm or meta['schema']!=schema or meta['features']!=feature_names(schema):raise ValueError('Policy schema/architecture mismatch')
    if meta['protocol_sha256']!=sha256(BASE/'protocol.json') or meta['twin_id']!=twin.identity or meta['contract']!=asdict(contract):raise ValueError('Policy scientific identity mismatch')
    ledger=Ledger(ROOT/CFG['forecast_ledgers']['netflix'])
    if meta['forecast_sha256']!=ledger.digest or meta['training_workload']!='netflix':raise ValueError('Policy training provenance mismatch')
    for name,h in meta['artifacts'].items():
        if sha256(path/name)!=h:raise ValueError('Policy/normalizer artifact changed')
    _,val,_=horizons('netflix')
    dummy=DummyVecEnv([lambda:ControlEnv(val,twin,ledger,contract,schema)])
    normalizer=VecNormalize.load(path/'normalizer.pkl',dummy);normalizer.training=False;normalizer.norm_reward=False
    cls=RecurrentPPO if algorithm=='recurrent_ppo' else PPO
    model=cls.load(path/'policy.zip',device='cpu')
    return PolicyController(model,normalizer,algorithm=='recurrent_ppo',schema)


class Callback(PriorCallback):
    def __init__(self,path,algorithm,schema,seed,val,twin,ledger,contract):
        if val.partition.timeline.name!='netflix' or val.partition.role!='validation':raise ValueError('Checkpoint selection accepts Netflix validation only')
        super().__init__(path,algorithm,seed,val,twin,ledger,contract);self.schema=schema
    @preserve_rng
    def capture(self):
        if self.num_timesteps in [s['timesteps'] for s in self.scores]:return
        path=self.directory/f'checkpoint_{self.num_timesteps}'
        save_bundle(self.model,self.training_env,path,self.algorithm,self.schema,self.seed,self.twin,self.ledger,self.contract)
        controller=load_bundle(path,self.algorithm,self.schema,self.twin,self.contract)
        before=pickle.dumps(controller.normalizer.obs_rms)
        r=run_controller(self.val,controller,self.twin,self.ledger,self.contract,path.name,sha256(path/'bundle.json'),self.schema)
        if pickle.dumps(controller.normalizer.obs_rms)!=before:raise AssertionError('Validation changed normalizer')
        score={'timesteps':self.num_timesteps,'checkpoint':str(path.relative_to(ROOT)),
           'total_reward':r.summary['total_reward'],'metrics':summarize(r),'normalizer_updated_by_validation':False}
        dump(path/'validation.json',score);self.scores.append(score);controller.normalizer.close()
        print(self.algorithm,self.schema,self.seed,self.num_timesteps,'Wh',score['metrics']['energy_wh'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--algorithm',choices=CFG['algorithms'],required=True)
    parser.add_argument('--schema',choices=CFG['schemas'],required=True);parser.add_argument('--seed',type=int,choices=CFG['seeds'],required=True)
    args=parser.parse_args();algorithm,schema,seed=args.algorithm,args.schema,args.seed
    verify_inputs();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    train,val,contract=horizons('netflix');ledger=Ledger(ROOT/CFG['forecast_ledgers']['netflix']);twin=CachedTwin()
    path=directory(algorithm,schema,seed);path.mkdir(parents=True,exist_ok=False);record=path/'run_ledger.jsonl'
    start=time.perf_counter();cpu_start=time.process_time();rss_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    append(record,{'status':'started','algorithm':algorithm,'schema':schema,'seed':seed,'protocol_sha256':sha256(BASE/'protocol.json')})
    try:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        def make_env(i):return lambda:Monitor(ControlEnv(train,twin,ledger,contract,schema),str(path/f'environment_{i}.monitor.csv'))
        norm=VecNormalize(DummyVecEnv([make_env(i) for i in range(4)]),norm_obs=True,norm_reward=False,clip_obs=10.,gamma=CFG['gamma'])
        cls=ExactPPO if algorithm=='ppo' else ExactRecurrentPPO
        kwargs={k:CFG[k] for k in ['n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','ent_coef','clip_range','target_kl']}
        pk={'lstm_hidden_size':128,'n_lstm_layers':1,'shared_lstm':True,'enable_critic_lstm':False} if algorithm=='recurrent_ppo' else {}
        model=cls('MlpPolicy' if algorithm=='ppo' else 'MlpLstmPolicy',norm,seed=seed,device='cpu',verbose=0,policy_kwargs=pk,**kwargs)
        model.stage4_updates=[];model.stage4_training_log=str(path/'training_updates.jsonl');model.set_logger(configure(str(path/'sb3_logs'),['csv']))
        callback=Callback(path,algorithm,schema,seed,val,twin,ledger,contract)
        model.learn(total_timesteps=300000,callback=callback);callback.capture()
        training_wall=time.perf_counter()-start;training_cpu=time.process_time()-cpu_start
        assert model.num_timesteps==300000 and [s['timesteps'] for s in callback.scores]==CFG['checkpoint_timesteps']
        dump(path/'training_remainder.json',{'partial_episodes_not_independent':callback.running})
        best=min(callback.scores,key=lambda s:(-s['total_reward'],s['timesteps']))
        dump(path/'selection.json',{'selected_checkpoint':best['checkpoint'],'validation_scores':callback.scores,
          'selection_objective':'netflix_validation_total_reward','selection_workload':'netflix','youtube_used':False,'final_test_used':False,'actual_timesteps':300000})
        with isolated_rng():
            selected=ROOT/best['checkpoint'];controller=load_bundle(selected,algorithm,schema,twin,contract)
            name=run_name(algorithm,schema,seed);before=pickle.dumps(controller.normalizer.obs_rms)
            result=run_controller(val,controller,twin,ledger,contract,name,sha256(selected/'bundle.json'),schema)
            latency=list(controller.latency_ns);state_norms=list(controller.state_norms);calls=controller.calls
            repeated=run_controller(val,controller,twin,ledger,contract,name,sha256(selected/'bundle.json'),schema)
            assert result==repeated and pickle.dumps(controller.normalizer.obs_rms)==before
            steps=OUT/'netflix';steps.mkdir(exist_ok=True);export_steps(name,result,steps/f'{name}_selected_steps.csv',ledger)
            parameter_bytes=sum(p.numel()*p.element_size() for p in controller.model.policy.parameters())
            controller.normalizer.close()
        metrics={**summarize(result),'selected_checkpoint':best['checkpoint'],'selected_timesteps':best['timesteps'],'actual_training_timesteps':300000,
          'schema':schema,'algorithm':algorithm,'seed':seed,'deterministic_repeat_equal':True,'recurrent_calls':calls if algorithm=='recurrent_ppo' else 0,
          'recurrent_state_norm_min':min(state_norms) if state_norms else None,
          'training_wall_seconds_including_checkpoint_validation':training_wall,'training_cpu_seconds_including_checkpoint_validation':training_cpu,
          'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,'pre_model_process_peak_rss_mib':rss_before/1024.,
          'policy_parameter_bytes':parameter_bytes,'inference_median_ms':float(np.median(latency)/1e6),'inference_p95_ms':float(np.percentile(latency,95)/1e6),
          'inference_calls_timed':len(latency),'optimizer_updates':len(model.stage4_updates)}
        dump(path/'result.json',metrics);norm.close();append(record,{'status':'completed','seconds':time.perf_counter()-start,'timesteps':300000,'metrics':metrics})
    except Exception as exc:
        append(record,{'status':'failed','seconds':time.perf_counter()-start,'error':repr(exc),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()

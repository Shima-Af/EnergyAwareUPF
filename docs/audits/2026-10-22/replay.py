import sys,json,copy,contextlib,io
from pathlib import Path
sys.path.insert(0,'/home/ubuntu/EnergyAwareUPF')
from src import tf_config,utils
from src.environment import ManualCooldownEnv
import numpy as np,pandas as pd,yaml,torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
torch.set_num_threads(1)
root=Path('/home/ubuntu/EnergyAwareUPF');out=Path('/tmp/upf-scientific-audit'); results={}
cfg=yaml.safe_load((root/'experiments/results/B1_top2_seeds/mlp_hybrid/seed_42/config.yaml').read_text())
def summary(rows):
 d=pd.DataFrame(rows)
 return dict(n=len(d),energy_wh=float(d.power.sum()*.25),sec=float(d.sec.mean()),violation=float((d.performance<.9).mean()),switches=int((d.type_switch_penalty>0).sum()),scales=int((d.scale_penalty>0).sum()),blocked=int(d.cooldown_blocked.sum()))
for trace,csv in [('main','processed_traffic_normalized.csv'),('youtube','processed_traffic_youtube_with_forecast.csv')]:
 c=copy.deepcopy(cfg);c['paths']['traffic_data_csv']=str(root/'data/processed'/csv)
 pre=utils.load_and_preprocess_data(c);results[trace]={'train_len':len(pre['train']['traffic_data']),'test_len':len(pre['test']['traffic_data'])}
 def envmaker(c=c,pre=pre):return ManualCooldownEnv(**pre['test'],env_config=c['environment'],reward_config=c['reward'])
 for action,name in [(0,'static_dpdk'),(1,'static_usr1')]:
  e=envmaker();e.reset();rows=[]
  while True:
   _,_,done,_,info=e.step(action);rows.append(info)
   if done:break
  results[trace][name]=summary(rows)
  e.close()
 for kind,cls in [('mlp_hybrid',PPO),('lstm_history',RecurrentPPO)]:
  for seed in [42,123,456,789,1024]:
   run=root/f'experiments/results/B1_top2_seeds/{kind}/seed_{seed}'
   c['environment']['observation_schema']='hybrid' if kind=='mlp_hybrid' else 'history'
   v=VecNormalize.load(str(run/'vec_normalize_stats.pkl'),DummyVecEnv([envmaker]));v.training=False;v.norm_reward=False
   model=cls.load(str(run/'best_model.zip'),env=v,device='cpu')
   is_rec=getattr(model.policy,'is_recurrent',False)
   obs=v.reset();state=None;starts=np.ones((1,),dtype=bool);rows=[]
   while True:
    if is_rec:act,state=model.predict(obs,state=state,episode_start=starts,deterministic=True)
    else:act,_=model.predict(obs,deterministic=True)
    obs,_,done,infos=v.step(act);rows.append(infos[0]);starts=np.asarray(done,dtype=bool)
    if done[0]:break
   label=f'{kind}_{seed}';results[trace][label]=summary(rows);results[trace][label]['evaluator_recurrent_flag']=bool(is_rec)
   saved=root/f'polished_results/PPO/steps/steps_B1_{kind}_seed_{seed}_{trace}.csv'
   old=pd.read_csv(saved);new=pd.DataFrame(rows)
   results[trace][label]['archived_energy_sum']=float(old.energy_wh.sum())
   results[trace][label]['max_power_delta']=float(np.max(np.abs(new.power.to_numpy()-old.power_w.to_numpy())))
   results[trace][label]['action_mismatches']=int(np.sum(new.executed_upf.to_numpy()!=old.upf_config.to_numpy()))
   if seed==42 and kind=='mlp_hybrid':new.to_csv(out/f'replay_{trace}_seed42.csv',index=False)
   print('AUDIT',trace,label,results[trace][label],flush=True);v.close()
 (out/'replay_results.json').write_text(json.dumps(results,indent=2))
print('AUDIT_STATIC',json.dumps({k:{n:v[n] for n in ['train_len','test_len','static_dpdk','static_usr1']} for k,v in results.items()},indent=2))

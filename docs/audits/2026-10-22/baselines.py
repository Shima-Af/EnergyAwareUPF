import sys,ast,json
from pathlib import Path
sys.path.insert(0,'/home/ubuntu/EnergyAwareUPF')
from src import tf_config,utils
from src.offline_optimal import compute_offline_optimal
from scripts.baseline_rule import dpdk_predict,oai_predict
import numpy as np,pandas as pd,yaml
cfg=yaml.safe_load(Path('experiments/results/B1_top2_seeds/mlp_hybrid/seed_42/config.yaml').read_text())
# Evaluate unchanged, inspected pure function definitions; omit module's file-writing top-level runner.
p=Path('experiments/results/paper_results_assets/compile_baselines_table.py');tree=ast.parse(p.read_text());defs=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
ns={'np':np,'pd':pd,'dpdk_predict':dpdk_predict,'oai_predict':oai_predict};exec(compile(defs,str(p),'exec'),ns)
results={}
for trace,csv in [('main','processed_traffic_normalized.csv'),('youtube','processed_traffic_youtube_with_forecast.csv')]:
 cfg['paths']['traffic_data_csv']='data/processed/'+csv;pre=utils.load_and_preprocess_data(cfg);e=cfg['environment'];v=pre['test'];d={'dpdk_lookup':v['dpdk_lookup'],'oai_lookup':v['oai_lookup'],'interval_h':.25,'qos_threshold':.9,'max_power':e['max_power'],'cooldown_steps':e['cooldown_period']}
 results[trace]={}
 for start,label in [(95,'compiler_721'),(96,'archive_720_reconstruction')]:
  d['Ts_act']=np.asarray(v['traffic_data'],dtype=np.float64)[start:];d['Ts_exp']=np.asarray(v['forecast_data'],dtype=np.float64)[start:]
  results[trace][label]={}
  for name,fn,args in [('rule',ns['rule_based_metrics'],{}),('dpdk',ns['static_metrics'],{}),('usr1',ns['static_metrics'],{'upf':'oai','n':1})]:
   s,df=fn(d,**args);results[trace][label][name]=s
 optimal=compute_offline_optimal(v,e,cfg['reward'],objective_mode='discounted',gamma=cfg['agent']['gamma']);results[trace]['offline']=optimal.summary
 print('BASELINE',trace,results[trace],flush=True)
Path('/tmp/upf-scientific-audit/baseline_results.json').write_text(json.dumps(results,indent=2))

import sys,json
from pathlib import Path
sys.path.insert(0,'/home/ubuntu/EnergyAwareUPF')
from src import tf_config
from scripts.compute_profiling_model_accuracy import evaluate_mode
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
out={}
for mode,name in [('DPDK','dpdk'),('USR','oai')]:
 path=Path('/home/ubuntu/DDT_UPF_Selection_DRL/data/processed')/(name+'_processed.csv')
 metrics,rows=evaluate_mode(mode,name,path,Path('saved_models/prediction_models'),'Objective Throughput (DL/UL)','power_consumption_watts','QoS_w',{'test_size':.2,'random_state':42,'shuffle':True},.001)
 q=pd.DataFrame(rows);q=q[q.target=='QoS score'];df=pd.read_csv(path);tr,te=train_test_split(df,test_size=.2,random_state=42,shuffle=True)
 y=q.y_true.to_numpy();p=q.y_pred.to_numpy();near=abs(y-.9)<=.05
 detail=dict(actual_below=int(sum(y<.9)),predicted_below=int(sum(p<.9)),false_safe=int(sum((y<.9)&(p>=.9))),false_unsafe=int(sum((y>=.9)&(p<.9))),near_threshold_n=int(sum(near)),near_threshold_rmse=float(np.sqrt(np.mean((y[near]-p[near])**2))),train_mean_baseline_rmse=float(np.sqrt(np.mean((y-tr.QoS_w.mean())**2))),min_pred=float(p.min()),max_pred=float(p.max()))
 out[mode]={'metrics':metrics,'threshold_diagnostic':detail};print(mode,out[mode],flush=True)
Path('/tmp/upf-scientific-audit/residual_results.json').write_text(json.dumps(out,indent=2))

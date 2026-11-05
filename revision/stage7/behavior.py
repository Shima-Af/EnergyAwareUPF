"""Descriptive adaptation strata from saved trajectories, no new policy rollout."""
import json
import numpy as np
import pandas as pd
from revision.evaluation.validate_static import write_csv
from .common import OUT,horizons


def main():
    c=json.loads((OUT/'comparison.json').read_text())['workloads'];rows=[]
    for workload,scope in c.items():
        _,horizon,_=horizons(workload);loads=np.asarray(horizon.partition.timeline.load_mbps)
        for name in scope['controllers_selected']:
            d=pd.read_csv(OUT/workload/f'{name}_selected_steps.csv');previous=loads[d.source_index.to_numpy(int)-1]
            increment=d.actual_load_mbps.to_numpy()-previous
            groups=[('all',np.ones(len(d),bool)),('rising',increment>0),('falling',increment<0)]
            groups += [(f'load_{lo}_{hi}',(d.actual_load_mbps.to_numpy()>=lo)&(d.actual_load_mbps.to_numpy()<hi)) for lo,hi in [(0,50),(50,100),(100,200),(200,400),(400,float('inf'))]]
            for label,mask in groups:
                g=d[mask]
                if not len(g):continue
                rows.append({'workload':workload,'controller':name,'stratum':label,'slots':len(g),
                  'energy_wh':float(g.energy_wh.sum()),'mean_sec_j_per_mbit':float(g.sec_j_per_mbit.mean()),
                  'mean_predicted_qos':float(g.qos.mean()),'type_switches':int(g.type_switch.sum()),
                  'scale_up':int(g.scale_up.sum()),'scale_down':int(g.scale_down.sum()),'blocked':int(g.blocked.sum()),
                  'dpdk_slots':int(sum(g.executed==0)),'usr1_slots':int(sum(g.executed==1)),'usr2_slots':int(sum(g.executed==2))})
    write_csv(OUT/'adaptation_by_load_and_direction.csv',rows)
    print('Descriptive adaptation strata written; no action optimization or new rollout',flush=True)

if __name__=='__main__':main()

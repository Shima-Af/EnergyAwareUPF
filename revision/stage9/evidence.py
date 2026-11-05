"""Additional observed sweep variability and unchanged-model fit diagnostics."""
import json
import numpy as np
import pandas as pd
from revision.evaluation.twin import ROOT,sha256
from revision.evaluation.revised_twin import RevisionTwin
from .run import CFG,OUT,csv_write
from revision.forecast_service.core import data
from .core import segments


def boundary_joins():
    """Record joins from pre-horizon context as well as within-horizon flags."""
    rows=[]
    for w in CFG['workloads']:
        times,loads,_=data(w)
        for name,spec in CFG['scenarios'].items():
            for role,d in segments(loads,times,name,spec).items():
                i=int(d['indices'][0]);new=float(d['factors'][0])
                old=(spec['factors'][((i-1)//spec['block_slots'])%len(spec['factors'])]
                     if spec['kind']=='mixed' else new)
                rows.append(dict(workload=w,scenario=name,role=role,source_index=i,
                    previous_context_timestamp=times[i-1].isoformat(),timestamp=times[i].isoformat(),
                    previous_context_factor=old,current_factor=new,artificial_factor_join=old!=new,
                    original_ramp_mbps=float(loads[i]-loads[i-1]),
                    transformed_ramp_mbps=float(new*loads[i]-old*loads[i-1]),
                    natural_component_mbps=float(old*(loads[i]-loads[i-1])),
                    imposed_component_mbps=float((new-old)*loads[i])))
    csv_write(OUT/'context_boundary_joins.csv',rows)


def main():
    twin=RevisionTwin();rows=[]
    for mode in ['dpdk','usr']:
        spec=CFG['profiling_data'][mode];path=ROOT/CFG['profiling_data']['directory']/spec['file']
        assert sha256(path)==spec['sha256']
        data=pd.read_csv(path)
        for (sweep,load),g in data.groupby(['Test Set','Objective Throughput (DL/UL)']):
            q=float(twin.components[mode+'_qos'].predict([load])[0])
            p=float(twin.components[mode+'_power'].predict([load])[0])
            y=g.QoS_w.to_numpy();py=g.power_consumption_watts.to_numpy()
            safe=q>=.9;failed=y<.9
            rows.append(dict(mode=mode,sweep=int(sweep),load_mbps=float(load),n=len(y),
               measured_qos_mean=float(y.mean()),measured_qos_min=float(y.min()),measured_qos_max=float(y.max()),
               measured_unsafe=int(failed.sum()),unsafe_rate=float(failed.mean()),predicted_qos=q,
               false_safe=int(failed.sum()) if safe else 0,false_unsafe=int((~failed).sum()) if not safe else 0,
               measured_mean_shortfall=float(np.maximum(0,.9-y).mean()),measured_max_shortfall=float(np.maximum(0,.9-y).max()),
               qos_rmse=float(np.sqrt(np.mean((q-y)**2))),
               measured_power_mean_w=float(py.mean()),predicted_power_w=p,
               power_rmse_w=float(np.sqrt(np.mean((p-py)**2)))))
    csv_write(OUT/'measurement_by_load_sweep.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(rows);fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,mode in zip(axes,['dpdk','usr']):
        sub=frame[frame['mode']==mode]
        for sweep,g in sub.groupby('sweep'):
            ax.plot(g.load_mbps,g.unsafe_rate,'o-',ms=3,alpha=.65,label=f'sweep {sweep}')
        ax.set_xscale('log');ax.set_ylim(-.02,1.02);ax.set_title(mode.upper()+' observed Q<.90')
        ax.set_xlabel('Measured per-instance offered load (Mbps)');ax.set_ylabel('Measured failure fraction');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'figures/measured_failure_rates.{ext}',dpi=160)
    plt.close(fig);twin.verify_artifacts();boundary_joins()


if __name__=='__main__':main()

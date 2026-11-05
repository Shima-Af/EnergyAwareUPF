"""Workload-only inference and plots. Run in a fresh output namespace."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

from revision.evaluation.twin import ROOT, sha256
from revision.evaluation.revised_twin import RevisionTwin
from revision.forecast_service.core import data, data_hash
from revision.stage3.common import dump
from .core import segments, qos_roots, distribution

BASE=ROOT/'revision/stage9'; OUT=ROOT/'revision/reports/stage9'
CFG=json.loads((BASE/'protocol.json').read_text())


def csv_write(path, rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def bulk(twin, loads):
    """Same frozen transformations and networks; diagnostic batch profile explicit."""
    loads=np.asarray(loads,float)
    if not np.isfinite(loads).all() or loads.min()<.02 or loads.max()>5000:
        raise ValueError('Scenario outside common interpolation domain; no clipping')
    q=[];p=[]
    for a in range(3):
        mode='dpdk' if a==0 else 'usr';k=2 if a==2 else 1
        q.append(twin.components[mode+'_qos'].predict(loads/k))
        p.append(k*twin.components[mode+'_power'].predict(loads/k))
    return np.array(p).T,np.array(q).T


def main():
    OUT.mkdir(exist_ok=True)
    dump(OUT/'protocol_frozen.json',dict(protocol=CFG,protocol_sha256=sha256(BASE/'protocol.json'),
         implementation_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [BASE/'core.py',BASE/'run.py']},
         no_scenario_outputs_seen=True))
    prepared={};source={}
    # Freeze all scaling/window choices before any new twin inference.
    for w in CFG['workloads']:
        times,loads,scale=data(w)
        source[w]=dict(prefix_sha256=data_hash(loads,times,2520),scale=scale)
        prepared[w]={name:segments(loads,times,name,spec) for name,spec in CFG['scenarios'].items()}
    selections={w:{n:{r:dict(source_indices=d['indices'].tolist(),factors=d['factors'].tolist(),
                   timestamps=d['timestamps']) for r,d in parts.items()} for n,parts in scenarios.items()}
                for w,scenarios in prepared.items()}
    dump(OUT/'scenario_definitions_frozen.json',dict(sources=source,scenarios=selections,
          selection_uses_validation_values=False,derived_before_twin_inference=True))
    twin=RevisionTwin();twin.verify_artifacts()
    dump(OUT/'twin_manifest.json',twin.manifest)
    # Recovered profiling evidence is read only and hash authenticated.
    measurement={};measured=[];validity=[]
    for mode,spec in CFG['profiling_data'].items():
        if mode=='directory':continue
        path=Path(CFG['profiling_data']['directory'])/spec['file']
        assert sha256(path)==spec['sha256']
        frame=pd.read_csv(path);measurement[mode]=frame
        for load,g in frame.groupby('Objective Throughput (DL/UL)'):
            y=g.QoS_w.to_numpy();prediction=float(twin.components[mode+'_qos'].predict([load])[0])
            measured.append(dict(mode=mode,per_instance_mbps=float(load),n=len(g),
                 predicted_qos=prediction,measured_unsafe=int((y<.9).sum()),
                 unsafe_rate=float((y<.9).mean()),mean_qos=float(y.mean()),
                 q05=float(np.quantile(y,.05)),q25=float(np.quantile(y,.25)),
                 q50=float(np.quantile(y,.5)),q75=float(np.quantile(y,.75)),q95=float(np.quantile(y,.95)),
                 min_qos=float(y.min()),max_qos=float(y.max())))
        for sweep,g in frame.groupby('Test Set'):
            for region,mask in [('low_le50',g['Objective Throughput (DL/UL)']<=50),
                                 ('study_le400',g['Objective Throughput (DL/UL)']<=400),
                                 ('above400',g['Objective Throughput (DL/UL)']>400)]:
                sub=g[mask];y=sub.QoS_w.to_numpy()
                pred=twin.components[mode+'_qos'].predict(sub['Objective Throughput (DL/UL)'].to_numpy())
                e=pred-y;safe=pred>=.9
                validity.append(dict(mode=mode,sweep=int(sweep),region=region,n=len(sub),
                    measured_unsafe=int((y<.9).sum()),predicted_safe=int(safe.sum()),
                    false_safe=int((safe&(y<.9)).sum()),false_unsafe=int((~safe&(y>=.9)).sum()),
                    rmse=float(np.sqrt(np.mean(e**2))),mae=float(abs(e).mean()),
                    max_false_safe_shortfall=float(np.max(.9-y[safe&(y<.9)])) if (safe&(y<.9)).any() else None))
    csv_write(OUT/'measurement_by_load.csv',measured);csv_write(OUT/'measurement_by_sweep_region.csv',validity)
    roots={};pieces={};brackets={};geometry_checks={}
    for mode in ['dpdk','usr']:
        model=twin.components[mode+'_qos'];roots[mode],pieces[mode]=qos_roots(model)
        brackets[mode]=[]
        for root in roots[mode]:
            l=max(.01,root*.999);h=min(5000.,root*1.001)
            ql=float(model.predict([l])[0]);qh=float(model.predict([h])[0])
            assert (ql>=.9)!=(qh>=.9)
            for _ in range(35):
                mid=(l+h)/2;q=float(model.predict([mid])[0])
                if (q>=.9)==(ql>=.9):l=mid
                else:h=mid
            brackets[mode].append(dict(analytic_root_mbps=root,float32_transition_bracket_mbps=[l,h],
                                       lower_safe=ql>=.9,upper_safe=qh>=.9))
        # Check analytic structural map against actual frozen-network inference.
        z=np.array([(l+h)/2 for l,h,_,_ in pieces[mode]])
        x=np.expm1(z*model.metadata['input_scaler']['scale']+model.metadata['input_scaler']['mean'])
        x=np.minimum(5000,np.maximum(.01,x))  # Geometry anchors, never workload repair.
        expected=np.array([expit((m*zz+b)[0]) for zz,(_,_,m,b) in zip(z,pieces[mode])])
        actual=model.predict(x)
        geometry_checks[mode]=dict(pieces=len(pieces[mode]),max_abs_difference=float(abs(expected-actual).max()))
        assert abs(expected-actual).max()<2e-6
    edges=sorted([.02,5000.]+[r*k for mode,k in [('dpdk',1),('usr',1),('usr',2)]
                               for r in roots[mode] if .02<r*k<5000])
    regions=[]
    for l,h in zip(edges,edges[1:]):
        _,q=bulk(twin,[np.sqrt(l*h)]);safe=(q[0]>=.9)
        regions.append(dict(lower_mbps=l,upper_mbps=h,safe_actions=np.flatnonzero(safe).tolist(),
              risk_region='all_safe' if safe.all() else 'all_unsafe' if not safe.any() else 'avoidable'))
    dump(OUT/'qos_regions.json',dict(common_total_domain_mbps=[.02,5000],per_instance_domain_mbps=[.01,5000],
          note='Analytic real-coefficient intervals; float32 transition brackets supplied, not physical confidence bounds. Endpoint Q=.90 is safe.',
          roots=roots,transition_brackets=brackets,structural_checks=geometry_checks,regions=regions))
    # Diagnostic grid is not a generated traffic workload.
    grid=np.unique(np.r_[np.geomspace(.02,5000,2500),np.linspace(.02,50,1000),np.linspace(50,5000,2000),edges])
    gp,gq=bulk(twin,grid)
    csv_write(OUT/'qos_load_grid.csv',[dict(total_load_mbps=float(x),**{f'qos_{a}':float(gq[i,a]) for a in range(3)},
                 **{f'power_w_{a}':float(gp[i,a]) for a in range(3)}) for i,x in enumerate(grid)])
    rows=[];scenario_metrics=[];action_metrics=[];inference_checks=[]
    for w,scenarios in prepared.items():
        for name,parts in scenarios.items():
            for role,d in parts.items():
                x=d['loads'];p,q=bulk(twin,x);short=np.maximum(0,.9-q);bad=q<.9
                # Verify singleton profile, including every near-threshold value.
                check=np.unique(np.r_[np.linspace(0,len(x)-1,min(12,len(x)),dtype=int),
                                     np.flatnonzero(np.min(abs(q-.9),axis=1)<1e-5)])
                maxp=maxq=0.;different=0
                for idx in check:
                    for a in range(3):
                        o=twin.outcome(a,float(x[idx]));maxp=max(maxp,abs(o.power_w-p[idx,a]));maxq=max(maxq,abs(o.qos-q[idx,a]))
                        different+=int((o.qos>=.9)!=(q[idx,a]>=.9))
                inference_checks.append(dict(workload=w,scenario=name,role=role,checked_loads=len(check),
                     max_power_difference_w=maxp,max_qos_difference=maxq,class_disagreements=different))
                if different:raise ValueError('Diagnostic profile changes threshold classes; investigate without repair')
                risks=np.where(bad.all(axis=1),'all_unsafe',np.where(bad.any(axis=1),'avoidable','all_safe'))
                delta=np.diff(x);natural=delta[~d['joins'][1:]];join=delta[d['joins'][1:]]
                cheapest=p.argmin(axis=1);energy_term=100*p/np.maximum(x[:,None],1e-6)
                penalty=30*short;immediate=(energy_term+penalty).argmin(axis=1)
                feasible_energy=np.where(~bad,p,np.inf).min(axis=1)
                min_energy=p.min(axis=1);avoid=risks=='avoidable'
                metric=dict(workload=w,scenario=name,role=role,n=len(x),source_start=int(d['indices'][0]),source_stop_exclusive=int(d['indices'][-1])+1,
                   load_distribution=distribution(x),absolute_ramp_mbps_per_slot=distribution(abs(delta)),
                   natural_ramp_count=len(natural),natural_max_abs_ramp=float(abs(natural).max()) if len(natural) else None,
                   artificial_joins=int(d['joins'].sum()),artificial_join_max_abs_ramp=float(abs(join).max()) if len(join) else None,
                   risk_counts={r:int((risks==r).sum()) for r in ['all_safe','avoidable','all_unsafe']},
                   any_unsafe_rate=float(bad.any(axis=1).mean()),avoidable_rate=float(avoid.mean()),all_unsafe_rate=float(bad.all(axis=1).mean()),
                   mean_action_qos_spread=float(np.ptp(q,axis=1).mean()),max_action_qos_spread=float(np.ptp(q,axis=1).max()),
                   usr2_protects_usr1_count=int((bad[:,1]&~bad[:,2]).sum()),usr1_protects_usr2_count=int((bad[:,2]&~bad[:,1]).sum()),
                   cheapest_action_unsafe_count=int(bad[np.arange(len(x)),cheapest].sum()),
                   avoidable_cheapest_unsafe_count=int((bad[np.arange(len(x)),cheapest]&avoid).sum()),
                   qos_term_changes_immediate_ranking_count=int((cheapest!=immediate).sum()),
                   safe_min_power_premium_wh_on_avoidable=float(.25*(feasible_energy[avoid]-min_energy[avoid]).sum()),
                   above400_count=int((x>400).sum()),above400_per_instance_counts=[int((x/(2 if a==2 else 1)>400).sum()) for a in range(3)])
                scenario_metrics.append(metric)
                for a in range(3):
                    conditional=short[bad[:,a],a]
                    action_metrics.append(dict(workload=w,scenario=name,role=role,action=a,n=len(x),
                         violations=int(bad[:,a].sum()),violation_rate=float(bad[:,a].mean()),
                         mean_shortfall=float(short[:,a].mean()),max_shortfall=float(short[:,a].max()),
                         conditional_shortfall=distribution(conditional) if len(conditional) else None,
                         energy_wh=float(np.sum(p[:,a]*.25)),mean_power_w=float(p[:,a].mean()),
                         mean_qos_penalty=float(penalty[:,a].mean()),sum_qos_penalty=float(penalty[:,a].sum()),
                         mean_energy_reward_term=float(energy_term[:,a].mean()),
                         violation_load_min=float(x[bad[:,a]].min()) if len(conditional) else None,
                         violation_load_max=float(x[bad[:,a]].max()) if len(conditional) else None))
                for i,index in enumerate(d['indices']):
                    rows.append(dict(workload=w,scenario=name,role=role,source_index=int(index),source_timestamp=d['timestamps'][i],
                        scenario_timestamp=d['timestamps'][i],original_load_mbps=float(d['original'][i]),factor=float(d['factors'][i]),
                        load_mbps=float(x[i]),artificial_factor_join=bool(d['joins'][i]),risk_region=risks[i],
                        **{f'power_{a}_w':float(p[i,a]) for a in range(3)},**{f'qos_{a}':float(q[i,a]) for a in range(3)},
                        **{f'qos_penalty_{a}':float(penalty[i,a]) for a in range(3)}))
                print(w,name,role,len(x),metric['risk_counts'],flush=True)
    csv_write(OUT/'scenario_samples.csv',rows)
    dump(OUT/'scenario_metrics.json',scenario_metrics);dump(OUT/'action_metrics.json',action_metrics)
    csv_write(OUT/'action_metrics.csv',[{k:(json.dumps(v) if isinstance(v,dict) else v) for k,v in m.items()} for m in action_metrics])
    dump(OUT/'inference_profile_checks.json',inference_checks)
    twin.verify_artifacts()
    plot(grid,gq,measured,rows)
    dump(OUT/'summary.json',dict(twin_id=twin.identity,scenarios=len(scenario_metrics),scored_source_rows=len(rows),
         independent_measurement_campaign=False,model_refits=0,controller_evaluations=0,final_test_used=False,
         maximum_checked_qos_profile_difference=max(r['max_qos_difference'] for r in inference_checks),
         maximum_checked_power_profile_difference_w=max(r['max_power_difference_w'] for r in inference_checks),
         total_checked_loads=sum(r['checked_loads'] for r in inference_checks),
         recommendation='Conditional workload stress diagnostics only; no operational safety or automatic RL authorization'))


def plot(grid,q,measured,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.joinpath('figures').mkdir(exist_ok=True)
    names=['DPDK','USR(1)','USR(2)'];colors=['tab:blue','tab:orange','tab:green']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax in axes:
        for a in range(3):
            ax.plot(grid,q[:,a],label=names[a]+' mean model',color=colors[a])
            mode='dpdk' if a==0 else 'usr';k=2 if a==2 else 1
            subset=[r for r in measured if r['mode']==mode and r['per_instance_mbps']*k<=5000]
            xx=np.array([r['per_instance_mbps']*k for r in subset]);yy=np.array([r['q50'] for r in subset])
            ax.errorbar(xx,yy,yerr=[yy-np.array([r['q05'] for r in subset]),np.array([r['q95'] for r in subset])-yy],
                        fmt='.',color=colors[a],alpha=.45,capsize=2,label=names[a]+' measured 5–95%' if ax is axes[0] else None)
        ax.axhline(.9,color='black',ls='--');ax.set_ylim(0,1.02);ax.set_ylabel('Composite QoS');ax.set_xlabel('Total study offered load (Mbps)');ax.grid(alpha=.2)
    axes[0].set_xscale('log');axes[0].set_xlim(.02,5000);axes[0].legend(fontsize=7,ncol=2)
    axes[0].set_title('Declared common interpolation range')
    axes[1].set_xlim(0,50);axes[1].set_title('Low loads; USR(2) measurement axis ×2 is an approximation')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'figures/qos_vs_load.{ext}',dpi=160)
    plt.close(fig)
    frame=pd.DataFrame(rows)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,w in enumerate(['netflix','youtube']):
        for role,ls in [('train','-'),('validation','--')]:
            for name in CFG['scenarios']:
                sub=frame[(frame.workload==w)&(frame.role==role)&(frame.scenario==name)]
                xs=np.sort(sub.load_mbps.to_numpy());ys=np.arange(1,len(xs)+1)/len(xs)
                axes[i,0].plot(xs,ys,ls,label=name+' '+role,alpha=.8,lw=1)
        axes[i,0].set_xscale('log');axes[i,0].set_xlabel('Study load Mbps');axes[i,0].set_ylabel('Empirical CDF');axes[i,0].set_title(w)
        for name in ['original','mixed_daily','rapid_real_day']:
            sub=frame[(frame.workload==w)&(frame.role=='train')&(frame.scenario==name)]
            axes[i,1].plot(pd.to_datetime(sub.source_timestamp),sub.load_mbps,label=name,lw=.8)
        axes[i,1].set_title(w+' training chronology');axes[i,1].set_ylabel('Study load Mbps');axes[i,1].legend(fontsize=7)
    axes[0,0].legend(fontsize=5,ncol=2);fig.autofmt_xdate();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'figures/traffic_distributions.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()

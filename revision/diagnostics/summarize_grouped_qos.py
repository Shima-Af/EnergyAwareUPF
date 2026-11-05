"""Post-selection summaries and plots only; no fitting, selection or traffic."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from revision.diagnostics.qos_reliability import LOAD,TARGET
from revision.profiling.refit_usr_power import write_json


def decompose_prediction_variance(frame):
    y=frame[TARGET].to_numpy();p=frame.predicted_qos.to_numpy()
    means=frame.groupby(LOAD)[TARGET].transform('mean').to_numpy()
    sst=float(np.sum((y-y.mean())**2));sse=float(np.sum((p-y)**2))
    within=float(np.sum((y-means)**2));fit=float(np.sum((p-means)**2))
    cross=float(2*np.sum((p-means)*(means-y)))
    if not np.isclose(sse,within+fit+cross,rtol=1e-12,atol=1e-12):raise ValueError('General residual identity failed')
    span=frame.groupby(LOAD).predicted_qos.agg(lambda v:float(v.max()-v.min())).max()
    return {'n':len(y),'target_sst':sst,'actual_model_sse':sse,'within_load_sse':within,
       'load_mean_misfit_sse':fit,'numerical_prediction_cross_term':cross,
       'within_load_fraction_target_variation':within/sst,
       'within_load_fraction_model_sse':within/sse,'in_sample_load_mean_r2_ceiling':1-within/sst,
       'max_same_load_prediction_span':float(span),
       'interpretation':'In-sample target-variability decomposition, not an independent fitted predictor. Cross term retains float32 batch-position effects; actual predictions are unchanged.'}


def main():
    root=Path(__file__).resolve().parents[2];out=root/'revision/reports/task2h'
    selected=json.loads((out/'selection_frozen.json').read_text())['modes']
    metrics=json.loads((out/'metrics.json').read_text())['modes']
    predictions=pd.read_csv(out/'predictions.csv')
    grid=pd.read_csv(out/'domain_predictions.csv')
    details={};variance={}
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.1),layout='constrained')
    for col,mode in enumerate(('DPDK','USR')):
        winner=selected[mode]['selected_name'];details[mode]={};variance[mode]={}
        for role in ('train','validation','internal_test'):
            p=predictions[(predictions['mode']==mode)&(predictions.model==winner)&(predictions.role==role)]
            frame=p.rename(columns={'load_mbps':LOAD,'measured_qos':TARGET,'sweep':'Test Set'})
            variance[mode][role]=decompose_prediction_variance(frame)
            if role=='train':continue
            v=metrics[mode][winner][role];constant=metrics[mode]['training_mean_constant'][role]
            old=predictions[(predictions['mode']==mode)&(predictions.model=='original_saved_nn_nonindependent')&(predictions.role==role)].set_index('source_row')
            paired=p.set_index('source_row')
            changed=(paired.predicted_qos>=.9)!=(old.loc[paired.index,'predicted_qos']>=.9)
            details[mode][role]={'rmse_reduction_relative_to_constant_percent':100*(1-v['rmse']/constant['rmse']),
                'mse_reduction_relative_to_constant_percent':100*(1-(v['rmse']/constant['rmse'])**2),
                'safety_classification_changes_from_original_saved_network':int(changed.sum()),
                'loads_with_changed_classification_mbps':sorted(float(x) for x in paired.loc[changed,'load_mbps'].unique()),
                'unexplained_fraction_target_variance':1-v['r2'],
                'no_approved_false_safe_risk_threshold':True}
        axis=axes[0,col]
        for name,color,style,label in [(winner,'tab:blue','-','Selected bounded NN'),
               ('training_load_mean_pwl','black','--','Training load-mean curve'),
               ('training_mean_constant','tab:green',':','Training constant')]:
            g=grid[(grid['mode']==mode)&(grid.model==name)]
            axis.plot(g.load_mbps,g.predicted_qos,color=color,linestyle=style,label=label)
        for role,marker in [('validation','s'),('internal_test','^')]:
            f=predictions[(predictions['mode']==mode)&(predictions.model==winner)&(predictions.role==role)]
            means=f.groupby('load_mbps').measured_qos.mean()
            axis.scatter(means.index,means.values,s=20,marker=marker,label=f'Measured {role.replace("_"," ")} mean')
        axis.axhline(.9,color='gray',linewidth=.8);axis.set_xscale('log');axis.set_ylim(0,1.03)
        axis.set_title(mode+' grouped profiling comparison')
        axis.set_xlabel('Recorded per-instance offered load (Mbps)');axis.set_ylabel('Composite QoS score')
        axis.legend(fontsize=7)
        f=predictions[(predictions['mode']==mode)&(predictions.model==winner)&(predictions.role=='internal_test')]
        axis=axes[1,col]
        axis.scatter(f.measured_qos,f.predicted_qos,s=10,alpha=.35,label='Internal test, sweep 5')
        bad=f[(f.predicted_qos>=.9)&(f.measured_qos<.9)]
        axis.scatter(bad.measured_qos,bad.predicted_qos,color='tab:red',s=17,label=f'False safe: {len(bad)}')
        axis.plot([0,1],[0,1],color='gray',linewidth=.8)
        axis.axhline(.9,color='black',linestyle='--',linewidth=.8);axis.axvline(.9,color='black',linestyle='--',linewidth=.8)
        axis.set_xlim(0,1.03);axis.set_ylim(0,1.03)
        axis.set_xlabel('Measured composite QoS_w');axis.set_ylabel('Predicted composite score')
        axis.legend(fontsize=7);axis.grid(alpha=.2)
    write_json(out/'selected_variance_decomposition.json',variance)
    write_json(out/'decision_diagnostics.json',details)
    for extension in ('png','pdf'):
        path=out/f'grouped_qos_comparison.{extension}'
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__=='__main__':main()

"""Plot fixed-model errors and empirical sweep ranges; no fitting."""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    folder=Path(__file__).resolve().parents[2]/'revision/reports/task2e'
    data=pd.read_csv(folder/'predictions.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for col,mode in enumerate(('dpdk','usr')):
        f=data[data['mode']==mode]
        axis=axes[0,col]
        groups=f.groupby(['sweep','offered_load_mbps']).measured_qos.mean().unstack(0)
        low,high=groups.min(axis=1),groups.max(axis=1)
        axis.fill_between(groups.index,low,high,color='gray',alpha=.25,label='Five sweep means: range, not CI')
        axis.plot(groups.index,groups.mean(axis=1),'o',markersize=3,label='Mean of sweep means')
        prediction=f.groupby('offered_load_mbps').predicted_qos.first()
        axis.plot(prediction.index,prediction.values,label='Original neural prediction')
        axis.axhline(.9,color='black',linestyle='--',linewidth=1)
        axis.set_xscale('log');axis.set_ylim(0,1.03)
        axis.set_xlabel('Recorded per-instance offered load (Mbps)')
        axis.set_ylabel('Stored QoS score / prediction')
        axis.set_title(mode.upper()+': descriptive, all sweeps historically exposed')
        axis.legend(fontsize=6.8)
        v=f[f.historical_role=='early_stopping_validation']
        axis=axes[1,col]
        axis.scatter(v.measured_qos,v.predicted_qos,s=8,alpha=.3,label='Historical early-stopping validation')
        fs=v[v.false_safe]
        axis.scatter(fs.measured_qos,fs.predicted_qos,s=16,color='tab:red',label=f'False safe: {len(fs)}')
        axis.plot([0,1],[0,1],color='gray',linewidth=.8)
        axis.axhline(.9,color='black',linestyle='--',linewidth=.8)
        axis.axvline(.9,color='black',linestyle='--',linewidth=.8)
        axis.set_xlim(0,1.03);axis.set_ylim(0,1.03)
        axis.set_xlabel('Measured stored QoS_w');axis.set_ylabel('Predicted QoS')
        axis.legend(fontsize=6.8)
        axis.grid(alpha=.2)
    for extension in ('png','pdf'):
        path=folder/f'qos_reliability.{extension}'
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__=='__main__':main()

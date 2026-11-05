"""Targeted surviving-notebook reproduction, with no surrogate/controller fitting."""

import argparse
import ast
import base64
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import io
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from revision.evaluation.twin import ROOT, sha256
from revision.evaluation.validate_static import write_csv
from revision.profiling.refit_usr_power import write_json

NOTEBOOK=ROOT/'revision/reference/03_QoS.ipynb'
DATA=Path('/home/ubuntu/DDT_UPF_Selection_DRL/data/processed')
KEYS=['timestamp','Test Set','Test Case','Objective Throughput (DL/UL)']


def notebook_functions(notebook):
    tree=ast.parse(''.join(notebook['cells'][4]['source']))
    if not all(isinstance(node,ast.FunctionDef) for node in tree.body):
        raise ValueError('Expected definitions only; do not execute arbitrary notebook cells')
    names={node.name for node in tree.body}
    if names!={'bounded_score_low_is_bad','throughput_fulfillment','compute_component_scores','weighted_qos','data_driven_qos'}:
        raise ValueError('Unexpected notebook definition block')
    namespace={'np':np,'StandardScaler':StandardScaler,'MinMaxScaler':MinMaxScaler,'PCA':PCA}
    exec(compile(tree,str(NOTEBOOK)+':cell4','exec'),namespace)
    return namespace


def differences(predicted, reference):
    p,y=np.asarray(predicted,dtype=float),np.asarray(reference,dtype=float)
    if p.shape!=y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Comparable finite rows are required')
    e=np.abs(p-y)
    return {'comparable_rows':len(y),'bitwise_equal_float_values':int(np.sum(p==y)),
            'matches_at_1e-10':int(np.sum(e<=1e-10)),'mae':float(e.mean()),
            'maximum_absolute_difference':float(e.max()),
            'tolerance_note':'Absolute serialization/numerical reconstruction tolerance, not scientific model accuracy'}


def components(frame, functions, direction='DL', throughput='DN_Rx_Mbits/s', loss_multiplier=100.):
    mapping={'DL':('downlink_delay_ms','RAN_jitter','RAN_PLR_interval'),
             'UL':('uplink_delay_ms','DN_jitter','DN_PLR_interval')}
    lat,jit,loss=mapping[direction]
    work=frame.copy()
    work['_loss_candidate']=frame[loss]*loss_multiplier
    S=functions['compute_component_scores'](work,delivered_col=throughput,
       objective_col='Objective Throughput (DL/UL)',delay_col=lat,jitter_col=jit,
       loss_frac_col='_loss_candidate')
    return S,{'delivered':throughput,'objective':'Objective Throughput (DL/UL)',
              'delay':lat,'jitter':jit,'loss':loss,'loss_multiplier':loss_multiplier,
              'caps':[40.,10.,5.],'weights':[.4,.2,.2,.2]}


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__();self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.row=[]
        if tag in ('td','th'):self.cell=''
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:
            self.row.append(self.cell.strip());self.cell=None
        if tag=='tr' and self.row is not None:self.rows.append(self.row);self.row=None


def displayed_head_checks(notebook,frame,output_index):
    html=''.join(notebook['cells'][1]['outputs'][output_index]['data']['text/html'])
    parser=TableParser();parser.feed(html)
    headers=parser.rows[0];comparisons=[]
    for row in parser.rows[1:]:
        index=int(row[0])
        for col,text in zip(headers[1:],row[1:]):
            if col=='...':continue
            actual=frame.loc[index,col]
            if isinstance(actual,str):match=actual==text
            else:
                decimals=len(text.split('.')[1]) if '.' in text else 0
                tolerance=.5*10**(-decimals)+1e-12
                match=abs(float(actual)-float(text))<=tolerance
            comparisons.append({'source_row':index,'column':col,'displayed':text,
                                'later_value':str(actual),'matches_display_precision':bool(match)})
    return comparisons


def source_history(output):
    repositories=[ROOT,Path('/home/ubuntu/xRL_UpfSelection'),Path('/home/ubuntu/DDT_UPF_Selection_DRL'),
                  Path('/home/ubuntu/dynamic-upf-selection-rl')]
    histories=[]
    exact_names={'dpdk_df_processed.csv','oai_df_processed.csv'}
    found=[]
    # Filesystem names are checked without treating ignored files as absent.
    command=['rg','--files','--hidden','--no-ignore','/home/ubuntu','/tmp',
       '-g','dpdk_df_processed.csv','-g','oai_df_processed.csv','-g','!**/.venv/**',
       '-g','!**/node_modules/**','-g','!**/.git/**']
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode not in (0,1,2):raise ValueError('Exact-name filesystem search failed')
    found=[p for p in result.stdout.splitlines() if Path(p).name in exact_names]
    for repo in repositories:
        for filename in ['dpdk_df_processed.csv','oai_df_processed.csv','dpdk_processed.csv','oai_processed.csv']:
            path='data/processed/'+filename
            log=subprocess.run(['git','log','--all','--reverse','--format=%H|%aI|%s','--',path],
                cwd=repo,capture_output=True,text=True,check=True).stdout.strip().splitlines()
            item={'repository':str(repo),'path':path,'local_history_entries':log}
            if log:
                commit=log[0].split('|')[0]
                blob=subprocess.run(['git','show',f'{commit}:{path}'],cwd=repo,capture_output=True,check=True).stdout
                frame=pd.read_csv(io.BytesIO(blob))
                item.update({'earliest_available_blob_sha256':hashlib.sha256(blob).hexdigest(),
                             'rows':len(frame),'columns':list(frame.columns),
                             'qos_w_present':'QoS_w' in frame,'qos_dd_present':'QoS_dd' in frame})
            histories.append(item)
    return {'exact_file_search_roots':['/home/ubuntu','/tmp'],'exact_named_files_found':found,
            'filesystem_search_returncode':result.returncode,
            'filesystem_search_diagnostics':result.stderr.splitlines(),
            'filesystem_search_complete':result.returncode in (0,1),
            'local_git_histories':histories,'remote_fetch_performed':False,
            'limitation':'Earliest locally recoverable occurrence is not proof of original creation time'}


def run(output):
    notebook=json.loads(NOTEBOOK.read_text());functions=notebook_functions(notebook)
    cells=[]
    image_dir=output/'notebook_outputs';image_dir.mkdir()
    for i,c in enumerate(notebook['cells']):
        outputs=[]
        for j,o in enumerate(c.get('outputs',[])):
            data=o.get('data',{})
            item={'type':o['output_type'],'text':''.join(o.get('text',[])),
                  'text_plain':''.join(data.get('text/plain',[])),'mime_types':list(data)}
            if 'image/png' in data:
                path=image_dir/f'cell_{i}_output_{j}.png'
                path.write_bytes(base64.b64decode(''.join(data['image/png'])))
                item['image_path']=str(path.relative_to(ROOT))
            outputs.append(item)
        cells.append({'index':i,'type':c['cell_type'],'execution_count':c.get('execution_count'),
                      'source':''.join(c.get('source',[])),'outputs':outputs})
    write_json(output/'notebook_inspection.json',{'path':str(NOTEBOOK),'sha256':sha256(NOTEBOOK),'cells':cells})
    lineage=source_history(output)
    provenance=[{'role':'recovered_notebook','path':str(NOTEBOOK),'sha256':sha256(NOTEBOOK)}]
    earlier=Path('/home/ubuntu/xRL_UpfSelection/notebooks/03_QoS.ipynb')
    provenance.append({'role':'different_notebook_examined_in_task2e','path':str(earlier),'sha256':sha256(earlier)})
    scores,rows,pca_details,direct,heads,copy_checks,component_effects,full_components=[],[],{},{},[],[],[],[]
    column_lists=re.findall(r'Index\((\[[\s\S]*?\]),\s*dtype=',
         ''.join(notebook['cells'][2]['outputs'][0]['text']))
    if len(column_lists)!=2:raise ValueError('Saved full column lists unavailable')
    displayed_columns=[ast.literal_eval(s) for s in column_lists]
    for mode,prefix,head_index in [('DPDK','dpdk',1),('USR','oai',3)]:
        path=DATA/f'{prefix}_processed.csv';frame=pd.read_csv(path)
        expected_columns=displayed_columns[0 if mode=='DPDK' else 1]
        if [c for c in frame.columns if c not in ('QoS_w','QoS_dd')]!=expected_columns:
            raise ValueError('Saved 28-column schema differs from later 30-column schema')
        provenance.append({'role':'later_training_csv','mode':mode,'path':str(path),'sha256':sha256(path)})
        if frame.duplicated(KEYS).any():raise ValueError('Lineage keys not unique')
        for project in ('xRL_UpfSelection','dynamic-upf-selection-rl'):
            copy=Path('/home/ubuntu')/project/'data/processed'/path.name
            cf=pd.read_csv(copy)
            match=frame.merge(cf,on=KEYS,suffixes=('_reference','_copy'),validate='one_to_one',how='outer',indicator=True)
            equal=sha256(copy)==sha256(path)
            if not equal or not (match['_merge']=='both').all():raise ValueError('Recovered training copies disagree')
            copy_checks.append({'mode':mode,'path':str(copy),'sha256':sha256(copy),'matched_keys':len(match),
                                'all_rows_and_values_byte_identical':equal})
        visible=displayed_head_checks(notebook,frame,head_index)
        heads.extend({'mode':mode,**r} for r in visible)
        if not all(r['matches_display_precision'] for r in visible):raise ValueError('Saved head differs from training prefix')
        try:
            functions['compute_component_scores'](frame)
        except KeyError as e:
            direct[mode]={'status':'missing_columns_as_written','first_error':str(e),
              'missing_expected_columns':[x for x in ['Avg DL Throughput (Mbps)','Avg Objective Throughput (Mbps)',
                      'Avg DL Delay (ms)','Avg DL Jitter (us)','Total Packet Loss (%)'] if x not in frame]}
        else:raise ValueError('Unexpected default notebook schema')
        candidates={}
        # Finite recorded-column/unit screen. Tx fields are explicit negative
        # controls for offered-vs-delivered confusion, not candidate final semantics.
        for throughput in ['RAN_Rx_Mbits/s','DN_Rx_Mbits/s','RAN_Tx_Mbits/s','DN_Tx_Mbits/s']:
            for direction in ['DL','UL']:
                for factor in (1.,100.):
                    S,mapping=components(frame,functions,direction,throughput,factor)
                    name=f'notebook_{throughput}_{direction}_loss_x{factor:g}'
                    candidates[name]=(functions['weighted_qos'](S),mapping)
        S,mapping=components(frame,functions)
        # Evidence-supported combined-direction interpretation, all in common units.
        b=functions['bounded_score_low_is_bad']
        combined=np.column_stack([
            functions['throughput_fulfillment']((frame['DN_Rx_Mbits/s']+frame['RAN_Rx_Mbits/s'])/2,
                frame['Objective Throughput (DL/UL)']),
            b((frame.downlink_delay_ms+frame.uplink_delay_ms)/2,40),
            b((frame.RAN_jitter+frame.DN_jitter)/2,10),
            b(100*(frame.RAN_PLR_interval+frame.DN_PLR_interval)/2,5)])
        candidates['notebook_combined_direction_mean']=(functions['weighted_qos'](combined),
             {'interpretation':'mean of direction-specific received Mbps, delay, jitter and loss fractions; equal per-direction objectives; caps/weights unchanged'})
        candidates['notebook_without_throughput_no_renormalization']=(S[:,1:]@np.asarray([.2,.2,.2]),
             {'interpretation':'component omission diagnostic; weighted residual must equal 0.4*S_th'})
        for direction,lat,jit,loss in [('DL','downlink_delay_ms','RAN_jitter','RAN_PLR_interval'),
                                     ('UL','uplink_delay_ms','DN_jitter','DN_PLR_interval')]:
            pred=.4*b(frame[lat],300)+.4*b(frame[jit]/1000,100)+.2*b(frame[loss],1e-6)
            candidates[f'manuscript_linear_{direction}']=(pred,{'weights':[.4,.4,.2],
               'mapping':[lat,jit,loss],'caps':[300,100,1e-6],
               'note':'Notebook-style bounded linear interpretation; manuscript does not completely specify component functions; raw jitter assumed us and converted to ms'})
        qdd,artifacts=functions['data_driven_qos'](S)
        candidates['notebook_pca_per_mode_all_rows']=(qdd,{'mapping':mapping,
            'fit':'StandardScaler over four component scores, PCA(1, random_state=0), sign toward S_th, MinMax over PC1, separately per mode on all rows'})
        pca=artifacts['pca'];z=artifacts['scaler_z'].transform(S)
        pc=pca.transform(z).ravel()
        pca_details[mode]={'standardizer_mean':artifacts['scaler_z'].mean_.tolist(),
            'standardizer_scale':artifacts['scaler_z'].scale_.tolist(),
            'components_before_sign_correction':pca.components_.tolist(),
            'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
            'correlation_pc1_with_throughput_score':float(np.corrcoef(pc,S[:,0])[0,1]),
            'minmax_min':artifacts['scaler_s'].min_.tolist(),'minmax_scale':artifacts['scaler_s'].scale_.tolist(),
            'comparison_to_stored_QoS_dd':differences(qdd,frame.QoS_dd),
            'new_surrogate_training':False,'historical_PCA_artifact_recovered':False}
        for name,(pred,mapping) in candidates.items():
            scores.append({'mode':mode,'candidate':name,'target':'QoS_w','mapping':mapping,**differences(pred,frame.QoS_w)})
            for i,value in enumerate(pred):
                rows.append({'mode':mode,'source_row':i,'candidate':name,'historical_qos_w':float(frame.QoS_w.iloc[i]),
                    'candidate_score':float(value),'difference':float(value-frame.QoS_w.iloc[i])})
        weighted=functions['weighted_qos'](S)
        for i,r in frame.iterrows():
            full_components.append({'mode':mode,'source_row':int(i),'timestamp':r.timestamp,
             'sweep':int(r['Test Set']),'test_case':r['Test Case'],'offered_load_mbps':float(r['Objective Throughput (DL/UL)']),
             'dn_receive_mbps':float(r['DN_Rx_Mbits/s']),'s_th':float(S[i,0]),'s_latency':float(S[i,1]),
             's_jitter':float(S[i,2]),'s_loss':float(S[i,3]),'historical_qos_w':float(r.QoS_w),
             'reconstructed_qos_w':float(weighted[i]),'historical_qos_dd':float(r.QoS_dd),'reconstructed_qos_dd':float(qdd[i])})
        affected=0;max_error=0.
        for key,block in frame.groupby(['Objective Throughput (DL/UL)','downlink_delay_ms','RAN_jitter','RAN_PLR_interval']):
            if len(block)<2 or block.QoS_w.max()-block.QoS_w.min()<=1e-8:continue
            indices=block.index.to_numpy();residue=block.QoS_w.to_numpy()-.4*S[indices,0]
            remaining_span=float(residue.max()-residue.min())
            affected+=1;max_error=max(max_error,remaining_span)
        component_effects.append({'mode':mode,'previous_same_DL_KPI_discrepancy_groups':affected,
            'maximum_within_group_label_span_after_subtracting_throughput_component':max_error,
            'interpretation':'The omitted throughput component accounts for the Task 2E same-DL-KPI label differences to floating precision'})
    lineage.update({'training_copy_checks':copy_checks,
       'saved_notebook_columns':displayed_columns,
       'later_training_columns_equal_saved_columns_plus_QoS_w_and_QoS_dd':True,
       'saved_head_checks':'Visible cells match later training rows 0..4 to their displayed precision',
       'prelabel_files_not_recovered':not bool(lineage['exact_named_files_found']),
       'full_prelabel_to_training_row_join_available':False})
    write_json(output/'data_lineage.json',lineage)
    write_json(output/'direct_execution_checks.json',direct)
    write_json(output/'candidate_results.json',scores)
    write_json(output/'pca_reproduction.json',pca_details)
    write_json(output/'omitted_throughput_explanation.json',component_effects)
    write_csv(output/'saved_head_comparisons.csv',heads)
    write_csv(output/'candidate_predictions.csv',rows)
    write_csv(output/'reconstructed_components.csv',full_components)
    write_json(output/'provenance.json',{'inputs':provenance,
       'dependencies':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn']},
       'scope':'Scoring/PCA reconstruction only; no saved surrogate fitting, label replacement or traffic access'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir=args.output_dir.resolve()
    if not args.output_dir.resolve().is_relative_to(ROOT/'revision'):raise ValueError('Diagnostics must remain under revision/')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    run(args.output_dir)


if __name__=='__main__':main()

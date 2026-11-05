"""Bounded composite-QoS inference; explicit artifacts, no default twin changes."""

import json
from pathlib import Path
import numpy as np

from revision.evaluation.twin import _runtime, sha256
from .neural_power import transformed_input
from .power_model import ProfilingPartition, LOAD, linear_basis

TARGET='QoS_w'


def create_qos_network(tf, specification, seed):
    tf.keras.utils.set_random_seed(seed)
    layers=[tf.keras.layers.Input(shape=(1,))]
    for i,width in enumerate(specification['hidden_units']):
        layers.append(tf.keras.layers.Dense(width,activation=specification['hidden_activation']))
        if i in specification['batch_norm_after_hidden_layers']:
            layers.append(tf.keras.layers.BatchNormalization())
        if specification.get('dropout_rate',0) and i==specification.get('dropout_after_hidden_layer'):
            layers.append(tf.keras.layers.Dropout(specification['dropout_rate'],seed=seed))
    layers.append(tf.keras.layers.Dense(1,activation='sigmoid'))
    return tf.keras.Sequential(layers)


class BoundedQoSModel:
    def __init__(self,manifest_path):
        self.manifest_path=Path(manifest_path).resolve()
        self.manifest_hash=sha256(self.manifest_path)
        self.metadata=json.loads(self.manifest_path.read_text())
        if (self.metadata['schema_version']!=1 or self.metadata['target']!='QoS_w'
                or self.metadata['target_transform']!='identity_score'
                or self.metadata['output_activation']!='sigmoid'):
            raise ValueError('Unsupported bounded QoS artifact')
        self.model_path=self.manifest_path.parent/self.metadata['model_file']
        if self.model_path.parent!=self.manifest_path.parent:
            raise ValueError('Model must be in the manifest directory')
        self.model_hash=sha256(self.model_path)
        if self.model_hash!=self.metadata['model_sha256']:raise ValueError('QoS artifact hash mismatch')
        _,tf,_,_=_runtime()
        self.network=tf.keras.models.load_model(self.model_path,compile=False)
        if self.network.layers[-1].get_config()['activation']!='sigmoid':raise ValueError('QoS head must be sigmoid')
        if any(not np.isfinite(w).all() for w in self.network.get_weights()):raise ValueError('Nonfinite network weights')
        self.name=self.metadata['name']

    def predict(self,loads):
        x=transformed_input(loads,self.metadata['input_scaler'],self.metadata['domain_mbps'])
        if not len(x):return np.empty(0,dtype=np.float64)
        result=np.asarray(self.network(x,training=False)).reshape(-1).astype(np.float64)
        if not np.isfinite(result).all() or np.any((result<0)|(result>1)):
            raise ValueError('Invalid QoS predictions; no clipping or repair')
        return result

    def verify_artifacts(self):
        if sha256(self.manifest_path)!=self.manifest_hash or sha256(self.model_path)!=self.model_hash:
            raise ValueError('QoS artifact changed after loading')


class SimpleQoSModel:
    def __init__(self,metadata):
        self.metadata=metadata;self.name=metadata['name']

    @classmethod
    def fit_training(cls,training:ProfilingPartition,name,domain):
        if training.role!='train':raise ValueError('Baseline fitting requires TRAIN groups only')
        frame=training.frame
        values=frame[TARGET].to_numpy()
        if not np.isfinite(values).all() or np.any((values<0)|(values>1)):
            raise ValueError('Invalid source QoS labels')
        metadata={'name':name,'domain_mbps':list(domain),'fit_sweeps':list(training.sweeps),
                  'fit_rows':len(frame),'target':'QoS_w'}
        if name=='training_mean_constant':metadata['mean']=float(values.mean())
        elif name=='training_load_mean_pwl':
            grouped=frame.groupby(LOAD)[TARGET].mean()
            metadata['loads_mbps']=grouped.index.tolist();metadata['scores']=grouped.tolist()
            if grouped.index.min()!=domain[0] or grouped.index.max()!=domain[1]:raise ValueError('Baseline domain not covered')
        else:raise ValueError('Unspecified simple baseline')
        return cls(metadata)

    def predict(self,loads):
        x=np.asarray(loads,dtype=np.float64).reshape(-1);low,high=self.metadata['domain_mbps']
        if not np.isfinite(x).all() or np.any((x<low)|(x>high)):raise ValueError('Outside declared QoS domain')
        if self.name=='training_mean_constant':return np.full(len(x),self.metadata['mean'])
        return linear_basis(x,self.metadata['loads_mbps'])@np.asarray(self.metadata['scores'])

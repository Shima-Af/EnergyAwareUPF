"""Versioned four-model revision twin. Historical default remains untouched."""

from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import numpy as np

from .twin import ROOT,RawPrediction,InvalidPrediction,sha256
from .contract import Outcome
from .data import fingerprint
from revision.profiling.power_model import PowerModel
from revision.profiling.neural_power import NeuralPowerModel
from revision.profiling.neural_qos import BoundedQoSModel

CONFIG=ROOT/'revision/configs/revised_twin_v1.json'


class RevisionTwin:
    """Conditional means, process-attributed watts and composite score only."""
    def __init__(self,config_path=CONFIG):
        self.config_path=Path(config_path).resolve();self.config=json.loads(self.config_path.read_text())
        if self.config['version']!='comcom.revised_twin.v1':raise ValueError('Unsupported revision twin version')
        self._pinned={self.config_path:sha256(self.config_path)};self.components={}
        for r in [self.config['qos_specification'],*self.config['implementations']]:
            path=ROOT/r['path']
            if sha256(path)!=r['sha256']:raise ValueError('Frozen score or implementation identity mismatch')
            self._pinned[path]=r['sha256']
        self._cache={}
        if set(self.config['models'])!={'dpdk_power','usr_power','dpdk_qos','usr_qos'}:raise ValueError('Exactly four selected models required')
        # Verify every manifest/weight identity before deserialization.
        for name,item in self.config['models'].items():
            for r in item['artifacts']:
                p=(ROOT/r['path']).resolve()
                if not p.is_relative_to(ROOT/'revision') or sha256(p)!=r['sha256']:raise ValueError('Revised artifact identity mismatch')
                self._pinned[p]=r['sha256']
        for name,item in self.config['models'].items():
            path=ROOT/item['path'];kind=item['kind']
            if name.endswith('qos'):
                if kind!='bounded_qos':raise ValueError('QoS component must use its frozen bounded model')
                model=BoundedQoSModel(path)
            elif kind=='simple_power':model=PowerModel.load(path)
            elif kind=='neural_power':model=NeuralPowerModel(path)
            else:raise ValueError('Unsupported power component')
            self.components[name]=model
        self.identity=fingerprint({'version':self.config['version'],'config_sha256':self._pinned[self.config_path],
             'adapter_sha256':sha256(Path(__file__)),
             'implementation_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [
               ROOT/'revision/profiling/neural_qos.py',ROOT/'revision/profiling/neural_power.py',ROOT/'revision/profiling/power_model.py']}})

    def raw_prediction(self,action,load_mbps):
        if isinstance(action,bool) or not isinstance(action,(int,np.integer)) or action not in (0,1,2):raise ValueError('Action must be DPDK=0, USR1=1 or USR2=2')
        load=float(load_mbps)
        if not math.isfinite(load) or load<0:raise ValueError('Invalid offered load')
        if (action,load) in self._cache:return self._cache[action,load]
        k=2 if action==2 else 1;per_load=load/k;mode='dpdk' if action==0 else 'usr'
        power=float(self.components[mode+'_power'].predict([per_load])[0])
        qos=float(self.components[mode+'_qos'].predict([per_load])[0])
        raw=RawPrediction(int(action),load,k,per_load,float(np.float32(per_load)),power,k*power,qos)
        self._cache[action,load]=raw
        return raw

    def outcome(self,action,load_mbps):
        r=self.raw_prediction(action,load_mbps)
        if r.failures:raise InvalidPrediction(r,self.identity)
        return Outcome(r.power_w,r.qos)

    def verify_artifacts(self):
        for path,digest in self._pinned.items():
            if sha256(path)!=digest:raise ValueError('Frozen twin artifact changed')
        for model in self.components.values():
            if hasattr(model,'verify_artifacts'):model.verify_artifacts()

    @property
    def manifest(self):
        return {'version':self.config['version'],'twin_id':self.identity,'config':self.config,
            'pinned_paths_and_hashes':{str(p):h for p,h in self._pinned.items()},
            'units':{'power':'process-attributed W','energy':'W multiplied by slot hours = Wh','qos':'composite QoS_w score'},
            'semantics':'balanced equal load; summed power; per-instance QoS unchanged; no transient energy or interference',
            'legacy_float32_model_load_field':'float32 representation for compatibility; actual transforms are pinned in each model manifest',
            'inference_profile':'CPU, deterministic, one instance-load per model call; no output clipping',
            'dependency_versions':{p:importlib.metadata.version(p) for p in ['numpy','tensorflow','keras','scikit-learn','scipy']},
            'default_historical_twin_changed':False,'operational_safety_guarantee':False}

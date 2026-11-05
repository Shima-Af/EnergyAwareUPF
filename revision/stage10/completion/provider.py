"""Balanced three-configuration KPI/power adapter, with no missing-KPI fallback."""
from revision.stage10.selected_policy import require_authenticated_targets
from .engine import PhysicalOutcome


TARGETS=('throughput_fraction','delay_ms','jitter_us','loss_fraction')


class FrozenKPIModel:
    def __init__(self,manifest):
        import json,joblib
        from pathlib import Path
        from revision.evaluation.twin import sha256
        path=Path(manifest);self.metadata=json.loads(path.read_text())
        model_path=path.parent/'model.joblib'
        if sha256(model_path)!=self.metadata['model_sha256']:raise ValueError('KPI artifact changed')
        self.model=joblib.load(model_path)

    def predict(self,loads):
        import numpy as np
        x=np.asarray(loads,float)
        if not np.isfinite(x).all() or (x<.01).any() or (x>5000).any():raise ValueError('Outside KPI domain')
        p=self.model.predict(np.log1p(x.reshape(-1,1)))
        if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid KPI output')
        if self.metadata['target']=='loss_fraction' and (p>1).any():raise ValueError('Invalid loss fraction')
        return p


class KPIProvider:
    def __init__(self,power_models,kpi_models,provenance):
        require_authenticated_targets(provenance)
        for mode in ['dpdk','usr']:
            if mode not in power_models:raise ValueError('Missing power model')
            if mode not in kpi_models or any(t not in kpi_models[mode] for t in TARGETS):
                raise ValueError('Missing authenticated KPI model; loss cannot be substituted')
        self.power_models=power_models;self.kpi_models=kpi_models;self.provenance=provenance

    def outcome(self,action,load):
        if action not in (0,1,2):raise ValueError('Invalid configuration')
        k=2 if action==2 else 1;mode='dpdk' if action==0 else 'usr';x=load/k
        if not .01<=x<=5000:raise ValueError('Outside declared per-instance domain')
        q=[float(self.kpi_models[mode][name].predict([x])[0]) for name in TARGETS]
        p=k*float(self.power_models[mode].predict([x])[0])
        return PhysicalOutcome(p,*q)

"""Shared causal environment, forecast ledger, policy adapter and fair rule."""
import csv,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from revision.evaluation.contract import State,Forecast,Observation,Outcome,actuate,account,observe,evaluate
from revision.evaluation.selection import Candidate
from revision.evaluation.data import fingerprint
from revision.evaluation.revised_twin import RevisionTwin
from revision.evaluation.twin import ROOT,sha256
from .common import CFG,OUT,MODELS,inputs


class ForecastLedger:
    def __init__(self,path=OUT/'forecasts.csv'):
        self.path=Path(path);self.digest=sha256(self.path)
        self.rows={int(r['index']):r for r in csv.DictReader(self.path.open())}
    def at(self,index,context):
        r=self.rows[index]
        if int(r['fit_stop_exclusive'])>index:raise ValueError('Forecast fit includes target/future labels')
        if r['target_timestamp']!=context.decision_at.isoformat() or r['observed_through']!=context.completed_interval_at.isoformat():raise ValueError('Forecast ledger time misalignment')
        p=float(r['forecast_mbps'])
        if not np.isfinite(p) or p<0:raise ValueError('Invalid forecast')
        return Forecast(context.decision_at,context.completed_interval_at,context.decision_at,p)


class CachedTwin:
    """Exact pinned revision predictions at authorized workload loads, no surrogates substituted."""
    def __init__(self):
        self.real=RevisionTwin();self.identity=self.real.identity
        path=ROOT/'revision/reports/final_twin/integration/workload_prediction_census.csv'
        self.census_sha256=sha256(path);self.table={}
        for r in csv.DictReader(path.open()):
            if r['dataset']=='netflix' and r['partition'] in ('train','validation'):
                if r['failures']:raise ValueError('Invalid frozen prediction')
                self.table[int(r['action']),float(r['load_mbps'])]=Outcome(float(r['power_w']),float(r['qos']))
        # Cache is numerical optimization only: validate all unique inputs against real frozen inference.
        for (a,load),value in self.table.items():
            if self.real.outcome(a,load)!=value:raise ValueError('Pinned cache differs from frozen twin')
    def outcome(self,action,load):
        key=(int(action),float(load))
        if key not in self.table:raise ValueError('Outcome outside authorized training/validation cache')
        return self.table[key]


def feature_names(schema='hybrid'):
    names=['completed_load','previous_qos','previous_power','previous_sec','previous_valid',
       'configuration_dpdk','configuration_usr1','configuration_usr2','cooldown',
       'time_sin','time_cos','weekday_sin','weekday_cos']
    if schema in ('history','hybrid'):names += [f'completed_history_{i}' for i in range(95)]
    if schema in ('forecast','hybrid'):names+=['causal_target_forecast']
    return names


def vector(observation):
    o=observation;p=o.previous_outcome;c=o.decision_calendar
    values=[o.current_load_mbps/400.,p.qos if p else 0.,p.power_w if p else 0.,
        p.power_w/max(o.current_load_mbps,1e-6) if p else 0.,float(p is not None),
        *[float(o.state.action==a) for a in range(3)],o.state.elapsed_slots/4.,
        c.time_sin,c.time_cos,c.weekday_sin,c.weekday_cos]
    if o.schema in ('history','hybrid'):values.extend(np.asarray(o.history_mbps)/400.)
    if o.schema in ('forecast','hybrid'):values.append(o.forecast.load_mbps/400.)
    result=np.asarray(values,np.float32)
    if result.shape!=(len(feature_names(o.schema)),) or not np.isfinite(result).all():raise ValueError('Invalid observation vector')
    return result


class ControlEnv(gym.Env):
    metadata={}
    def __init__(self,horizon,twin,ledger,contract,schema='hybrid'):
        if horizon.partition.role not in ('train','validation') or horizon.partition.timeline.usage!='development':raise ValueError('Pilot prohibits final-test or transfer environment')
        self.horizon=horizon;self.twin=twin;self.ledger=ledger;self.contract=contract;self.schema=schema
        self.indices=list(horizon.indices);self.action_space=spaces.Discrete(3)
        self.observation_space=spaces.Box(-np.inf,np.inf,(len(feature_names(schema)),),dtype=np.float32)
    def _obs(self):
        i=self.indices[self.position];ctx=self.horizon.context(i);forecast=self.ledger.at(i,ctx)
        return observe(ctx,self.schema,self.state,self.previous,forecast)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.position=0;self.previous=None
        self.state=State(self.contract.initial_action,self.contract.cooldown_slots)
        return vector(self._obs()),{}
    def step(self,action):
        i=self.indices[self.position];transition=actuate(self.state,int(action),self.contract)
        load=self.horizon.partition.timeline.load_mbps[i];outcome=self.twin.outcome(transition.executed,load)
        accounting=account(outcome,load,900.,transition,self.contract)
        self.state=transition.next_state;self.previous=outcome;self.position+=1
        terminated=self.position==len(self.indices)
        obs=np.zeros(self.observation_space.shape,np.float32) if terminated else vector(self._obs())
        info={'index':i,'timestamp':self.horizon.partition.timeline.timestamps[i].isoformat(),
             'transition':asdict(transition),'outcome':asdict(outcome),'accounting':asdict(accounting)}
        return obs,accounting.reward,terminated,False,info


class PolicyController:
    def __init__(self,model,normalizer,recurrent):
        self.model=model;self.normalizer=normalizer;self.recurrent=recurrent
        if normalizer.training:raise ValueError('Evaluation normalizer must be frozen')
    def reset(self,*,seed):
        self.state=None;self.starts=np.array([True]);self.calls=0;self.state_norms=[]
    def predict(self,observation,*,deterministic):
        if not deterministic:raise ValueError('Evaluation must be deterministic')
        obs=self.normalizer.normalize_obs(vector(observation).reshape(1,-1))
        if self.recurrent:
            action,self.state=self.model.predict(obs,state=self.state,episode_start=self.starts,deterministic=True)
            self.starts[:]=False
            self.state_norms.append(float(sum(np.linalg.norm(s) for s in self.state)))
        else:action,_=self.model.predict(obs,deterministic=True)
        self.calls+=1
        return int(np.asarray(action).ravel()[0])


class RuleController:
    """Forecast-conditioned three-action curve selector, with explicit relative-power deadband."""
    def __init__(self,twin):self.twin=twin
    def reset(self,*,seed):pass
    def predict(self,observation,*,deterministic):
        if not deterministic:raise ValueError('Rule requires deterministic evaluation')
        load=observation.forecast.load_mbps
        outcomes=[self.twin.outcome(a,load) for a in range(3)]
        feasible=[a for a,o in enumerate(outcomes) if o.qos>=CFG['rule']['target']]
        best=min(feasible,key=lambda a:(outcomes[a].power_w,a)) if feasible else min(range(3),key=lambda a:(-outcomes[a].qos,outcomes[a].power_w,a))
        current=observation.state.action
        if best!=current and current in feasible and best in feasible:
            saving=(outcomes[current].power_w-outcomes[best].power_w)/max(outcomes[current].power_w,1e-12)
            if saving<=CFG['rule']['relative_power_deadband']:return current
        return best


class LedgerForecaster:
    """Bridge into the existing runner; per-row fit cutoff checked by ledger.at."""
    fit=None
    def __init__(self,ledger,horizon):self.ledger=ledger;self.horizon=horizon;self.bytime={t:i for i,t in zip(horizon.indices,horizon.timestamps)}
    def predict(self,context):return self.ledger.at(self.bytime[context.decision_at],context)


def run_controller(horizon,controller,twin,ledger,contract,name,bundle_hash):
    if horizon.partition.role!='validation':raise ValueError('Controller reporting restricted to validation')
    candidate=Candidate(name,(('bundle',bundle_hash),('frozen_twin',twin.identity),('forecasts',ledger.digest)))
    result=evaluate(horizon,controller,twin,candidate=candidate,contract=contract,schema=CFG['schema'],
                    forecaster=LedgerForecaster(ledger,horizon))
    return result


def summarize(result):
    summary=dict(result.summary);steps=result.steps
    counts=np.bincount([s.transition.executed for s in steps],minlength=3)
    requested=np.bincount([s.transition.requested for s in steps],minlength=3)
    bad=[s.accounting.qos_shortfall for s in steps if s.accounting.qos_violation]
    summary.update({'action_counts':counts.tolist(),'requested_action_counts':requested.tolist(),
       'min_predicted_qos':min(s.outcome.qos for s in steps),'mean_predicted_qos':float(np.mean([s.outcome.qos for s in steps])),
       'violation_shortfall_mean_conditional':float(np.mean(bad)) if bad else None,
       'violation_shortfall_max':max(bad) if bad else 0.,
       'energy_reward_magnitude':sum(s.accounting.energy_term for s in steps),
       'qos_reward_magnitude':sum(s.accounting.qos_term for s in steps),
       'orchestration_reward_magnitude':sum(s.accounting.orchestration_term for s in steps),
       'horizon_identity':result.horizon.identity})
    return summary

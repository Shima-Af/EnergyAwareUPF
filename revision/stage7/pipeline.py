"""Causal W96 schemas and authorized validation-only transfer accounting."""
from dataclasses import asdict
import csv,json,time
import numpy as np
from gymnasium import spaces
from revision.evaluation.contract import State,Outcome,actuate,account,observe,Step,EvaluationResult
from revision.evaluation.selection import Candidate
from revision.stage3.pipeline import ForecastLedger,ControlEnv as PriorEnv,PolicyController as PriorController
from revision.stage3.pipeline import summarize
from revision.evaluation.validate_static import write_csv
from .common import CFG,ROOT,OUT,sha256


def feature_names(schema):
    if schema not in CFG['schemas']:raise ValueError('Unknown observation schema')
    names=['completed_load','previous_qos','previous_power','previous_sec','previous_valid',
      'configuration_dpdk','configuration_usr1','configuration_usr2','cooldown','time_sin','time_cos','weekday_sin','weekday_cos']
    if schema in ('history','hybrid'):names += [f'completed_history_{i}' for i in range(96)]
    if schema in ('forecast','hybrid'):names += ['causal_target_forecast']
    return names


def vector(o):
    p=o.previous_outcome;c=o.decision_calendar
    values=[o.current_load_mbps/400.,p.qos if p else 0.,p.power_w if p else 0.,
      p.power_w/max(o.current_load_mbps,1e-6) if p else 0.,float(p is not None),
      *[float(o.state.action==a) for a in range(3)],o.state.elapsed_slots/4.,c.time_sin,c.time_cos,c.weekday_sin,c.weekday_cos]
    if o.schema in ('history','hybrid'):
        if len(o.history_mbps)!=96:raise ValueError('Stage7 requires96 completed history loads')
        values.extend(np.asarray(o.history_mbps)/400.)
    if o.schema in ('forecast','hybrid'):
        if o.forecast is None:raise ValueError('Missing causal forecast')
        values.append(o.forecast.load_mbps/400.)
    result=np.asarray(values,np.float32)
    if result.shape!=(len(feature_names(o.schema)),) or not np.isfinite(result).all():raise ValueError('Invalid observation vector')
    return result


class Ledger(ForecastLedger):
    def at(self,index,context):
        r=self.rows[index]
        if int(r.get('selection_stop_exclusive',0))>index:raise ValueError('Provider selected using future labels')
        return super().at(index,context)


class CachedTwin:
    def __init__(self):
        self.meta=json.loads((OUT/'prepared.json').read_text());self.identity=self.meta['twin_id']
        path=OUT/'outcome_cache.csv'
        if sha256(path)!=self.meta['cache_sha256']:raise ValueError('Outcome cache changed')
        self.table={(int(r['action']),float(r['load_mbps'])):Outcome(float(r['power_w']),float(r['qos'])) for r in csv.DictReader(path.open())}
    def outcome(self,action,load):
        key=(int(action),float(load))
        if key not in self.table:raise ValueError('Outcome outside frozen authorized cache')
        return self.table[key]


class ControlEnv(PriorEnv):
    def __init__(self,horizon,twin,ledger,contract,schema):
        if horizon.partition.role=='train' and horizon.partition.timeline.name!='netflix':raise ValueError('No YouTube policy training')
        super().__init__(horizon,twin,ledger,contract,schema)
        self.observation_space=spaces.Box(-np.inf,np.inf,(len(feature_names(schema)),),dtype=np.float32)
    def _obs(self):
        i=self.indices[self.position];ctx=self.horizon.context(i)
        return vector(observe(ctx,self.schema,self.state,self.previous,self.ledger.at(i,ctx)))


    def reset(self,*,seed=None,options=None):
        import gymnasium as gym
        gym.Env.reset(self,seed=seed);self.position=0;self.previous=None
        self.state=State(self.contract.initial_action,self.contract.cooldown_slots)
        return self._obs(),{}
    def step(self,action):
        i=self.indices[self.position];transition=actuate(self.state,int(action),self.contract)
        load=self.horizon.partition.timeline.load_mbps[i];outcome=self.twin.outcome(transition.executed,load)
        accounting=account(outcome,load,self.horizon.partition.timeline.slot.total_seconds(),transition,self.contract)
        self.state=transition.next_state;self.previous=outcome;self.position+=1
        terminated=self.position==len(self.indices)
        obs=np.zeros(self.observation_space.shape,np.float32) if terminated else self._obs()
        info={'index':i,'timestamp':self.horizon.partition.timeline.timestamps[i].isoformat(),
              'transition':asdict(transition),'outcome':asdict(outcome),'accounting':asdict(accounting)}
        return obs,accounting.reward,terminated,False,info


class PolicyController(PriorController):
    def __init__(self,model,normalizer,recurrent,schema):
        super().__init__(model,normalizer,recurrent);self.schema=schema
    def reset(self,*,seed):
        super().reset(seed=seed);self.latency_ns=[]
    def predict(self,o,*,deterministic):
        if not deterministic or o.schema!=self.schema:raise ValueError('Policy schema/determinism mismatch')
        start=time.perf_counter_ns();obs=self.normalizer.normalize_obs(vector(o).reshape(1,-1))
        if self.recurrent:
            action,self.state=self.model.predict(obs,state=self.state,episode_start=self.starts,deterministic=True)
            self.starts[:]=False;self.state_norms.append(float(sum(np.linalg.norm(s) for s in self.state)))
        else:action,_=self.model.predict(obs,deterministic=True)
        result=int(np.asarray(action).ravel()[0]);self.calls+=1;self.latency_ns.append(time.perf_counter_ns()-start)
        return result


def run_controller(horizon,controller,twin,ledger,contract,name,bundle_hash,schema,*,transfer=False):
    # The old generic runner restricted transfer to final-test reporting. This new
    # scope explicitly authorizes YouTube *validation*, guarded by Netflix selections.
    workload=horizon.partition.timeline.name
    if horizon.partition.role!='validation':raise ValueError('Stage7 permits validation scoring only')
    if workload=='youtube':
        if not transfer or not (OUT/'all_netflix_selections_frozen.json').exists():raise ValueError('Transfer before Netflix selection freeze')
    elif workload!='netflix' or transfer:raise ValueError('Invalid report role')
    candidate=Candidate(name,(('bundle',bundle_hash),('twin',twin.identity),('forecast',ledger.digest)))
    controller.reset(seed=0);state=State(contract.initial_action,contract.cooldown_slots);previous=None;steps=[]
    for i in horizon.indices:
        ctx=horizon.context(i);forecast=ledger.at(i,ctx);o=observe(ctx,schema,state,previous,forecast)
        transition=actuate(state,controller.predict(o,deterministic=True),contract)
        load=horizon.partition.timeline.load_mbps[i];outcome=twin.outcome(transition.executed,load)
        accounting=account(outcome,load,horizon.partition.timeline.slot.total_seconds(),transition,contract)
        steps.append(Step(i,ctx,o.forecast,load,transition,outcome,accounting));state=transition.next_state;previous=outcome
    return EvaluationResult(candidate.identity,horizon,contract,schema,tuple(steps))


def export_steps(name,result,path,ledger):
    rows=[]
    for s in result.steps:
        rows.append({'controller':name,'schema':result.schema,'workload':result.horizon.partition.timeline.name,
          'timestamp':s.context.decision_at.isoformat(),'source_index':s.index,'actual_load_mbps':s.load_mbps,
          'available_forecast_mbps':float(ledger.rows[s.index]['forecast_mbps']),
          'observed_forecast_mbps':s.forecast.load_mbps if s.forecast else None,
          **{k:v for k,v in asdict(s.transition).items() if k!='next_state'},**asdict(s.outcome),**asdict(s.accounting)})
    write_csv(path,rows)

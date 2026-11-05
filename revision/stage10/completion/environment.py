"""Causal Gym adapter for the new kernel; no default KPI twin or final-test path."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from revision.evaluation.contract import State,causal_forecast


def features(schema):
    if schema not in ('instant','history','forecast','hybrid'):raise ValueError('Unknown schema')
    return 13+(96 if schema in ('history','hybrid') else 0)+(1 if schema in ('forecast','hybrid') else 0)


class ControlEnv(gym.Env):
    def __init__(self,horizon,engine,ledger,schema):
        if horizon.partition.role not in ('train','validation'):raise ValueError('Final testing requires a separately frozen campaign runner')
        if horizon.partition.role=='train' and horizon.partition.timeline.name!='netflix':raise ValueError('Netflix-only controller training')
        if horizon.spec.history_slots!=96:raise ValueError('Common W96 history required')
        self.horizon=horizon;self.engine=engine;self.ledger=ledger;self.schema=schema
        self.indices=tuple(horizon.indices)
        self.action_space=spaces.Discrete(3);self.observation_space=spaces.Box(-np.inf,np.inf,(features(schema),),np.float32)

    def observation(self):
        i=self.indices[self.position];c=self.horizon.context(i);cal=c.decision_calendar
        p=self.previous
        values=[c.history_mbps[-1]/400.,p['qos'] if p else 0.,
                p['total_wh']*4 if p else 0.,p['sec_j_per_mbit'] if p else 0.,float(p is not None),
                *[float(self.state.action==a) for a in range(3)],self.state.elapsed_slots/4,
                cal.time_sin,cal.time_cos,cal.weekday_sin,cal.weekday_cos]
        if self.schema in ('history','hybrid'):values.extend(np.asarray(c.history_mbps)/400.)
        if self.schema in ('forecast','hybrid'):
            f=self.ledger.at(i,c)
            if (f.issued_at!=c.decision_at or f.target_at!=c.decision_at or f.observed_through!=c.completed_interval_at):raise ValueError('Noncausal or misaligned forecast')
            if not np.isfinite(f.load_mbps) or f.load_mbps<0:raise ValueError('Invalid forecast')
            values.append(f.load_mbps/400.)
        return np.asarray(values,np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.position=0;self.state=State(0,4);self.previous=None
        return self.observation(),{}

    def step(self,action):
        i=self.indices[self.position];load=self.horizon.partition.timeline.load_mbps[i]
        result=self.engine.step(self.state,int(action),load)
        self.state=result['transition'].next_state;self.previous=result;self.position+=1
        terminated=self.position==len(self.indices)
        obs=np.zeros(self.observation_space.shape,np.float32) if terminated else self.observation()
        return obs,result['reward'],terminated,False,dict(source_index=i,actual_load_mbps=load,**result)

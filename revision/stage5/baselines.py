"""Distinct request rules. All execution timing belongs to the shared actuator."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq


def utility(qos, power, *, performance_weight, energy_weight, minimum, maximum,
            score_scale=100., bounded_efficiency=True):
    if not maximum > minimum:
        raise ValueError('Utility normalization requires a positive power range')
    efficiency=100.*(maximum-power)/(maximum-minimum)
    if bounded_efficiency:
        # This bounds a published utility score, never a twin power/QoS output.
        efficiency=min(100., max(0., efficiency))
    return performance_weight*score_scale*qos + energy_weight*efficiency


@dataclass(frozen=True)
class Region:
    left: float
    right: float
    action: int


class SlicesExtendedManual:
    """Paper-inspired explicit three-action extension; not historical reproduction."""
    def __init__(self,twin,regions,minimum,maximum,margin=26.):
        self.twin=twin;self.regions=tuple(Region(**r) if isinstance(r,dict) else r for r in regions)
        if not self.regions or margin<0:raise ValueError('Invalid region definition')
        for i,r in enumerate(self.regions):
            if r.action not in (0,1,2) or r.left>=r.right:raise ValueError('Invalid region')
            if i and self.regions[i-1].right!=r.left:raise ValueError('Noncontiguous regions')
        self.minimum=minimum;self.maximum=maximum;self.margin=margin
    def reset(self,*,seed):pass
    def utilities(self,load):
        return [utility(o.qos,o.power_w,performance_weight=.2,energy_weight=.8,
                        minimum=self.minimum,maximum=self.maximum)
                for o in [self.twin.outcome(a,load) for a in range(3)]]
    def request(self,load,current):
        if not math.isfinite(load) or not self.regions[0].left<=load<=self.regions[-1].right:
            raise ValueError('Forecast outside declared manual-rule profiling domain')
        values=self.utilities(load);best=max(range(3),key=lambda a:(values[a],-a))
        if best==current:return current
        # At a crossover, retain execution state irrespective of numerical tie rounding.
        for i,r in enumerate(self.regions):
            if r.left<=load<=r.right and r.action==best:
                if i and load-r.left<=self.margin:return current
                if i<len(self.regions)-1 and r.right-load<=self.margin:return current
                return best
        raise ValueError('Utility winner inconsistent with frozen crossover map')
    def predict(self,observation,*,deterministic):
        if not deterministic:raise ValueError('Deterministic rule required')
        return self.request(observation.forecast.load_mbps,observation.state.action)


class ComcomHistorical:
    """Actual two-option inline ranking; no plotted-margin enforcement."""
    def __init__(self,twin):self.twin=twin
    def reset(self,*,seed):pass
    def utilities(self,load):
        return [utility(o.qos,o.power_w,performance_weight=.3,energy_weight=.7,
                        minimum=0.,maximum=3.58,score_scale=1.,bounded_efficiency=False)
                for o in [self.twin.outcome(a,load) for a in (0,1)]]
    def predict(self,observation,*,deterministic):
        if not deterministic:raise ValueError('Deterministic rule required')
        values=self.utilities(observation.forecast.load_mbps)
        return 0 if values[0]>=values[1] else 1


def profile_regions(grid,values,utility_at,xtol=1e-5):
    """Bracket every sampled winner change, refine it, preserve nonmonotone regions.

    A finite profiling grid cannot certify absence of narrower unsampled regions.
    Runtime rejects any observed winner that contradicts the frozen region map.
    """
    grid=np.asarray(grid,float);values=np.asarray(values,float)
    if len(grid)<2 or np.any(np.diff(grid)<=0) or values.shape[0]!=len(grid) or not np.isfinite(values).all():
        raise ValueError('Invalid profiling grid')
    winners=np.argmax(values,axis=1);regions=[];left=float(grid[0]);current=int(winners[0])
    for i in range(1,len(grid)):
        nxt=int(winners[i])
        if nxt==current:continue
        old=current
        boundary=float(brentq(lambda x:utility_at(x)[old]-utility_at(x)[nxt],
                              grid[i-1],grid[i],xtol=xtol))
        regions.append(Region(left,boundary,old));left=boundary;current=nxt
    regions.append(Region(left,float(grid[-1]),current))
    return regions

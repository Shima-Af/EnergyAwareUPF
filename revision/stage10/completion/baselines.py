"""New-scope adapters preserving legacy A/C principles; common Engine accounting.

Region maps and power bounds must be rebuilt from authenticated new KPI curves.
No historical crossovers or default loss values are substituted here.
"""
from revision.evaluation.contract import Outcome
from revision.stage5.baselines import SlicesExtendedManual,ComcomHistorical,profile_regions


class PointScoreTwin:
    def __init__(self,engine):self.engine=engine
    def outcome(self,a,load):
        physical=self.engine.provider.outcome(a,load)
        return Outcome(physical.power_w,physical.policy()['qos'])


class Static:
    def __init__(self,action):
        if action not in (0,1,2):raise ValueError('Invalid static configuration')
        self.action=action
    def reset(self,*,seed):pass
    def predict(self,observation,*,deterministic):
        if not deterministic:raise ValueError('Deterministic baseline required')
        return self.action


class Strong:
    """Forecast-only, transition-feasible minimum energy, original 2% deadband."""
    def __init__(self,engine):self.engine=engine
    def reset(self,*,seed):pass
    def predict(self,observation,*,deterministic):
        if not deterministic:raise ValueError('Deterministic baseline required')
        current=observation.state.action;load=observation.forecast.load_mbps
        candidates=[self.engine.step(observation.state,a,load) for a in range(3)]
        feasible=[a for a,s in enumerate(candidates) if s['compliant']]
        best=min(feasible,key=lambda a:(candidates[a]['total_wh'],a)) if feasible else min(range(3),key=lambda a:(candidates[a]['severity'],candidates[a]['total_wh'],a))
        hold=candidates[current]
        if hold['compliant'] and (hold['total_wh']-candidates[best]['total_wh'])/max(hold['total_wh'],1e-12)<=.02:
            return current
        return best


def legacy_rules(engine,*,regions,power_minimum,power_maximum,training_margin):
    twin=PointScoreTwin(engine)
    return {'historical_comcom':ComcomHistorical(twin),
            'extended_slices_manual':SlicesExtendedManual(twin,regions,power_minimum,power_maximum,margin=26.),
            'slices_training_recalibrated':SlicesExtendedManual(twin,regions,power_minimum,power_maximum,margin=training_margin),
            'strong_diagnostic':Strong(engine),**{f'static_{a}':Static(a) for a in range(3)}}

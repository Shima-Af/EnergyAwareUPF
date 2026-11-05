"""Transition-aware kernel shared by controllers and exact oracles.

No fitted KPI provider is substituted. A real provider must authenticate its
four targets; synthetic providers are accepted only with an explicit test flag.
"""
from dataclasses import dataclass
import math
from revision.evaluation.contract import Contract,actuate
from revision.stage10.primitives import KPIs,activation_burst_wh,transition_energy_wh
from revision.stage10.selected_policy import selected_qos,require_authenticated_targets


@dataclass(frozen=True)
class PhysicalOutcome:
    power_w: float
    received_fraction: float
    delay_ms: float
    jitter_us: float
    loss_fraction: float

    def __post_init__(self):
        if any(not math.isfinite(v) or v<0 for v in vars(self).values()) or self.loss_fraction>1:
            raise ValueError('Invalid physical prediction; no clipping')

    def policy(self):
        return selected_qos(received_mbps=self.received_fraction,offered_mbps=1.,
                            delay_ms=self.delay_ms,jitter_us=self.jitter_us,loss_fraction=self.loss_fraction)


@dataclass(frozen=True)
class Dynamics:
    dpdk_idle_w: float
    usr_idle_w: float
    drain_seconds: float  # Required explicit assumption, no invented measurement.
    dpdk_seconds: float=24.
    usr_seconds_per_instance: float=3.3
    dpdk_burst_ratio: float=1.73
    usr_burst_ratio: float=.37
    slot_seconds: float=900.

    def __post_init__(self):
        if any(not math.isfinite(v) or v<0 for v in vars(self).values()):raise ValueError('Invalid dynamics')
        if self.slot_seconds<=0 or max(self.dpdk_seconds,2*self.usr_seconds_per_instance,self.drain_seconds)>self.slot_seconds:
            raise ValueError('Activation/drain does not fit interval')


@dataclass(frozen=True)
class Reward:
    energy_scale_wh: float
    role: str
    training_identity: str

    def __post_init__(self):
        if self.role!='train' or not self.training_identity or not math.isfinite(self.energy_scale_wh) or self.energy_scale_wh<=0:
            raise ValueError('Reward scaling requires identified positive training-only energy bound')

    def terms(self,energy,violation,severity,changed):
        # One formula. For E<=training bound, safe immediate cost <=1.01;
        # unsafe cost >=3, regardless of low offered load. No SEC denominator.
        return dict(energy_cost=energy/self.energy_scale_wh,strict_qos_cost=3.*violation,
                    severity_cost=severity/(1+severity),stability_cost=.01*changed)


class Engine:
    def __init__(self,provider,dynamics,reward,*,synthetic=False):
        if not synthetic:require_authenticated_targets(provider.provenance)
        elif getattr(provider,'synthetic',False) is not True:raise ValueError('Explicit synthetic fixture required')
        self.provider=provider;self.dynamics=dynamics;self.reward=reward;self.contract=Contract()

    def step(self,state,request,load):
        if not math.isfinite(load) or load<=0:raise ValueError('Positive domain load required')
        tr=actuate(state,request,self.contract);d=self.dynamics
        changed=tr.executed!=state.action;duration=0.;burst=0.
        if changed:
            if tr.executed==0:
                duration=d.dpdk_seconds
                burst=activation_burst_wh(burst_ratio=d.dpdk_burst_ratio,attributed_idle_w=d.dpdk_idle_w,
                       per_instance_seconds=duration,launched_instances=1)
            elif tr.usr_instances_launched:
                # Sequential launch convention; existing/outgoing capacity serves
                # until all requested replicas are ready. Parallel launch is not assumed.
                duration=tr.usr_instances_launched*d.usr_seconds_per_instance
                burst=activation_burst_wh(burst_ratio=d.usr_burst_ratio,attributed_idle_w=d.usr_idle_w,
                       per_instance_seconds=d.usr_seconds_per_instance,launched_instances=tr.usr_instances_launched)
            else:duration=d.drain_seconds
        old=self.provider.outcome(state.action,load);new=self.provider.outcome(tr.executed,load)
        phases=[]
        if duration>0:phases.append((duration,old.policy()))
        if duration<d.slot_seconds:phases.append((d.slot_seconds-duration,new.policy()))
        compliant=all(p['compliant'] for _,p in phases)
        severity=max(p['severity'] for _,p in phases)
        energy=transition_energy_wh(outgoing_w=old.power_w,incoming_w=new.power_w,
                    outgoing_seconds=duration,slot_seconds=d.slot_seconds,activation_wh=burst)
        # Under the explicit serial-launch/barrier-cutover convention, earlier
        # ready replicas wait idle for later ones. Account for that modeled
        # overlap using the declared portable idle parameter, never loaded power.
        launches=tr.usr_instances_launched
        ready_idle=d.usr_idle_w*d.usr_seconds_per_instance*launches*(launches-1)/2/3600
        energy['operational_wh']+=ready_idle;energy['total_wh']+=ready_idle
        terms=self.reward.terms(energy['total_wh'],not compliant,severity,changed)
        return dict(transition=tr,**energy,reward=-sum(terms.values()),reward_components=terms,
            ready_idle_wh=ready_idle,
            compliant=compliant,severity=severity,qos=min(p['qos'] for _,p in phases),
            kpi_violations=tuple(any(p['kpi_violations'][i] for _,p in phases) for i in range(4)),
            unsafe_serving_seconds=sum(s for s,p in phases if not p['compliant']),
            startup_delay_exposure_seconds=duration if tr.usr_instances_launched or (changed and tr.executed==0) else 0.,
            outgoing_serving_seconds=duration,reconfiguration_violation=changed and not compliant,
            sec_j_per_mbit=energy['total_wh']*3600/(load*d.slot_seconds),incoming=new)

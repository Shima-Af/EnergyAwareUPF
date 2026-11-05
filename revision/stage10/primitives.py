"""Units-explicit primitives, without invented budgets or a replacement twin.

No physical target mapping, service budget, idle power or draining duration is
defaulted here. These require the documented Stage 10 scientific dependencies.
"""
from dataclasses import dataclass
import math


def finite_nonnegative(*values):
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Finite nonnegative quantities required; no repair')


@dataclass(frozen=True)
class Budgets:
    throughput_min_fraction: float
    delay_ms: float
    jitter_us: float
    loss_limit: float
    loss_units: str
    loss_sampling_seconds: float | None

    def __post_init__(self):
        values=(self.throughput_min_fraction,self.delay_ms,self.jitter_us,self.loss_limit)
        finite_nonnegative(*values)
        if any(v == 0 for v in values) or self.throughput_min_fraction > 1:
            raise ValueError('Positive normalized budgets required; zero-loss policy needs separate severity scale')
        if self.loss_units not in ('fraction','packets_per_sample'):
            raise ValueError('Explicit physical loss units required')
        if self.loss_units == 'fraction' and self.loss_limit > 1:
            raise ValueError('Loss fraction budget exceeds one')
        if self.loss_units == 'packets_per_sample':
            if self.loss_sampling_seconds is None or not math.isfinite(self.loss_sampling_seconds) or self.loss_sampling_seconds <= 0:
                raise ValueError('Packet-count loss budget requires its sample duration')
        elif self.loss_sampling_seconds is not None:
            raise ValueError('Do not imply a packet-count sample for a loss fraction')


@dataclass(frozen=True)
class KPIs:
    throughput_fraction: float
    delay_ms: float
    jitter_us: float
    loss: float

    def __post_init__(self):
        finite_nonnegative(self.throughput_fraction,self.delay_ms,self.jitter_us,self.loss)


def compliance(kpis: KPIs, budgets: Budgets):
    """Conjunction of budgets; maximum normalized exceedance, no compensation.

    Throughput shortfall is normalized to its required minimum; the other
    exceedances are normalized to their upper budgets. No .90 score threshold.
    This is a proposed mathematical primitive, not a calibrated safety signal.
    """
    if budgets.loss_units == 'fraction' and kpis.loss > 1:
        raise ValueError('Loss fraction exceeds one')
    strict=(kpis.throughput_fraction < budgets.throughput_min_fraction,
            kpis.delay_ms > budgets.delay_ms, kpis.jitter_us > budgets.jitter_us,
            kpis.loss > budgets.loss_limit)
    exceed=(max(0.,1-kpis.throughput_fraction/budgets.throughput_min_fraction),
            max(0.,kpis.delay_ms/budgets.delay_ms-1),
            max(0.,kpis.jitter_us/budgets.jitter_us-1),
            max(0.,kpis.loss/budgets.loss_limit-1))
    return {'kpi_violations':strict,'compliant':not any(strict),
            'normalized_exceedances':exceed,'severity':max(exceed)}


def activation_burst_wh(*, burst_ratio, attributed_idle_w, per_instance_seconds, launched_instances):
    """Thesis ratio model: sum once per launched instance, in Wh.

    Caller must authenticate/provide idle watts and the launch convention.
    No loaded power proxy, increment (k-1), or fabricated zero-load observation.
    """
    finite_nonnegative(burst_ratio,attributed_idle_w,per_instance_seconds)
    if type(launched_instances) is not int or launched_instances < 0:
        raise ValueError('Nonnegative integer launch count required')
    return burst_ratio*attributed_idle_w*per_instance_seconds*launched_instances/3600


def transition_energy_wh(*, outgoing_w, incoming_w, outgoing_seconds, slot_seconds, activation_wh):
    """Operational integration plus separately logged activation energy once.

    Serving duration must be supplied by the resolved launch/draining model.
    Activation energy is not also charged as a separate reward penalty here.
    """
    finite_nonnegative(outgoing_w,incoming_w,outgoing_seconds,slot_seconds,activation_wh)
    if slot_seconds <= 0 or outgoing_seconds > slot_seconds:
        raise ValueError('Serving duration outside control interval')
    operational=(outgoing_w*outgoing_seconds+incoming_w*(slot_seconds-outgoing_seconds))/3600
    return {'operational_wh':operational,'activation_wh':activation_wh,
            'total_wh':operational+activation_wh}


def serving_compliance(outgoing, incoming, budgets, *, outgoing_seconds, slot_seconds):
    """Worst serving-phase severity; never conceal an unsafe startup interval."""
    finite_nonnegative(outgoing_seconds,slot_seconds)
    if slot_seconds <= 0 or outgoing_seconds > slot_seconds:
        raise ValueError('Serving duration outside interval')
    phases=[]
    if outgoing_seconds > 0:phases.append(compliance(outgoing,budgets))
    if outgoing_seconds < slot_seconds:phases.append(compliance(incoming,budgets))
    return {'compliant':all(p['compliant'] for p in phases),
            'severity':max(p['severity'] for p in phases),
            'kpi_violations':tuple(any(p['kpi_violations'][i] for p in phases) for i in range(4))}
